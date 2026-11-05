# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Analytics of the Harvard EEG Database (HEEDB) next to the TUH corpora.
# ---------------------------------------------------------
"""Tables and signal checks behind ``notebooks/heedb_data_report.ipynb``.

Every table is computed the same way for HEEDB (``dataset_maker.heedb_index``)
and for TUH (``dataset_maker.tuh_index`` + ``tuh_clinical``), on columns the
two indexes share, so the report reads as a side-by-side comparison. Signal
quality is measured with the TUH QC helpers (``labram.eval.preprocessing_qc``)
on short segments fetched by S3 byte-range reads: an EDF stores fixed-size data
records after its header, so ``[start, start + dur)`` is one contiguous byte
range and a two-minute excerpt of a 90 MB file costs ~3 MB.
"""

import math
import os
import tempfile
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from dataset_maker import heedb_index as hx
from dataset_maker import tuh_clinical
from labram.data.tuh_metadata import (TUH_CHANNELS, bids_canonical_channel, channel_map,
                                      parse_edf_signal_header)
from labram.eval import preprocessing_qc as qc

TUH_INDEX = "/data/datasets/EEG-public/TUH_EEG/index/tuh_index.parquet"
TUH_SHEET = "/data/datasets/EEG-public/TUH_EEG/index/nedc/metadata_v12.xlsm.xlsx"
TUH_QC = "/data/datasets/EEG-public/TUH_EEG/qc_sample/qc_results.pkl"
DECADES = list(range(0, 100, 10))
FLAT_STD_UV = 0.5      # a channel is flat when its std is below this (µV)
PHYSIOLOGICAL_UV = (2.0, 150.0)   # median channel std of a probe that looks like EEG (µV)


# ------------------------------------------------------------------ loading
def load_tuh(index_path: str = TUH_INDEX, sheet_path: str = TUH_SHEET,
             cache: Optional[str] = None) -> pd.DataFrame:
    """TUH recording index joined with the NEDC clinical sheet
    (``tuh_clinical.join_index``), plus ``corpus`` = 'tuh' and ``cohort`` =
    'TUAB' / 'TUEG' (the copy the index uses). Cached as parquet if asked."""
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    index = pd.read_parquet(index_path)
    df = tuh_clinical.join_index(index, tuh_clinical.load_sheet(sheet_path))
    df["corpus"] = "tuh"
    df["cohort"] = np.where(df["source"].eq("tuab"), "TUAB", "TUEG")
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


# Small-cell suppression: HEEDB is credentialed health data, so published
# tables show no count of 1-9 individuals and no statistic computed on them.
SMALL_CELL = 10


def suppress_small(df: pd.DataFrame, cols=None, k: int = SMALL_CELL) -> pd.DataFrame:
    """Copy of ``df`` with counts 1..k-1 in ``cols`` (default: all numeric
    columns) replaced by the string ``'<k'``; zeros and counts >= k are kept."""
    out = df.copy()
    for col in (cols if cols is not None else out.select_dtypes("number").columns):
        v = pd.to_numeric(out[col], errors="coerce")
        small = v.gt(0) & v.lt(k)
        if small.any():
            out[col] = out[col].astype(object)
            out.loc[small, col] = f"<{k}"
    return out


# ------------------------------------------------------------ summary tables
def describe_age(age: pd.Series) -> Dict[str, float]:
    a = pd.to_numeric(age, errors="coerce").dropna()
    return {"n": int(a.size), "mean": a.mean(), "sd": a.std(), "p5": a.quantile(.05),
            "median": a.median(), "p95": a.quantile(.95)}


def overview(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One column per named frame: recordings, patients, hours, age, sex,
    report-labelled share and abnormal rate (Eqs. (1.1)-(1.3) of the notebook)."""
    rows = {}
    for name, df in frames.items():
        lab = df["abnormal_report"].dropna().astype(bool) if "abnormal_report" in df else pd.Series(dtype=bool)
        age = describe_age(df["age"])
        rows[name] = {
            "recordings": len(df), "patients": df["subject"].nunique(),
            "hours": df["duration_sec"].sum() / 3600,
            "median minutes": df["duration_sec"].median() / 60,
            "age mean": age["mean"], "age sd": age["sd"],
            "female %": 100 * df["sex"].eq("F").sum() / max(df["sex"].isin(["M", "F"]).sum(), 1),
            "with report label %": 100 * lab.size / max(len(df), 1),
            "abnormal % (of labelled)": 100 * lab.mean() if lab.size else np.nan,
        }
    return pd.DataFrame(rows)


def site_service_table(idx: pd.DataFrame) -> pd.DataFrame:
    """Sessions, patients and hours per site x service (NA service -> 'n/a')."""
    df = idx.assign(service=idx["service"].fillna("n/a"))
    t = df.groupby(["site", "service"]).agg(
        sessions=("session", "size"), patients=("subject", "nunique"),
        hours=("duration_sec", lambda s: s.sum() / 3600),
        median_min=("duration_sec", lambda s: s.median() / 60))
    return t.reset_index()


def duration_quantiles(groups: Dict[str, pd.Series], qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
    """Recording length (minutes) quantiles per named group."""
    rows = {name: {f"p{int(q * 100)}": (s.dropna() / 60).quantile(q) for q in qs} | {"n": int(s.notna().sum())}
            for name, s in groups.items()}
    return pd.DataFrame(rows).T


def age_by_decade(groups: Dict[str, pd.Series]) -> pd.DataFrame:
    """Share (%) of recordings per age decade (Eq. (3.1))."""
    out = {}
    for name, age in groups.items():
        d = pd.Series(hx.decade(age.dropna()))
        out[name] = 100 * d.value_counts(normalize=True).reindex(DECADES, fill_value=0)
    t = pd.DataFrame(out)
    t.index = [f"{d}–{d + 9}" if d < 90 else "90+" for d in t.index]
    return t


def abnormal_rate_by_decade(groups: Dict[str, pd.DataFrame], min_n: int = 30) -> pd.DataFrame:
    """Abnormal rate (%) per age decade among report-labelled recordings
    (Eq. (4.1)); cells with fewer than ``min_n`` recordings are NaN."""
    out = {}
    for name, df in groups.items():
        lab = df[df["abnormal_report"].notna()]
        dec = pd.Series(hx.decade(lab["age"]), index=lab.index)
        g = lab["abnormal_report"].astype(float).groupby(dec)
        rate = 100 * g.mean()
        rate[g.size() < min_n] = np.nan
        out[name] = rate.reindex(DECADES)
    t = pd.DataFrame(out)
    t.index = [f"{d}–{d + 9}" if d < 90 else "90+" for d in t.index]
    return t


def finding_table(idx: pd.DataFrame) -> pd.DataFrame:
    """Per report finding: sessions mentioning it by provenance (report /
    machine annotation / verified), share of report-labelled sessions with it
    from the report (Eq. (4.2)), mean age and abnormal rate of those sessions."""
    lab = idx[idx["abnormal_report"].notna()]
    rows = []
    for col in [c for c in idx.columns if c.startswith("rf_") and c not in ("rf_normal", "rf_abnormal")]:
        prov = lab[col].fillna("")
        name = col[3:]
        flag = lab.get(f"find_{name}")
        pos = lab[flag.fillna(False).astype(bool)] if flag is not None else lab.iloc[:0]
        rows.append({
            "finding": name,
            "group": next((g for g, m in hx.FINDING_GROUPS.items() if name in m), "other"),
            "report": int(prov.str.contains(r"\breport\b").sum()),
            "annotation": int(prov.str.contains(r"\bannotation\b").sum()),
            "verified": int(prov.str.contains(r"\bverified\b").sum()),
            "report %": 100 * len(pos) / max(len(lab), 1),
            "mean age": pos["age"].mean() if len(pos) >= SMALL_CELL else np.nan,
            "abnormal %": 100 * pos["abnormal_report"].astype(float).mean() if len(pos) >= SMALL_CELL else np.nan,
        })
    return pd.DataFrame(rows).sort_values("report", ascending=False).reset_index(drop=True)


def condition_table(df: pd.DataFrame, flags: Sequence[str], min_n: int = 20) -> pd.DataFrame:
    """Per clinical flag (``tuh_clinical`` column names), over recordings where
    the flag is known: n recordings / patients, prevalence (Eq. (5.1)), mean age,
    age difference to the rest (Eq. (5.2)) and abnormal rate (Eq. (5.3))."""
    rows = []
    for col in flags:
        if col not in df:
            rows.append({"flag": col})
            continue
        known = df[df[col].notna()]
        pos = known[known[col].astype(bool)]
        neg = known[~known[col].astype(bool)]
        lab = pos["abnormal_report"].dropna().astype(float) if "abnormal_report" in pos else pd.Series(dtype=float)
        rows.append({
            "flag": col, "known": len(known), "n": len(pos), "patients": pos["subject"].nunique(),
            "prevalence %": 100 * len(pos) / max(len(known), 1),
            "mean age": pos["age"].mean() if len(pos) >= min_n else np.nan,
            "age vs rest": pos["age"].mean() - neg["age"].mean() if len(pos) >= min_n else np.nan,
            "abnormal %": 100 * lab.mean() if len(lab) >= min_n else np.nan,
        })
    return pd.DataFrame(rows).set_index("flag")


def crosswalk_table() -> pd.DataFrame:
    """How each column of the NEDC TUEG sheet (``tuh_clinical``) is rebuilt
    from HEEDB, and whether the match is exact, approximate or missing."""
    rows = []
    for col in tuh_clinical.DIAGNOSES.values():
        if col == "dx_electrographic_seizure":
            src, kind = "report finding `seizure` (report/verified)", "exact"
        elif col in hx.ICD_TO_TUH:
            src, kind = "ICD-10 " + ", ".join(hx.ICD_TO_TUH[col]), "approx. (per patient, undated)"
        else:
            src, kind = "—", "missing"
        rows.append((col, "diagnosis", src, kind))
    for col, members in hx.FINDING_TO_TUH.items():
        rows.append((col, "EEG finding", "report finding " + ", ".join(members),
                     "exact" if hx.FINDING_TO_TUH_EXACT[col] else "approx. (no frontal/temporal split)"))
    rows.append(("f_oirda", "EEG finding", "—", "missing"))
    for col, pre in hx.ATC_TO_TUH.items():
        rows.append((col, "medication", "ATC " + ", ".join(pre), "approx. (ever prescribed, not at EEG)"))
    rows += [("abnormal_report", "report", "report `abnormal` (vs `normal`)", "exact"),
             ("age", "demographics", "AgeAtVisit (session)", "exact"),
             ("sex", "demographics", "SexDSC", "exact")]
    return pd.DataFrame(rows, columns=["TUH column", "kind", "HEEDB source", "match"])


def normative_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Recordings per age decade under nested normative rules: report-normal
    18-89 y; + no *known* neurological diagnosis (no ICD record passes); + no
    neurological diagnosis with an ICD record (``any_neuro_dx`` False); + not
    sedated, i.e. the full TUH rule (``tuh_clinical.normative_mask``). HEEDB
    medications are lifetime, so the last term excludes nearly everyone."""
    strict = tuh_clinical.normative_mask(df)
    relaxed = (df["in_sheet"] & ~df["abnormal_report"].fillna(True).astype(bool)
               & ~df["any_neuro_dx"].fillna(True).astype(bool) & df["age"].between(18, 89))
    report_only = df["in_sheet"] & ~df["abnormal_report"].fillna(True).astype(bool) & df["age"].between(18, 89)
    # No *known* neurological diagnosis: a patient without any ICD record passes.
    no_known = report_only & ~df["any_neuro_dx"].fillna(False).astype(bool)
    out = {}
    for name, mask in {"report-normal, 18–89": report_only, "+ no known neuro dx": no_known,
                       "+ no neuro dx (ICD record)": relaxed, "+ not sedated (TUH rule)": strict}.items():
        dec = pd.Series(hx.decade(df.loc[mask, "age"]))
        out[name] = dec.value_counts().reindex(DECADES[1:9], fill_value=0)
    t = pd.DataFrame(out)
    t.index = [f"{d}–{d + 9}" for d in t.index]
    t.loc["total"] = t.sum()
    return t


# ------------------------------------------------------------ EDF segments
def record_flatness(hdr, data: bytes) -> np.ndarray:
    """Per EDF data record: share of the 10-20 (TUH) channels whose digital
    std x LSB is below ``FLAT_STD_UV`` -- i.e. flat for that record."""
    spr = np.asarray(hdr.samples_per_record)
    rec = np.frombuffer(data, dtype="<i2").reshape(-1, int(spr.sum()))
    bounds = np.concatenate([[0], np.cumsum(spr)])
    present = channel_map(hdr.labels, bids_canonical_channel)
    flat = []
    for name in TUH_CHANNELS:
        if name not in present:
            continue
        i = hdr.labels.index(present[name])
        lsb = (hdr.physical_max[i] - hdr.physical_min[i]) / max(hdr.digital_max[i] - hdr.digital_min[i], 1)
        flat.append(rec[:, bounds[i]:bounds[i + 1]].std(axis=1) * abs(lsb) < FLAT_STD_UV)
    return np.mean(flat, axis=0) if flat else np.ones(len(rec))


def fetch_segment_edf(client, key: str, out_path: str, start_sec: float = 60.0,
                      dur_sec: float = 120.0, keep_good: Optional[float] = None) -> str:
    """Write ``[start, start + dur)`` of the EDF ``key`` (access point) as a
    small valid EDF: its header with ``n_records`` rewritten, followed by the
    data records of that span (one range GET). The span is clamped to the file.
    With ``keep_good`` (seconds), only data records in which fewer than half of
    the 10-20 channels are flat are kept, up to ``keep_good`` seconds."""
    head, block = hx.fetch_header(client, key)
    hdr = parse_edf_signal_header(head, block)
    rec_bytes = 2 * sum(hdr.samples_per_record)
    first = min(int(start_sec // hdr.record_duration), max(hdr.n_records - 1, 0))
    n = min(int(math.ceil(dur_sec / hdr.record_duration)), hdr.n_records - first)
    if n <= 0:
        raise ValueError(f"{key}: no data records")
    offset = len(head) + len(block) + first * rec_bytes
    data = client.get_object(Bucket=hx.ACCESS_POINT, Key=key,
                             Range=f"bytes={offset}-{offset + n * rec_bytes - 1}")["Body"].read()
    n = len(data) // rec_bytes
    data = data[:n * rec_bytes]
    if keep_good is not None:
        good = np.flatnonzero(record_flatness(hdr, data) < 0.5)
        good = good[:int(keep_good // hdr.record_duration)]
        if good.size == 0:
            raise ValueError(f"{key}: no record with signal in the span")
        data = b"".join(data[i * rec_bytes:(i + 1) * rec_bytes] for i in good)
        n = good.size
    new_head = head[:236] + str(n).encode().ljust(8) + head[244:]
    with open(out_path, "wb") as fh:
        fh.write(new_head + block + data)
    return out_path


def _read_excerpt(path: str):
    raw = qc.load_segment(path, 0.0, 1e9, bids_canonical_channel)
    return raw.get_data(units="uV"), raw.info["sfreq"]


def signal_profile(client, key: str, tmp: str, n_probes: int = 12, probe_sec: float = 10.0):
    """Probe ``n_probes`` evenly spaced ``probe_sec`` excerpts of a recording.
    Per probe: share of the TUH channels that are flat (std < ``FLAT_STD_UV``)
    and the median over channels of the std (µV; insensitive to DC offset). Returns (duration_sec, probe table)."""
    head, block = hx.fetch_header(client, key)
    dur = parse_edf_signal_header(head, block).duration_sec
    rows = []
    for t in np.linspace(0, max(dur - probe_sec, 0), n_probes):
        path = fetch_segment_edf(client, key, os.path.join(tmp, "probe.edf"), t, probe_sec)
        x, _ = _read_excerpt(path)
        std = x.std(axis=1)
        rows.append({"t": float(t), "flat_ch_frac": float((std < FLAT_STD_UV).mean()),
                     "median_std_uv": float(np.median(std))})
    return dur, pd.DataFrame(rows)


def _analyze_one(job):
    key, dur_sec, profile = job
    client = hx.s3_client(profile, pool=4)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            total, probes = signal_profile(client, key, tmp)
            flat = probes["flat_ch_frac"] >= 0.5
            physio = ~flat & probes["median_std_uv"].between(*PHYSIOLOGICAL_UV)
            good = probes[physio]
            out = {"duration_sec": total,
                   # Eqs. (8.1)-(8.2): share of probes that are flat / non-physiological.
                   "flat_time_frac": float(flat.mean()),
                   "noisy_time_frac": float((~flat & ~physio).mean()),
                   "first_signal_sec": float(probes.loc[~flat, "t"].min()) if (~flat).any() else float("nan"),
                   # Probe-by-probe state in time order: F = flat, N = noisy, S = signal.
                   "probe_pattern": "".join("F" if f else ("S" if p else "N")
                                            for f, p in zip(flat, physio))}
            if good.empty:
                return key, {**out, "error": "no physiological probe"}
            # Analyse 2 minutes of records with signal, searched in the 10 minutes
            # from the middle physiological probe (TUH QC uses 60-180 s).
            start = float(good["t"].iloc[len(good) // 2])
            path = fetch_segment_edf(client, key, os.path.join(tmp, "seg.edf"), start, 600.0,
                                     keep_good=dur_sec)
            res = qc.analyze_recording(path, start_sec=0.0, dur_sec=dur_sec,
                                       canonical=bids_canonical_channel)
            return key, {**out, "analysis_start_sec": start, **res}
        except Exception as exc:          # one bad file must not stop the sample
            return key, {"error": f"{type(exc).__name__}: {exc}"}


def analyze_sample(keys: Iterable[str], cache: Optional[str] = None, *, dur_sec: float = 120.0,
                   workers: int = 6, profile: Optional[str] = hx.DEFAULT_PROFILE):
    """Per EDF: the flat-signal profile (``signal_profile``) and
    ``preprocessing_qc.analyze_recording`` on a 2-minute excerpt with signal.
    Returns ``(table, spectra)`` like ``preprocessing_qc.analyze_sample``;
    cached as a pickle."""
    import concurrent.futures as cf
    import pickle
    if cache and os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    jobs = [(k, dur_sec, profile) for k in keys]
    with cf.ProcessPoolExecutor(workers) as pool:
        results = dict(pool.map(_analyze_one, jobs, chunksize=1))
    spectra = {k: {n: v for n, v in r.items() if n.endswith("psd")} for k, r in results.items()}
    table = pd.DataFrame([{"key": k, **{n: v for n, v in r.items() if not n.endswith("psd")}}
                          for k, r in results.items()])
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump((table, spectra), fh)
    return table, spectra


def stratified_sample(idx: pd.DataFrame, per_cell: int = 50, seed: int = 0) -> pd.DataFrame:
    """Up to ``per_cell`` recordings per site x report label (normal / abnormal
    / unlabelled), one recording per patient."""
    df = idx.assign(lab=idx["label"].fillna("unlabelled")).drop_duplicates("subject")
    return (df.groupby(["site", "lab"], group_keys=False)
              .apply(lambda g: g.sample(n=min(per_cell, len(g)), random_state=seed))
              .reset_index(drop=True))


def channel_presence(headers: pd.DataFrame) -> pd.DataFrame:
    """Share (%) of recordings in which each of the 23 TUH channels is present."""
    miss = headers["missing_channels"].dropna()
    n = max(len(miss), 1)
    return pd.DataFrame({"present %": [100 * (1 - sum(c in list(m) for m in miss) / n)
                                       for c in TUH_CHANNELS]}, index=list(TUH_CHANNELS))


# ------------------------------------------------------------------ brain-age scoring

def site_of(stem: str) -> str:
    """BDSP site of a BIDS stem: ``sub-S0001115…_ses-1_task-EEG`` -> ``S0001``."""
    sub = stem.split("_", 1)[0][len("sub-"):]
    return next((s for s in hx.MAIN_SITES if sub.startswith(s)), sub[:5])


def prediction_frame(predictions: Dict[str, float], metadata: Dict[str, dict]) -> pd.DataFrame:
    """One row per recording: ``stem, subject, site, cohort, age, pred`` from
    ``scripts/eval_age_by_cohort.py`` predictions and a cohort's ``age_metadata``."""
    rows = []
    for stem, pred in predictions.items():
        m = metadata[stem]
        rows.append({"stem": stem, "subject": m["subject"], "site": site_of(stem),
                     "cohort": m.get("label") or "unlabelled", "age": float(m["age"]),
                     "pred": float(pred)})
    return pd.DataFrame(rows)


def recalibrate_2fold(df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Out-of-fold linear recalibration: patients (in order of first appearance)
    are split in two halves by a seeded permutation; on each half the fit
    ``age ≈ a + b·pred`` (OLS) of the other half is applied."""
    subjects = list(dict.fromkeys(df["subject"]))
    perm = np.random.default_rng(seed).permutation(len(subjects))
    first = set(np.asarray(subjects)[perm[:len(subjects) // 2]])
    fold = df["subject"].isin(first).to_numpy()
    pred, age = df["pred"].to_numpy(), df["age"].to_numpy()
    out = np.empty_like(pred)
    for k in (False, True):
        b, a = np.polyfit(pred[fold != k], age[fold != k], 1)
        out[fold == k] = a + b * pred[fold == k]
    return out


def _age_stats(g: pd.DataFrame) -> Dict[str, float]:
    err = g["pred"] - g["age"]
    ss = ((g["age"] - g["age"].mean()) ** 2).sum()
    out = {"n": len(g), "MAE": err.abs().mean(), "bias": err.mean(),
           "R2": 1 - (err ** 2).sum() / ss if ss > 0 else float("nan")}
    if g["age"].nunique() > 1:
        out["slope_pred_on_age"] = np.polyfit(g["age"], g["pred"], 1)[0]
    if "recal" in g:
        out["MAE_recalibrated_2fold"] = (g["recal"] - g["age"]).abs().mean()
    return out


def brain_age_tables(df: pd.DataFrame, min_n: int = 10) -> Dict[str, pd.DataFrame]:
    """MAE / bias / R² / slope / recalibrated MAE of one model's predictions
    (``prediction_frame``), overall and per cohort, per site and per age
    decade. Groups with fewer than ``min_n`` recordings are dropped (DUA
    small-cell rule)."""
    df = df.assign(recal=recalibrate_2fold(df),
                   decade=(df["age"] // 10 * 10).clip(upper=80).astype(int))
    by_cohort = [{"cohort": "all", **_age_stats(df)}]
    by_cohort += [{"cohort": c, **_age_stats(df[df["cohort"] == c])}
                  for c in ("normal", "abnormal", "unlabelled") if (df["cohort"] == c).sum() >= min_n]
    by_site = [{"site": s, **_age_stats(g)} for s, g in df.groupby("site") if len(g) >= min_n]
    # R² and slope are not meaningful inside one decade (truncated age range).
    by_decade = [{"decade": f"{d}s" if d < 80 else "80+",
                  **{k: v for k, v in _age_stats(g).items() if k not in ("R2", "slope_pred_on_age")}}
                 for d, g in df.groupby("decade") if len(g) >= min_n]
    return {"cohort": pd.DataFrame(by_cohort), "site": pd.DataFrame(by_site),
            "decade": pd.DataFrame(by_decade)}
