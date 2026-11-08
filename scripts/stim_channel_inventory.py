#!/usr/bin/env python
"""Stimulus (photic / trigger / DC) channels in the TUH and HEEDB cohorts: present and valid?

Two parts, written to ``--out-dir``:

1. **Header inventory** (every recording, no signal read): whether the EDF
   lists a photic-stimulator channel. TUH (TUEG) labels it ``PHOTIC`` (from
   the NEDC header dump ``DOCS/headers.tar.gz``); HEEDB (BIDS / Natus) has no
   such label, so its candidates are the DC inputs ``DC1..DC16`` (where a
   photic stimulator is usually wired on Natus/XLTEK systems) and ``TRIG``.
   Shares per cohort, data split and HEEDB site -> ``header_inventory.csv``.
2. **Signal check** (a random sample): download the EDF, read the candidate
   channels and classify each as ``flat`` (constant), ``pulse train`` (a
   two-level signal with at least 20 rising edges, i.e. flashes / triggers)
   or ``analog`` (anything else: a continuous signal or noise), with the
   pulse count, the seconds containing a pulse and the median flash rate.
   -> ``signal_check.csv`` (one row per channel), ``signal_summary.csv``.

    AWS_PROFILE=mfa python scripts/stim_channel_inventory.py \\
        --out-dir /data/runs/heedb/stim_channels --n-tuh 120 --n-heedb 160
"""
import argparse
import os
import re
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

TUEG_ROOT = "/data/datasets/EEG-public/TUH_EEG/tueg/v2.0.2"
TUEG_S3 = ("eeg-data-public", "TUH_EEG/tueg/v2.0.2")
HEEDB_INDEX = "/data/datasets/HEEDB/v4.1/index/heedb_index.parquet"
COHORTS = {
    "all_both_v3": "/data/datasets/EEG-public/TUH_EEG/cohorts/all_both_v3",
    "all_both_v3_anymiss": "/data/datasets/EEG-public/TUH_EEG/cohorts/all_both_v3_anymiss",
    "heedb_routine_v1": "/data/datasets/HEEDB/v4.1/cohorts/heedb_routine_v1",
    "tuh_heedb_v1": "/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1",
}
DC_RE = re.compile(r"^DC\d+$")
_FILE = re.compile(r"^\s*(?:file|filename)\s*[:=]\s*(\S+\.edf)", re.I)


# ----------------------------------------------------------------- headers
def tueg_header_labels(path=os.path.join(TUEG_ROOT, "DOCS", "headers.tar.gz")) -> pd.DataFrame:
    """stem -> stimulus-like channel labels of every TUEG EDF, from the header dump."""
    from labram.data.tuh_meta_features import _DUMP_FILE, _DUMP_FIELD, _dump_values
    rows, stem = [], None
    with tarfile.open(path) as tar:
        member = next(m for m in tar.getmembers() if m.isfile())
        for raw in tar.extractfile(member):
            line = raw.decode("latin-1")
            m = _DUMP_FILE.match(line)
            if m:
                stem = os.path.splitext(os.path.basename(m.group(1)))[0]
                continue
            f = _DUMP_FIELD.match(line)
            if f and stem is not None and f.group(1) == "chan_labels":
                labels = [v.upper() for v in _dump_values(f.group(2))]
                rows.append({"stem": stem,
                             "photic": any("PHOTIC" in v for v in labels),
                             "trig_stim": any(re.search(r"TRIG|STIM|\bSTI\b|EVENT", v) for v in labels)})
                stem = None
    return pd.DataFrame(rows).drop_duplicates("stem")


def heedb_header_labels() -> pd.DataFrame:
    h = pd.read_parquet(HEEDB_INDEX, columns=["stem", "site", "key", "bytes", "labels"])
    labels = h["labels"].fillna("").str.upper().str.split("|")
    h["dc"] = labels.apply(lambda ls: any(DC_RE.match(x) for x in ls))
    h["trig"] = labels.apply(lambda ls: any(x.startswith("TRIG") for x in ls))
    h["photic"] = labels.apply(lambda ls: any("PHOT" in x or x.startswith("STI") for x in ls))
    return h


def cohort_table(name, path):
    import json
    coh = pd.read_parquet(os.path.join(path, "cohort.parquet"))
    rec = pd.DataFrame(json.load(open(os.path.join(path, "manifest.json")))["recordings"])
    cols = ["stem", "path"] + (["corpus"] if "corpus" in rec else [])
    df = coh[["stem", "split"]].merge(rec[cols], on="stem")
    # anything not HEEDB is a TUH recording (TUEG cohorts tag it "tueg")
    df["corpus"] = np.where(df.get("corpus", pd.Series("tuh", index=df.index)) == "heedb", "heedb", "tuh")
    return df.assign(cohort=name)


def header_inventory(tueg, heedb):
    rows = []
    all_tueg = tueg.assign(cohort="TUEG (whole corpus)", corpus="tuh", split="all")
    all_heedb = heedb.assign(cohort="HEEDB (whole index)", corpus="heedb", split="all")
    frames = [all_tueg, all_heedb]
    for name, path in COHORTS.items():
        c = cohort_table(name, path)
        frames.append(c[c.corpus == "tuh"].merge(tueg, on="stem", how="left"))
        frames.append(c[c.corpus == "heedb"].merge(heedb[["stem", "site", "dc", "trig", "photic"]],
                                                   on="stem", how="left"))
    for f in frames:
        if f.empty:
            continue
        corpus = f["corpus"].iloc[0]
        for split, g in [("all", f)] + ([] if (f["split"] == "all").all() else list(f.groupby("split"))):
            row = {"cohort": f["cohort"].iloc[0], "corpus": corpus, "split": split, "n": len(g),
                   "header_missing": float(g["photic"].isna().mean()),
                   "photic_label": float(g["photic"].fillna(False).mean())}
            if corpus == "tuh":
                row["trig_or_stim_label"] = float(g["trig_stim"].fillna(False).mean())
            else:
                row["dc_inputs"] = float(g["dc"].fillna(False).mean())
                row["trig_label"] = float(g["trig"].fillna(False).mean())
                row.update({f"dc_inputs_{s}": float(gs["dc"].mean()) for s, gs in g.groupby("site")})
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------ signal check
def classify(x: np.ndarray, sfreq: float) -> dict:
    """Classify one channel (values in volts, as MNE returns them).

    ``flat`` -- constant. ``no stimulation`` -- a level with fewer than 20
    excursions (glitches). ``pulse train`` -- a two-level signal (at least 98 %
    of samples within 10 % of the minimum or maximum), at least 20 excursions
    into the *rarer* level (either polarity), all within less than half of the
    recording's seconds -- a stimulator or trigger. ``analog`` -- anything
    else (a continuous signal, drift or an unconnected input's noise)."""
    x = np.asarray(x, float)
    ptp = float(np.ptp(x))
    out = {"ptp_uv": ptp * 1e6, "duration_sec": len(x) / sfreq, "two_level": False,
           "n_pulses": 0, "pulse_sec": 0.0, "pulse_frac": 0.0, "rate_hz": np.nan}
    if ptp == 0 or not np.isfinite(ptp):
        return {**out, "kind": "flat"}
    lo, hi = np.percentile(x, [0.1, 99.9])                 # robust to a few glitch samples
    if hi <= lo:
        lo, hi = x.min(), x.max()
    v = np.clip((x - lo) / (hi - lo), 0, 1)
    two_level = float(np.mean((v < 0.1) | (v > 0.9))) >= 0.98
    high = v > 0.5
    rare = high if high.mean() <= 0.5 else ~high          # the stimulus state, either polarity
    edges = np.flatnonzero(~rare[:-1] & rare[1:])
    per_sec = np.bincount((edges / sfreq).astype(int)) if len(edges) else np.zeros(1, int)
    n_sec = max(int(len(x) / sfreq), 1)
    active = per_sec[per_sec > 0]
    out.update(two_level=two_level, n_pulses=int(len(edges)), pulse_sec=float(len(active)),
               pulse_frac=float(len(active) / n_sec), rare_frac=float(rare.mean()),
               rate_hz=float(np.median(active)) if len(active) else np.nan)
    if two_level and len(edges) >= 20 and out["pulse_frac"] < 0.5:
        kind = "pulse train"
    elif len(edges) < 20 and (two_level or out["rare_frac"] < 0.01):
        kind = "no stimulation"                           # a level with a few glitches
    else:
        kind = "analog"
    return {**out, "kind": kind}


def read_channels(path, pick):
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    names = [c for c in raw.ch_names if pick(c.upper())]
    if not names:
        return {}, float(raw.info["sfreq"])
    raw.pick(names).load_data(verbose="ERROR")
    return dict(zip(names, raw.get_data())), float(raw.info["sfreq"])


def check_one(job):
    corpus, stem, bucket, key, pick, tmp = job
    import boto3
    from dataset_maker import heedb_index as hx
    client = hx.s3_client() if corpus == "heedb" else boto3.Session().client("s3")
    path = os.path.join(tmp, f"{stem}.edf")
    try:
        client.download_file(bucket, key, path)
        chans, sfreq = read_channels(path, pick)
        if not chans:
            return [{"corpus": corpus, "stem": stem, "channel": None, "kind": "no channel"}]
        return [{"corpus": corpus, "stem": stem, "channel": n, "sfreq": sfreq, **classify(x, sfreq)}
                for n, x in chans.items()]
    except Exception as exc:  # noqa: BLE001 -- one bad file must not stop the sample
        return [{"corpus": corpus, "stem": stem, "channel": None, "kind": f"error: {exc}"[:120]}]
    finally:
        if os.path.exists(path):
            os.remove(path)


def signal_check(tueg, heedb, n_tuh, n_heedb, workers, seed=0):
    from dataset_maker import heedb_index as hx
    tuh_c = cohort_table("tuh_heedb_v1", COHORTS["tuh_heedb_v1"])
    tuh_c = tuh_c[tuh_c.corpus == "tuh"].merge(tueg, on="stem")
    tuh_s = tuh_c[tuh_c.photic].sample(min(n_tuh, int(tuh_c.photic.sum())), random_state=seed)
    he_c = cohort_table("heedb_routine_v1", COHORTS["heedb_routine_v1"]).merge(
        heedb[["stem", "site", "key", "dc", "trig"]], on="stem")
    cand = he_c[he_c.dc | he_c.trig]
    # equal share per site among sessions with candidate channels
    per_site = max(n_heedb // cand.site.nunique(), 1)
    he_s = cand.groupby("site", group_keys=False).apply(
        lambda g: g.sample(min(per_site, len(g)), random_state=seed))
    tmp = tempfile.mkdtemp(prefix="stim_")
    jobs = [("tuh", r.stem, TUEG_S3[0], f"{TUEG_S3[1]}/{r.path}", lambda c: "PHOTIC" in c, tmp)
            for r in tuh_s.itertuples()]
    jobs += [("heedb", r.stem, hx.ACCESS_POINT, r.key,
              lambda c: bool(DC_RE.match(c)) or c.startswith("TRIG"), tmp) for r in he_s.itertuples()]
    with ThreadPoolExecutor(workers) as ex:
        rows = [row for res in ex.map(check_one, jobs) for row in res]
    df = pd.DataFrame(rows)
    site = he_s.set_index("stem")["site"]
    df["site"] = df["stem"].map(site).fillna("TUH")
    return df


def summarize(sig):
    out = []
    sig = sig.copy()
    sig["group"] = np.where(sig.corpus == "tuh", "PHOTIC",
                            np.where(sig.channel.fillna("").str.upper().str.startswith("TRIG"),
                                     "TRIG", "DC"))
    for (corpus, site, group), g in sig.groupby(["corpus", "site", "group"]):
        rec = g.groupby("stem")["kind"].agg(lambda k: "pulse train" if (k == "pulse train").any()
                                            else ("analog" if (k == "analog").any() else k.iloc[0]))
        pulses = g[g.kind == "pulse train"]
        out.append({"corpus": corpus, "site": site, "channel": group, "recordings": len(rec),
                    "channels": len(g), **{f"rec_{k}": float((rec == k).mean()) for k in
                                           ("pulse train", "no stimulation", "flat", "analog")},
                    "median_pulse_sec": float(pulses.groupby("stem")["pulse_sec"].max().median())
                    if len(pulses) else np.nan,
                    "median_rate_hz": float(pulses["rate_hz"].median()) if len(pulses) else np.nan})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-tuh", type=int, default=120)
    ap.add_argument("--n-heedb", type=int, default=160)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tueg, heedb = tueg_header_labels(), heedb_header_labels()
    inv = header_inventory(tueg, heedb)
    inv.to_csv(os.path.join(args.out_dir, "header_inventory.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(inv.round(3).to_string(index=False), flush=True)
    if args.n_tuh or args.n_heedb:
        sig = signal_check(tueg, heedb, args.n_tuh, args.n_heedb, args.workers)
        sig.to_csv(os.path.join(args.out_dir, "signal_check.csv"), index=False)
        summ = summarize(sig[sig.channel.notna()])
        summ.to_csv(os.path.join(args.out_dir, "signal_summary.csv"), index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(summ.round(3).to_string(index=False))
            print(sig[sig.channel.isna()]["kind"].value_counts().to_string())


if __name__ == "__main__":
    main()
