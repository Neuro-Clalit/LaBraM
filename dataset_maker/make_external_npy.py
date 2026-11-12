#!/usr/bin/env python
"""Convert external resting-state EEG corpora to an evaluation cohort in the TUH npy format.

The signal pipeline is ``make_TUH_npy.preprocess_raw``, the one every training
cohort went through (0.1-75 Hz band-pass, spectrum-interpolation line removal at
50 and 60 Hz, 200 Hz, µV, float16, the 23 TUH channels in model order with
missing ones zero and flagged in ``channel_mask``, whole 10 s windows). Only the
reader and the channel naming differ per corpus:

* ``lemon`` -- MPI-LEMON (Babayan et al. 2019), BrainVision, 61 EEG channels at
  2500 Hz, FCz reference (kept: the models apply a common average reference at
  their input), 16 min of alternating 1-min eyes-closed / eyes-open blocks. Ages
  are only published as 5-year bins ("20-25"); ``age`` is the bin midpoint and
  ``raw_age`` the bin.
* ``mipdb`` -- CMI MIPDB (Langer et al. 2017), EGI HydroCel 128 at 500 Hz, the
  resting-state file (trigger 90, alternating eyes open / closed). The
  recording reference is the vertex (Cz = E129/VREF, all zero), so the 19
  mapped channels are re-referenced to their mean on reading
  (``reference="average"``); the models' input CAR makes this equivalent to any
  other reference. Ages are whole years from ``MIPDB_PublicFile.csv``.
* ``chbp`` -- Cuban Human Brain Mapping Project (Valdes-Sosa et al. 2021,
  Synapse ``syn22324937``, ``download_synapse.py``), BIDS EDF at 200 Hz,
  recorded 0.5-50 Hz with a 60 Hz notch, 120 EEG channels (10-5). Only the
  resting part (eyes closed, then eyes open / closed) up to the first
  hyperventilation event is kept. Ages are whole years from
  ``Demographic_data.csv``.

* ``tdbrain`` -- TDBRAIN (van Dijk et al. 2022, brainclinics.com/resources,
  ORCID login + DUA), BIDS BrainVision at 500 Hz, DC-coupled, 26 EEG channels
  (10-10) referenced to averaged mastoids, 2 min eyes closed + 2 min eyes open
  per session, each converted as its own recording. Psychiatric patients
  (MDD, ADHD, SMC, OCD, ...) and healthy controls, 5-88 years, decimal ages and
  clinical flags from ``participants.tsv``; ``clinical.parquet`` keeps every
  participants.tsv column. See ``docs/tdbrain_onboarding.md``.

The healthy-volunteer corpora are labelled ``normal`` (MIPDB's ``DX_Status``
is kept in the manifest); in TDBRAIN only the healthy controls are, the
patients stay unlabelled. Every recording is placed in the **test** split, so a
model trained elsewhere scores the whole cohort
(``scripts/eval_age_by_cohort.py --splits test --pred-stats-from-log``).

    python -m dataset_maker.make_external_npy --dataset lemon \\
        --raw /data/datasets/EEG-public/LEMON/raw \\
        --meta /data/datasets/EEG-public/LEMON/meta/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv \\
        --out /data/datasets/EEG-public/LEMON/cohorts/lemon_v1 --jobs 6
"""
import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import sys

import numpy as np

try:
    from dataset_maker.make_TUAB_npy import npy_bytes
    from dataset_maker.make_TUH_npy import (FLOAT16_MAX, MIN_DURATION_SEC, SAMPLE_RATE,
                                            WINDOW_SAMPLES, preprocess_raw)
    from labram.data.tuh_metadata import TUH_CHANNELS, bids_canonical_channel, channel_map
except ImportError:      # run as a plain script from dataset_maker/
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dataset_maker.make_TUAB_npy import npy_bytes
    from dataset_maker.make_TUH_npy import (FLOAT16_MAX, MIN_DURATION_SEC, SAMPLE_RATE,
                                            WINDOW_SAMPLES, preprocess_raw)
    from labram.data.tuh_metadata import TUH_CHANNELS, bids_canonical_channel, channel_map

NOTCH = (50.0, 60.0)
LINE = "interp"
# EGI HydroCel GSN 128 electrodes closest to the 10-20 positions (Luu & Ferree
# 2005; the vertex reference E129 is Cz). A1/A2/T1/T2 have no counterpart here.
EGI_HYDROCEL_1020 = {
    "FP1": "E22", "FP2": "E9", "F7": "E33", "F3": "E24", "FZ": "E11", "F4": "E124", "F8": "E122",
    "T3": "E45", "C3": "E36", "CZ": "E129", "C4": "E104", "T4": "E108",
    "T5": "E58", "P3": "E52", "PZ": "E62", "P4": "E92", "T6": "E96", "O1": "E70", "O2": "E83",
}


# ------------------------------------------------------------------ readers
def lemon_recordings(raw_dir, meta_csv):
    import pandas as pd
    meta = pd.read_csv(meta_csv).rename(columns={"Unnamed: 0": "ID"})
    meta = meta.set_index(meta.columns[0])
    out = []
    for vhdr in sorted(glob.glob(os.path.join(raw_dir, "sub-*", "RSEEG", "*.vhdr"))):
        sub = os.path.basename(vhdr)[:-5]
        if sub not in meta.index:
            continue
        m = meta.loc[sub]
        lo, hi = (float(v) for v in str(m["Age"]).split("-"))
        sex = {1: "F", 2: "M"}.get(int(m["Gender_ 1=female_2=male"]), None)
        out.append({"stem": f"{sub}_lemon-rs", "path": os.path.relpath(vhdr, raw_dir), "corpus": "lemon",
                    "subject": sub, "session": "1", "age": (lo + hi) / 2, "raw_age": str(m["Age"]),
                    "sex": sex, "label": "normal", "_file": vhdr})
    return out


def mipdb_recordings(raw_dir, meta_csv):
    import pandas as pd
    meta = pd.read_csv(meta_csv).set_index("ID")
    out = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "A*", "*.raw"))):
        sub = os.path.basename(os.path.dirname(f))
        if sub not in meta.index or pd.isna(meta.loc[sub, "Age"]):
            continue
        m = meta.loc[sub]
        dx = m.get("DX_Status")
        out.append({"stem": f"{sub}_mipdb-rs", "path": os.path.relpath(f, raw_dir), "corpus": "mipdb",
                    "subject": sub, "session": "1", "age": float(m["Age"]), "raw_age": float(m["Age"]),
                    "sex": {1: "M", 2: "F"}.get(int(m["Sex"]), None) if not pd.isna(m["Sex"]) else None,
                    "label": "normal", "dx_status": None if pd.isna(dx) else int(dx), "_file": f})
    return out


def open_lemon(path):
    """The published .vhdr files still name their data / marker files by the
    pre-BIDS ids (``sub-010002.eeg``); they are read through a temporary folder
    of symlinks carrying those names (the downloaded files stay untouched)."""
    import re
    import shutil
    import tempfile
    import mne
    text = open(path, encoding="utf-8", errors="replace").read()
    names = dict(re.findall(r"^(DataFile|MarkerFile)=(.+?)\s*$", text, re.M))
    base = path[:-5]
    tmp = tempfile.mkdtemp(prefix="lemon_")
    try:
        os.symlink(base + ".eeg", os.path.join(tmp, names.get("DataFile", os.path.basename(base) + ".eeg")))
        os.symlink(base + ".vmrk", os.path.join(tmp, names.get("MarkerFile", os.path.basename(base) + ".vmrk")))
        shutil.copy(path, os.path.join(tmp, "header.vhdr"))
        raw = mne.io.read_raw_brainvision(os.path.join(tmp, "header.vhdr"), preload=True, verbose="error")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return raw, channel_map(raw.ch_names, bids_canonical_channel), None


def open_mipdb(path):
    import mne
    raw = mne.io.read_raw_egi(path, preload=False, verbose="error")
    names = set(raw.ch_names)
    present = {}
    for tuh, egi in EGI_HYDROCEL_1020.items():
        for label in (egi, "VREF", "Vertex Reference") if egi == "E129" else (egi,):
            if label in names:
                present[tuh] = label
                break
    return raw, present, "average"


# CHBP protocol events (Spanish in release 1.0, English in 1.1) that end the resting part.
CHBP_NON_REST = ("hiperventilacion", "hyperventilation", "fotoestimulacion", "photic", "recuperacion", "recovery")


def chbp_recordings(raw_dir, meta_csv):
    import pandas as pd
    meta = pd.read_csv(meta_csv, skiprows=1, encoding="utf-8-sig").dropna(axis=1, how="all")
    meta = meta.dropna(subset=["Code"]).set_index("Code")
    out = []
    for edf in sorted(glob.glob(os.path.join(raw_dir, "sub-*", "eeg", "*_eeg.edf"))):
        code = os.path.basename(edf).split("_")[0][len("sub-"):]
        if code not in meta.index or pd.isna(meta.loc[code, "Age"]):
            continue
        m = meta.loc[code]
        out.append({"stem": f"sub-{code}_chbp-rs", "path": os.path.relpath(edf, raw_dir), "corpus": "chbp",
                    "subject": code, "session": "1", "age": float(m["Age"]), "raw_age": float(m["Age"]),
                    "sex": m["Gender"] if m["Gender"] in ("M", "F") else None, "label": "normal", "_file": edf})
    return out


def chbp_resting_end(edf):
    """Onset (s) of the first hyperventilation / recovery / photic event in the
    BIDS ``events.tsv`` next to ``edf`` (None if there is none)."""
    import csv
    events = edf[:-len("_eeg.edf")] + "_events.tsv"
    if not os.path.exists(events):
        return None
    with open(events, encoding="utf-8", errors="replace") as fh:
        onsets = [float(r["onset"]) for r in csv.DictReader(fh, delimiter="\t")
                  if any(k in r["trial_type"].lower() for k in CHBP_NON_REST)]
    return min(onsets) if onsets else None


def open_chbp(path):
    """EDF at 200 Hz, recorded 0.5-50 Hz with a 60 Hz notch, 120 EEG channels
    (10-5). Labels in release 1.0 carry a space and a reference suffix that
    varies (``'F3 -REF'``, ``'F3 -A12'`` for linked ears, a truncated ``'F4 -'``),
    so only the part before the hyphen is matched. Only the resting part is
    kept: eyes closed (~10 min) then alternating eyes open / closed, cut where
    hyperventilation starts."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="error")
    end = chbp_resting_end(path)
    if end is not None and end < raw.times[-1]:
        raw.crop(0, end, include_tmax=False)
    present = channel_map(raw.ch_names, lambda label: bids_canonical_channel(label.split("-")[0].strip()))
    return raw, present, None


# ------------------------------------------------------------------ TDBRAIN
TDBRAIN_TASKS = ("restEC", "restEO")
# Indication keyword (upper case, substring) -> clinical flag. The first match of
# each token of the indication wins; a token matching none sets ``dx_other``.
TDBRAIN_INDICATIONS = (("HEALTHY", "healthy"), ("CONTROL", "healthy"), ("REPLICATION", "blinded"),
                       ("UNKNOWN", "unknown"), ("MDD", "dx_mdd"), ("DEPRESS", "dx_mdd"),
                       ("ADHD", "dx_adhd"), ("ADD", "dx_adhd"), ("OCD", "dx_ocd"), ("SMC", "dx_smc"))
TDBRAIN_DX = ("dx_mdd", "dx_adhd", "dx_ocd", "dx_smc", "dx_other")


def _column(df, *names, required=False):
    """The column of ``df`` matching one of ``names`` (case, spaces, ``_`` and
    ``/`` ignored), else None; ``required`` raises listing the columns found."""
    def key(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())
    cols = {key(c): c for c in df.columns}
    for name in names:
        if key(name) in cols:
            return cols[key(name)]
    if required:
        raise KeyError(f"none of {names} in participants.tsv; columns: {list(df.columns)}")
    return None


def _number(value):
    """``'35,53'`` / ``'35.53'`` / 35.53 -> 35.53; blank or text -> NaN."""
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return float("nan")


def _bare_id(value):
    s = str(value).strip()
    return s[len("sub-"):] if s.startswith("sub-") else s


def tdbrain_indication_flags(indication):
    """``'MDD/ADHD'`` -> ``{'healthy': False, 'dx_mdd': True, 'dx_adhd': True, ...,
    'clinical_group': 'dx_mdd'}``. The group is ``healthy``, ``blinded`` (the
    REPLICATION set, diagnosis withheld by the authors), ``unknown`` (no
    indication; *not* healthy, per van Dijk et al. 2022) or the first diagnosis."""
    import re
    flags = {"healthy": False, "blinded": False, "unknown": False, **{d: False for d in TDBRAIN_DX}}
    text = "" if indication is None or str(indication).strip().lower() in ("", "nan", "n/a") else str(indication)
    tokens = [t.strip() for t in re.split(r"[/,;+&]|\bAND\b", text.upper()) if t.strip()]
    first = None
    for tok in tokens:
        flag = next((f for kw, f in TDBRAIN_INDICATIONS if kw in tok), "dx_other")
        flags[flag] = True
        first = first or flag
    if not tokens:
        flags["unknown"] = True
    group = ("healthy" if flags["healthy"] else "blinded" if flags["blinded"]
             else first if first in TDBRAIN_DX else "unknown")
    return {**flags, "clinical_group": group}


def tdbrain_clinical_table(participants_tsv):
    """One row per participants.tsv row with every original column kept and
    harmonised ones added: ``subject`` (id without ``sub-``), ``session``
    (str, None if the file has no session column), ``age`` (years, decimal
    comma accepted), ``sex`` (M / F from the ``participants.json`` levels
    next to the tsv; None when they cannot be decoded), ``indication``,
    ``formal_status`` (DSM diagnosis vs referral indication), ``disc_rep``
    (DISCOVERY / REPLICATION) and the flags of :func:`tdbrain_indication_flags`.
    TDBRAIN has no EEG report and no ICD / medication record (medication was
    allowed but not tracked); the clinical information is the indication."""
    import pandas as pd
    df = pd.read_csv(participants_tsv, sep="\t", dtype=str, keep_default_na=False)
    pid = _column(df, "participants_ID", "participant_id", required=True)
    age = _column(df, "age", required=True)
    ses = _column(df, "sessID", "session_id", "session", "ses")
    sex = _column(df, "gender", "sex")
    ind = _column(df, "indication")
    fstat = _column(df, "formal_status", "formal_dx", "formalstatus")
    drep = _column(df, "DISC/REP", "disc_rep", "discrep")
    levels = {}
    sidecar = os.path.splitext(participants_tsv)[0] + ".json"
    if sex and os.path.exists(sidecar):
        desc = json.load(open(sidecar, encoding="utf-8", errors="replace")).get(sex, {})
        levels = {str(k).strip(): str(v) for k, v in (desc.get("Levels") or {}).items()}

    def decode_sex(v):
        word = levels.get(str(v).strip(), str(v)).strip().lower()
        return {"m": "M", "male": "M", "man": "M", "f": "F", "female": "F", "woman": "F"}.get(word)

    out = df.copy()
    out["subject"] = df[pid].map(_bare_id)
    out["session"] = df[ses].map(lambda v: str(v).strip().removeprefix("ses-") or None) if ses else None
    out["age"] = df[age].map(_number)
    out["raw_age"] = df[age]
    out["sex"] = df[sex].map(decode_sex) if sex else None
    out["indication"] = df[ind].str.strip() if ind else None
    out["formal_status"] = df[fstat].str.strip().str.upper() if fstat else None
    out["disc_rep"] = df[drep].str.strip().str.upper() if drep else None
    flags = pd.DataFrame([tdbrain_indication_flags(v) for v in (out["indication"] if ind else [None] * len(df))],
                         index=df.index)
    # sub-19* is the blinded replication set even where the indication column says otherwise
    blinded = out["subject"].str.startswith("19") | (out["disc_rep"] == "REPLICATION")
    flags.loc[blinded & ~flags["healthy"], ["blinded", "clinical_group"]] = [True, "blinded"]
    return pd.concat([out, flags], axis=1)


def tdbrain_vhdr_files(raw_dir):
    """``(subject, session, task, path)`` of every BIDS resting file under
    ``raw_dir`` (``sub-*/ses-*/eeg/*_task-rest{EC,EO}_eeg.vhdr``), the
    ``derivatives`` copies (csv) skipped."""
    import re
    pat = re.compile(r"^sub-([^_]+)_ses-([^_]+)_task-(rest(?:EC|EO))_eeg\.vhdr$", re.I)
    out = []
    for dirpath, dirnames, filenames in os.walk(raw_dir):
        dirnames[:] = sorted(d for d in dirnames if d.lower() != "derivatives")
        for name in sorted(filenames):
            m = pat.match(name)
            if m:
                task = "restEC" if m[3].upper().endswith("EC") else "restEO"
                out.append((m[1], m[2], task, os.path.join(dirpath, name)))
    return out


def tdbrain_recordings(raw_dir, meta_tsv):
    """One recording per (subject, session, task): the 2-min eyes-closed and
    eyes-open files are converted separately (stem
    ``sub-<id>_ses-<n>_<task>_tdbrain``) and scored per file or pooled per
    session. Age, sex and the clinical flags come from the participants.tsv row
    of that session (or of the subject when the file lists no session).
    ``label`` is ``normal`` for healthy participants only; patients stay
    unlabelled (no EEG report exists, a psychiatric indication is not an EEG
    finding)."""
    table = tdbrain_clinical_table(meta_tsv)
    by_session = {(r.subject, r.session): r for r in table.itertuples()} if table["session"].notna().any() else {}
    by_subject = {}
    for r in table.itertuples():
        by_subject.setdefault(r.subject, r)
    keep = ("indication", "formal_status", "disc_rep", "healthy", "blinded", "unknown", *TDBRAIN_DX,
            "clinical_group")
    out = []
    for sub, ses, task, path in tdbrain_vhdr_files(raw_dir):
        row = by_session.get((sub, ses)) or by_session.get((sub, ses.lstrip("0") or "0"))
        matched = row is not None
        row = row if matched else by_subject.get(sub)
        if row is None or not np.isfinite(row.age):
            continue
        out.append({"stem": f"sub-{sub}_ses-{ses}_{task}_tdbrain", "path": os.path.relpath(path, raw_dir),
                    "corpus": "tdbrain", "subject": sub, "session": ses, "task": task,
                    "age": float(row.age), "raw_age": row.raw_age, "sex": row.sex,
                    "label": "normal" if row.healthy else None, "age_session_matched": matched,
                    **{k: (bool(getattr(row, k)) if isinstance(getattr(row, k), (bool, np.bool_))
                           else getattr(row, k)) for k in keep},
                    "_file": path})
    return out


def open_tdbrain(path):
    """BrainVision at 500 Hz, DC-coupled (low-pass 100 Hz), 26 EEG channels
    (10-10) + EOG / ECG / EMG, re-referenced offline to averaged mastoids, so
    A1/A2 (and T1/T2) are absent. The recording is DC-coupled: each channel's
    mean is removed before the pipeline's 0.1 Hz high-pass, which otherwise
    leaves a transient over the first seconds of a 2-min file. The original
    files are never written to (DUA)."""
    import mne
    raw = mne.io.read_raw_brainvision(path, preload=True, verbose="error")
    raw._data -= raw._data.mean(axis=1, keepdims=True)
    return raw, channel_map(raw.ch_names, bids_canonical_channel), None


DATASETS = {"lemon": (lemon_recordings, open_lemon), "mipdb": (mipdb_recordings, open_mipdb),
            "chbp": (chbp_recordings, open_chbp), "tdbrain": (tdbrain_recordings, open_tdbrain)}


# ------------------------------------------------------------------ convert
def convert_one(rec, opener, out_dir):
    raw, present, reference = opener(rec["_file"])
    # recorder markers (segment boundaries, events): a mid-file boundary means a discontinuity
    annotations = {}
    for desc in raw.annotations.description:
        annotations[str(desc)] = annotations.get(str(desc), 0) + 1
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq
    if duration < MIN_DURATION_SEC:
        return {"stem": rec["stem"], "error": f"too short ({duration:.0f} s)"}
    if len(present) < 19:
        return {"stem": rec["stem"], "error": f"only {len(present)} of the 19 10-20 channels found"}
    x, mask = preprocess_raw(raw, present, NOTCH, LINE, reference=reference)
    x = np.clip(x, -FLOAT16_MAX, FLOAT16_MAX).astype(np.float16)
    data = npy_bytes(np.ascontiguousarray(x))
    with open(os.path.join(out_dir, "recordings", rec["stem"] + ".npy"), "wb") as fh:
        fh.write(data)
    return {"stem": rec["stem"], "n_windows": x.shape[0] // WINDOW_SAMPLES, "n_samples": x.shape[0],
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "channel_mask": mask,
            "dtype": "float16", "notch": "50,60", "line": "interp:50,60",
            "reference": reference or "as recorded", "sfreq_native": sfreq, "duration_sec": duration,
            "annotations": annotations}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    lister, opener = DATASETS[args.dataset]
    recs = lister(args.raw, args.meta)[:args.limit or None]
    os.makedirs(os.path.join(args.out, "recordings"), exist_ok=True)
    print(f"{args.dataset}: {len(recs)} recordings", flush=True)
    results, errors = {}, []
    with cf.ProcessPoolExecutor(args.jobs) as pool:
        futs = {pool.submit(convert_one, r, opener, args.out): r for r in recs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:          # record, never abort the run
                res = {"stem": r["stem"], "error": f"{type(exc).__name__}: {exc}"[:300]}
            if "error" in res:
                errors.append(res)
            else:
                results[r["stem"]] = res
            if i % 20 == 0 or i == len(recs):
                print(f"  {i}/{len(recs)} done, {len(errors)} errors", flush=True)
    rows = []
    for r in recs:
        if r["stem"] not in results:
            continue
        row = {**results[r["stem"]], **{k: v for k, v in r.items() if not k.startswith("_")},
               "token": None, "year": None, "montage": None, "session_date": None,
               "abnormal": None, "epilepsy": None, "source_split": "test", "site": args.dataset}
        rows.append(row)
    manifest = {
        "format_version": 2, "layout": "recordings/<stem>.npy, [n_samples, n_channels], time-major",
        "dtype": "float16", "sample_rate": SAMPLE_RATE, "window_samples": WINDOW_SAMPLES,
        "channels": list(TUH_CHANNELS), "units": "uV", "notch_hz": "50,60", "line_removal": "interp:50,60",
        "provenance": (f"dataset_maker/make_external_npy.py --dataset {args.dataset}: "
                       "make_TUH_npy.preprocess_raw (0.1-75.0 Hz band-pass, line removal interp:50,60 Hz, "
                       "resampled to 200 Hz); missing channels zero-filled, see channel_mask"),
        "n_recordings": len(rows), "n_windows": int(sum(r["n_windows"] for r in rows)),
        "cohort": {"experiment": "external_eval", "corpus": args.dataset, "split": "all test",
                   "errors": errors},
        "recordings": rows,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh)
    if args.dataset == "tdbrain":
        # every participants.tsv column (questionnaires, demographics) + the harmonised clinical flags
        tdbrain_clinical_table(args.meta).to_parquet(os.path.join(args.out, "clinical.parquet"), index=False)
    fields = ("stem", "subject", "session", "token", "age", "sex", "year", "raw_age", "label")
    with open(os.path.join(args.out, "age_metadata.json"), "w") as fh:
        json.dump({"version": 1, "recordings": {r["stem"]: {k: r.get(k) for k in fields} for r in rows}}, fh)
    files = [f"test/{r['stem']}_{k}.pkl" for r in rows for k in range(r["n_windows"])]
    with open(os.path.join(args.out, "age_split.json"), "w") as fh:
        json.dump({"version": 1, "seed": None, "counts": {"train": 0, "val": 0, "test": len(rows)},
                   "files": {"train": [], "val": [], "test": files}}, fh)
    print(f"{len(rows)} recordings, {manifest['n_windows']} windows -> {args.out}; {len(errors)} errors")
    for e in errors[:10]:
        print("  ", e)


if __name__ == "__main__":
    main()
