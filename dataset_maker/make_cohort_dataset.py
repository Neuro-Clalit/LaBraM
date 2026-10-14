#!/usr/bin/env python
"""Build a TUH training cohort as a dataset the TUAB age trainer reads directly.

A cohort is a set of recordings with a patient-level train/val/test split,
materialised in the ``TUAB_AGE`` npy format so the existing loader, window
selection, normal/abnormal cohort metrics and NVMe staging work unchanged::

    <out>/manifest.json        the source manifest's rows for the cohort, with
                               ``source_split`` set to the cohort split
    <out>/age_metadata.json    age, sex and normal/abnormal label per recording
    <out>/age_split.json       window files per split (subject-disjoint, checked)
    <out>/recordings/<stem>.npy  hard links (local) or server-side copies (S3)

Recordings come from one npy source (the TUEG conversion, which also contains
every TUAB recording), so all cohorts share one preprocessing recipe.

Split rules (patient level):
* TUAB patients keep their TUAB age split (``--tuab-split``): train/val/test as
  in every TUAB-only run, so results stay comparable. TUAB-eval patients
  contribute only their TUAB eval recordings (the test set); their other TUEG
  recordings are excluded.
* Other TUEG patients: their split in ``--patient-split`` (one assignment
  shared by every cohort, ``dataset_maker/tueg_patient_split.py``; v2 cohorts),
  or (v1) a seeded shuffle of the cohort's own patients -- which differs
  between cohorts, so one cohort's val patients can be in another's train.
* At most ``--max-per-subject`` recordings per patient in train and val
  (seeded), and at most ``--train-minutes`` of windows per training recording
  (evenly spaced inside the trimmed range), so no patient or long recording
  dominates.

Experiments (``--experiment``):
* ``normal_both``: TUAB + TUEG, healthy recordings only in train/val (report
  normal, no neurological diagnosis, no sedating drugs, not TUAB-abnormal);
  test = TUAB eval (all recordings).
* ``normal_tueg``: TUEG patients outside TUAB, healthy only, train/val, no test.
* ``all_both``: as ``normal_both`` but every recording (normal and abnormal).

    python -m dataset_maker.make_cohort_dataset --experiment normal_both \\
        --source /data/.../tueg/v2.0.2/processed_npy_interp50-60_float16 \\
        --out /data/.../cohorts/normal_both_v1
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import pandas as pd

from dataset_maker import tuh_clinical as tc

WINDOW_SEC = 10
EXPERIMENTS = ("normal_both", "normal_tueg", "all_both")
DEFAULT_INDEX = "/data/datasets/EEG-public/TUH_EEG/index/tuh_index.parquet"
DEFAULT_SHEET = "/data/datasets/EEG-public/TUH_EEG/index/nedc/metadata_v12.xlsm.xlsx"
DEFAULT_TUAB_SPLIT = "/data/datasets/EEG-public/TUH_EEG/tuab/v3.0.1/edf/processed_npy/age_split.json"


def tuab_subject_split(path):
    """subject -> train/val/test from a TUAB age_split.json."""
    files = json.load(open(path))["files"]
    out = {}
    for split, names in files.items():
        for name in names:
            out[os.path.basename(name).split("_")[0]] = split
    return out


def label_of(row):
    """TUAB folder label where there is one, else the NEDC report finding."""
    if pd.notna(row.get("abnormal")):
        return "abnormal" if int(row["abnormal"]) == 1 else "normal"
    if row.get("in_sheet"):
        return "abnormal" if bool(row["abnormal_report"]) else "normal"
    return None


def select(j, experiment, *, min_age, max_age, min_minutes, allow_missing=()):
    """Boolean masks of the train/val candidates and of the test recordings.
    A recording may lack only channels in ``allow_missing`` (zero-filled in the
    npy; a model then drops them with data.drop_channels)."""
    allowed = {c.upper() for c in allow_missing}
    usable = (j.age.between(min_age, max_age) & (j.duration_sec >= 60 * min_minutes)
              & j.missing_channels.map(lambda m: {c.upper() for c in m} <= allowed))
    healthy = (tc.normative_mask(j) & (j.abnormal.fillna(0).astype(int) != 1)).fillna(False).astype(bool)
    pool = usable & (healthy if experiment.startswith("normal") else True)
    test = (j.source_split == "eval") & j.age.notna()
    return pool.fillna(False).astype(bool), test


def assign_splits(j, pool, test, experiment, tuab_split, *, val_fraction, seed,
                  patient_split=None):
    """Per-recording split ('train'/'val'/'test') or None (not in the cohort)."""
    tuab_subjects = set(tuab_split)
    split = pd.Series(None, index=j.index, dtype=object)
    other = sorted(set(j.loc[pool & ~j.subject.isin(tuab_subjects), "subject"]))
    if patient_split is not None:
        missing = [s for s in other if s not in patient_split]
        if missing:
            raise ValueError(f"{len(missing)} patients are not in the patient split "
                             f"(e.g. {missing[:3]})")
        other_split = {s: patient_split[s] for s in other}
    else:
        random.Random(seed).shuffle(other)
        n_val = int(round(val_fraction * len(other)))
        other_split = {s: ("val" if i < n_val else "train") for i, s in enumerate(other)}
    for i in j.index[pool]:
        subj = j.at[i, "subject"]
        if subj in tuab_subjects:
            if experiment == "normal_tueg" or tuab_split[subj] == "test":
                continue                       # outside TUAB only / TUAB-eval patient
            split[i] = tuab_split[subj]
        else:
            split[i] = other_split[subj]
    if experiment != "normal_tueg":
        split[test] = "test"
    return split


def cap_per_subject(j, split, max_per_subject, seed):
    """Keep at most ``max_per_subject`` train/val recordings per patient."""
    rng = random.Random(seed)
    keep = split.copy()
    for subj, idx in j[split.isin(["train", "val"])].groupby("subject").groups.items():
        idx = list(idx)
        if len(idx) > max_per_subject:
            rng.shuffle(idx)
            keep[idx[max_per_subject:]] = None
    return keep


def window_files(stem, n_windows, split, *, train_windows, trim_windows):
    """Window file names of a recording; training recordings are capped to
    ``train_windows`` evenly spaced windows inside the trimmed range."""
    ks = range(n_windows)
    if split == "train" and train_windows:
        inner = list(range(trim_windows, max(trim_windows, n_windows - trim_windows)))
        if len(inner) > train_windows:
            ks = sorted({inner[int(round(x))] for x in np.linspace(0, len(inner) - 1, train_windows)})
        else:
            ks = inner
    return [f"{split}/{stem}_{k}.pkl" for k in ks]


def build(args):
    idx = pd.read_parquet(args.index)
    j = tc.join_index(idx, tc.load_sheet(args.sheet))
    src_manifest = json.load(open(os.path.join(args.source, "manifest.json")))
    rows = {r["stem"]: r for r in src_manifest["recordings"]}
    j = j[j.stem.isin(rows)].reset_index(drop=True)          # converted recordings only
    tuab_split = tuab_subject_split(args.tuab_split)
    pool, test = select(j, args.experiment, min_age=args.min_age, max_age=args.max_age,
                        min_minutes=args.min_minutes, allow_missing=args.allow_missing)
    patient_split = (json.load(open(args.patient_split))["subjects"]
                     if args.patient_split else None)
    split = assign_splits(j, pool, test, args.experiment, tuab_split,
                          val_fraction=args.val_fraction, seed=args.seed,
                          patient_split=patient_split)
    split = cap_per_subject(j, split, args.max_per_subject, args.seed)
    j["split"] = split
    c = j[j.split.notna()].copy()

    trim = args.trim_sec // WINDOW_SEC
    train_windows = args.train_minutes * 60 // WINDOW_SEC
    files = {"train": [], "val": [], "test": []}
    manifest_rows, meta = [], {}
    for r in c.itertuples():
        src = dict(rows[r.stem])
        src["source_split"] = r.split
        manifest_rows.append(src)
        files[r.split] += window_files(r.stem, src["n_windows"], r.split,
                                       train_windows=train_windows, trim_windows=trim)
        meta[r.stem] = {"stem": r.stem, "subject": r.subject, "session": r.session,
                        "token": r.token, "age": int(r.age), "sex": r.sex,
                        "year": None if pd.isna(r.year) else int(r.year),
                        "raw_age": None if pd.isna(r.raw_age) else int(r.raw_age),
                        "label": label_of(r._asdict())}
    subjects = {s: set(c.loc[c.split == s, "subject"]) for s in files}
    for a in files:
        for b in files:
            if a < b and subjects[a] & subjects[b]:
                raise AssertionError(f"patients in both {a} and {b}: {sorted(subjects[a] & subjects[b])[:5]}")
    counts = {s: {"n_windows": len(files[s]), "n_recordings": int((c.split == s).sum()),
                  "n_subjects": len(subjects[s])} for s in files}
    manifest = {**{k: v for k, v in src_manifest.items() if k not in ("recordings", "errors", "excluded")},
                "n_recordings": len(manifest_rows),
                "n_windows": sum(r["n_windows"] for r in manifest_rows),
                "cohort": {"experiment": args.experiment, "source": args.source,
                           "seed": args.seed, "val_fraction": args.val_fraction,
                           "patient_split": args.patient_split,
                           "allow_missing": list(args.allow_missing),
                           "max_per_subject": args.max_per_subject,
                           "train_minutes": args.train_minutes, "min_age": args.min_age,
                           "max_age": args.max_age, "min_minutes": args.min_minutes,
                           "counts": counts,
                           "labels": {s: dict(Counter(meta[st]["label"] for st in c.loc[c.split == s, "stem"]))
                                      for s in files}},
                "recordings": manifest_rows}
    # An empty test list (not a missing key): the trainer then skips testing.
    return manifest, meta, files, counts, c


def write(args, manifest, meta, files, counts, c):
    out = args.out
    os.makedirs(os.path.join(out, "recordings"), exist_ok=True)
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    json.dump({"version": 1, "recordings": meta}, open(os.path.join(out, "age_metadata.json"), "w"))
    json.dump({"version": 1, "seed": args.seed, "val_fraction": args.val_fraction, "counts": counts,
               "files": files}, open(os.path.join(out, "age_split.json"), "w"))
    c[["stem", "subject", "split", "age", "sex", "corpora", "source_split"]].to_parquet(
        os.path.join(out, "cohort.parquet"), index=False)
    if args.link:
        missing = []
        for stem in c.stem:
            dst = os.path.join(out, "recordings", f"{stem}.npy")
            src = os.path.join(args.source, "recordings", f"{stem}.npy")
            if not os.path.exists(src):
                missing.append(stem)
                continue
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)             # same filesystem: no extra space
                except OSError:                   # another filesystem: symlink
                    os.symlink(os.path.abspath(src), dst)
        if missing:
            with open(os.path.join(out, "missing_recordings.txt"), "w") as fh:
                fh.write("\n".join(missing) + "\n")
            print(f"{len(missing)} recordings are not in the local source; listed in missing_recordings.txt")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    p.add_argument("--source", required=True, help="npy dir with manifest.json (local)")
    p.add_argument("--out", required=True)
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--sheet", default=DEFAULT_SHEET)
    p.add_argument("--tuab-split", default=DEFAULT_TUAB_SPLIT)
    p.add_argument("--val-fraction", type=float, default=0.1, help="for patients outside TUAB")
    p.add_argument("--patient-split", default=None,
                   help="shared TUEG patient split (dataset_maker/tueg_patient_split.py); "
                        "default: a seeded shuffle of this cohort's own patients (v1)")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-per-subject", type=int, default=10)
    p.add_argument("--train-minutes", type=int, default=20, help="per training recording; 0 = all")
    p.add_argument("--trim-sec", type=int, default=60, help="must match data.trim_start/end_sec")
    p.add_argument("--min-age", type=int, default=18)
    p.add_argument("--max-age", type=int, default=89)
    p.add_argument("--min-minutes", type=float, default=5.0)
    p.add_argument("--allow-missing", nargs="*", default=[], metavar="CHANNEL",
                   help="channels a recording may lack, e.g. T1 T2 (v3: 21-channel cohorts)")
    p.add_argument("--link", action="store_true", help="hard-link the recordings from --source")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    manifest, meta, files, counts, c = build(args)
    for s, v in counts.items():
        print(f"{s:5s}: {v['n_recordings']:6d} recordings  {v['n_subjects']:5d} patients  "
              f"{v['n_windows']:8d} windows  labels {manifest['cohort']['labels'][s]}")
    gb = sum(r["bytes"] for r in manifest["recordings"]) / 1e9
    print(f"total {manifest['n_recordings']} recordings, {gb:.1f} GB of npy")
    if not args.dry_run:
        write(args, manifest, meta, files, counts, c)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
