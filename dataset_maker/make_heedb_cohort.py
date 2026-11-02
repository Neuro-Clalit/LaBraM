#!/usr/bin/env python
"""Build a HEEDB training cohort in S3 in the TUH cohort format.

The cohort is read by the same loader as the TUH cohorts (``data.dataset=TUAB_AGE``,
``data.data_format=npy``): ``manifest.json`` (the source rows with
``source_split`` = cohort split), ``age_metadata.json``, ``age_split.json``
(window names ``<split>/<stem>_<k>.pkl``) and ``recordings/<stem>.npy``.
SageMaker mounts one dataset prefix per job, so the selected recordings are
copied server-side (S3 -> S3, no local disk) next to the sidecars.

Selection (``--experiment``):

* ``routine``: every converted adult (18-89) routine session.
* ``normal``: report-normal sessions without a *known* neurological diagnosis
  (``heedb_analysis.normative_counts``: no neurological ICD code or seizure
  finding; patients without a health record pass).

Split: ``index/heedb_patient_split.json`` (``heedb_index patient-split``), keyed
by BDSP patient id, with train / val / held-out test. At most
``--max-per-subject`` train/val recordings per patient and ``--train-minutes``
per training recording, as ``make_cohort_dataset``.

The recordings in train and val come from different patients, and the build
asserts that. ``labels.parquet`` holds the per-recording multi-label targets
(report findings, finding groups, ICD-10 categories, TUH-aligned flags; NA =
unknown) for the multi-task heads.

    python -m dataset_maker.make_heedb_cohort --experiment routine \\
        --source s3://eeg-data-public/HEEDB/v4.1/processed_npy_interp50-60_flat0.5_float16 \\
        --out s3://eeg-data-public/HEEDB/v4.1/cohorts/heedb_routine_v1
"""
import argparse
import concurrent.futures as cf
import io
import json
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset_maker import heedb_index as hx  # noqa: E402
from dataset_maker.make_cohort_dataset import WINDOW_SEC, cap_per_subject, window_files  # noqa: E402

EXPERIMENTS = ("routine", "normal")
LABEL_PREFIXES = ("find_", "fg_", "icd_", "dx_", "f_", "med_")


def split_s3(uri):
    bucket, _, key = uri[5:].partition("/")
    return bucket, key.rstrip("/")


def select(index: pd.DataFrame, rows: dict, experiment: str, *, min_minutes: float) -> pd.DataFrame:
    """Converted adult routine sessions of the experiment, one row per stem."""
    j = index[index["stem"].isin(rows) & index["routine"].fillna(False) & index["adult"].fillna(False)]
    j = j.drop_duplicates("stem").copy()
    j["n_windows"] = j["stem"].map(lambda s: rows[s]["n_windows"])
    j = j[j["n_windows"] * WINDOW_SEC / 60 >= min_minutes]
    if experiment == "normal":
        j = j[j["in_sheet"] & ~j["abnormal_report"].fillna(True).astype(bool)
              & ~j["any_neuro_dx"].fillna(False).astype(bool)]
    return j.reset_index(drop=True)


def label_table(j: pd.DataFrame) -> pd.DataFrame:
    """Per-recording targets for multi-task heads (nullable; NA = unknown)."""
    cols = [c for c in j.columns if c.startswith(LABEL_PREFIXES) and not c.endswith("_codes")]
    out = j[["stem", "split", "age", "sex", "site", "label"]].copy()
    for c in cols:
        out[c] = j[c].astype("boolean")
    return out


def build(index, src_manifest, patient_split, args):
    rows = {r["stem"]: r for r in src_manifest["recordings"]}
    j = select(index, rows, args.experiment, min_minutes=args.min_minutes)
    patients = patient_split["patients"]
    j["split"] = j["patient_id"].map(lambda p: patients.get(str(int(p))) if pd.notna(p) else None)
    j = j[j["split"].notna()].reset_index(drop=True)
    # cap_per_subject groups by "subject"; use the patient id so cross-site
    # sessions of one patient count together.
    capped = cap_per_subject(j.assign(subject=j["patient_id"].astype(str)), j["split"],
                             args.max_per_subject, args.seed)
    j["split"] = capped
    j = j[j["split"].notna()].reset_index(drop=True)

    trim = args.trim_sec // WINDOW_SEC
    train_windows = args.train_minutes * 60 // WINDOW_SEC
    files = {"train": [], "val": [], "test": []}
    manifest_rows, meta = [], {}
    for r in j.itertuples():
        src = dict(rows[r.stem])
        src["source_split"] = r.split
        manifest_rows.append(src)
        files[r.split] += window_files(r.stem, src["n_windows"], r.split,
                                       train_windows=train_windows, trim_windows=trim)
        meta[r.stem] = {"stem": r.stem, "subject": str(int(r.patient_id)), "session": str(r.session),
                        "token": None, "age": int(r.age), "sex": r.sex,
                        "year": None if pd.isna(r.year) else int(r.year),
                        "raw_age": int(r.age), "label": r.label if isinstance(r.label, str) else None}
    pats = {s: set(j.loc[j["split"] == s, "patient_id"].astype(int)) for s in files}
    for a in files:
        for b in files:
            if a < b and pats[a] & pats[b]:
                raise AssertionError(f"patients in both {a} and {b}: {sorted(pats[a] & pats[b])[:5]}")
    counts = {s: {"n_windows": len(files[s]), "n_recordings": int((j["split"] == s).sum()),
                  "n_subjects": len(pats[s])} for s in files}
    manifest = {**{k: v for k, v in src_manifest.items() if k not in ("recordings", "errors", "excluded")},
                "n_recordings": len(manifest_rows),
                "n_windows": sum(r["n_windows"] for r in manifest_rows),
                "cohort": {"experiment": args.experiment, "source": args.source, "corpus": "heedb",
                           "patient_split": "index/heedb_patient_split.json",
                           "max_per_subject": args.max_per_subject, "train_minutes": args.train_minutes,
                           "trim_sec": args.trim_sec, "min_minutes": args.min_minutes, "counts": counts,
                           "labels": {s: dict(Counter(j.loc[j["split"] == s, "label"].fillna("unlabelled")))
                                      for s in files}},
                "recordings": manifest_rows}
    return manifest, meta, files, counts, j


def write(args, client, manifest, meta, files, counts, j):
    bucket, prefix = split_s3(args.out)
    sbucket, sprefix = split_s3(args.source)

    def put(name, data):
        client.put_object(Bucket=bucket, Key=f"{prefix}/{name}", Body=data)

    put("manifest.json", json.dumps(manifest, indent=1).encode())
    put("age_metadata.json", json.dumps({"version": 1, "recordings": meta}).encode())
    put("age_split.json", json.dumps({"version": 1, "seed": args.seed, "counts": counts,
                                      "files": files}).encode())
    for name, frame in (("cohort.parquet", j[["stem", "patient_id", "subject", "site", "split", "age",
                                              "sex", "label"]]),
                        ("labels.parquet", label_table(j))):
        buf = io.BytesIO()
        frame.to_parquet(buf, index=False)
        put(name, buf.getvalue())

    def copy(stem):
        client.copy_object(Bucket=bucket, Key=f"{prefix}/recordings/{stem}.npy",
                           CopySource={"Bucket": sbucket, "Key": f"{sprefix}/recordings/{stem}.npy"})
        return stem

    stems = [r["stem"] for r in manifest["recordings"]]
    if args.skip_copy:
        return 0
    with cf.ThreadPoolExecutor(args.threads) as pool:
        n = sum(1 for _ in pool.map(copy, stems))
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    p.add_argument("--source", required=True, help="s3:// npy dir with manifest.json")
    p.add_argument("--out", required=True, help="s3:// cohort dir")
    p.add_argument("--root", default=hx.DEFAULT_ROOT)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-per-subject", type=int, default=10)
    p.add_argument("--train-minutes", type=int, default=20, help="per training recording; 0 = all")
    p.add_argument("--trim-sec", type=int, default=60, help="must match data.trim_start/end_sec")
    p.add_argument("--min-minutes", type=float, default=5.0, help="after flat-window removal")
    p.add_argument("--threads", type=int, default=128)
    p.add_argument("--skip-copy", action="store_true", help="write the sidecars only")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    for uri in (args.source, args.out):
        if not uri.startswith("s3://"):
            raise SystemExit(f"{uri}: expected an s3:// path")
    from dataset_maker.download_heedb import check_private
    client = hx.s3_client(None)
    check_private(client, split_s3(args.out)[0])
    sb, sp = split_s3(args.source)
    src_manifest = json.loads(client.get_object(Bucket=sb, Key=f"{sp}/manifest.json")["Body"].read())
    index = pd.read_parquet(os.path.join(args.root, "index", "heedb_index.parquet"))
    with open(os.path.join(args.root, "index", "heedb_patient_split.json")) as fh:
        patient_split = json.load(fh)
    manifest, meta, files, counts, j = build(index, src_manifest, patient_split, args)
    print(json.dumps({"experiment": args.experiment, "counts": counts,
                      "labels": manifest["cohort"]["labels"]}, indent=1))
    if args.dry_run:
        return
    n = write(args, client, manifest, meta, files, counts, j)
    print(f"wrote {args.out}: sidecars + {n} recordings copied")


if __name__ == "__main__":
    sys.exit(main())
