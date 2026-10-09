#!/usr/bin/env python
"""Scan TUH corpora, merge them into one recording index, and audit it.

Every TUH corpus is a subset of TUEG with the same subject ids and file stems,
so one recording can appear in several corpora (a TUAB recording is also a TUEG
recording). The index has **one row per stem**: the per-corpus labels are
merged onto it (``abnormal`` from TUAB, ``epilepsy`` from TUEP), ``corpora``
lists every corpus it came from, and ``source`` names the copy whose npy the
loader reads (TUAB first, since its npy already exists).

    # header-scan a local corpus (TUAB is local; TUEG chunks are scanned by
    # dataset_maker.download_tuh while downloading)
    python -m dataset_maker.tuh_index scan --root .../TUH_Abnormal/v3.0.0/edf \\
        --corpus tuab --out .../TUH_EEG/_state/tuab/recordings.parquet
    # merge the per-chunk scans of a downloaded corpus
    python -m dataset_maker.tuh_index merge-scans --scans .../_state/tueg/scans \\
        --out .../_state/tueg/recordings.parquet
    # one row per recording across corpora + the data report and leakage audit
    python -m dataset_maker.tuh_index build --corpus-scans .../_state/*/recordings.parquet \\
        --out .../TUH_EEG/index/tuh_index.parquet --report .../TUH_EEG/index/report.json
"""
import argparse
import glob
import json
import logging
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

from labram.data.tuh_metadata import scan_recordings

logger = logging.getLogger(__name__)

# Whose npy wins when a stem is in several corpora: the TUAB float32 npy
# already exists and is what every published TUAB number was measured on.
SOURCE_PRIORITY = ("tuab", "tuep", "tueg")
LABELS = ("abnormal", "epilepsy")
TUAB_SPLITS = ("train", "eval")


# ----------------------------------------------------------------- per corpus
def rows_to_frame(rows):
    df = pd.DataFrame(rows)
    if "error" not in df:
        df["error"] = None
    for col in LABELS:
        df[col] = pd.array(df[col] if col in df else [None] * len(df), dtype="Int8")
    # TUAB's own train/eval split, from the path's first component.
    first = df["path"].str.split("/").str[0]
    df["source_split"] = first.where(first.isin(TUAB_SPLITS) & (df["corpus"] == "tuab"))
    return df


def merge_scans(paths):
    rows = []
    for p in sorted(paths):
        with open(p) as fh:
            rows += json.load(fh)
    return rows_to_frame(rows)


# ---------------------------------------------------------------------- index
def build_index(frames):
    """Concatenate per-corpus scans into one row per stem (see module doc)."""
    df = pd.concat(frames, ignore_index=True)
    bad = df["error"].notna()
    if bad.any():
        logger.warning("dropping %d unreadable EDF(s)", int(bad.sum()))
    df = df[~bad].copy()
    rank = {c: i for i, c in enumerate(SOURCE_PRIORITY)}
    df["_rank"] = df["corpus"].map(lambda c: rank.get(c, len(rank)))
    df = df.sort_values(["stem", "_rank"])

    grouped = df.groupby("stem", sort=False)
    out = grouped.first()          # columns of the preferred copy
    out["source"] = out.pop("corpus")
    out["corpora"] = grouped["corpus"].agg(lambda s: ",".join(sorted(set(s))))
    for col in LABELS:             # a label from any corpus that has it
        out[col] = grouped[col].max()
    out["source_split"] = grouped["source_split"].first()
    # Copies of one recording must agree on demographics; report if not.
    conflicts = grouped["raw_age"].nunique(dropna=True)
    if (conflicts > 1).any():
        logger.warning("%d stem(s) with conflicting ages across corpora",
                       int((conflicts > 1).sum()))
    out["age_conflict"] = conflicts > 1
    return out.drop(columns=["_rank", "error"]).reset_index()


# --------------------------------------------------------------------- report
def _age_stats(ages):
    ages = ages.dropna()
    if ages.empty:
        return {"n": 0}
    return {"n": int(len(ages)), "mean": round(float(ages.mean()), 1),
            "std": round(float(ages.std()), 1), "min": int(ages.min()), "max": int(ages.max()),
            "median": float(ages.median())}


def _decades(ages):
    return {f"{int(d)}-{int(d) + 9}": int(n)
            for d, n in sorted(Counter((ages.dropna() // 10 * 10).clip(upper=90)).items())}


def summarize(index):
    """Counts and distributions behind the data report (all JSON-able)."""
    out = {"recordings": int(len(index)), "subjects": int(index["subject"].nunique()),
           "sessions": int(index[["subject", "session"]].drop_duplicates().shape[0])}
    per_corpus = {}
    for corpus in sorted({c for cs in index["corpora"] for c in cs.split(",")}):
        sub = index[index["corpora"].str.split(",").apply(lambda cs, c=corpus: c in cs)]
        per_corpus[corpus] = {
            "recordings": int(len(sub)), "subjects": int(sub["subject"].nunique()),
            "hours": round(float(sub["duration_sec"].sum()) / 3600, 1),
            "age": _age_stats(sub["age"]), "age_decades": _decades(sub["age"]),
            "redacted_90_plus": int((sub["raw_age"] == 999).sum()),
            "age_missing": int(sub["age"].isna().sum()),
            "sex": {str(k): int(v) for k, v in sub["sex"].value_counts(dropna=False).items()},
        }
    out["per_corpus"] = per_corpus
    for col in LABELS:
        lab = index[index[col].notna()]
        out[f"{col}_by_decade"] = {
            f"{int(dec)}-{int(dec) + 9}": {"n": int(len(g)), "positive": int(g[col].sum()),
                                           "rate": round(float(g[col].mean()), 3)}
            for dec, g in lab.groupby((lab["age"] // 10 * 10).clip(upper=90)) if len(g)}
    out["montage"] = {str(k): int(v) for k, v in index["montage"].value_counts(dropna=False).items()}
    out["sfreq"] = {str(k): int(v) for k, v in index["sfreq"].value_counts(dropna=False).items()}
    out["missing_channels"] = {
        ",".join(k) or "none": int(v)
        for k, v in Counter(tuple(m) for m in index["missing_channels"]).most_common(10)}
    out["duration_min_quantiles"] = {
        str(q): round(float(index["duration_sec"].quantile(q)) / 60, 1)
        for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)}
    out["recordings_per_subject"] = {
        str(k): int(v) for k, v in index.groupby("subject").size().describe().round(2).items()}
    return out


def leakage_audit(index):
    """Subject overlap between corpora and with the TUAB evaluation set.

    A TUAB-eval subject present anywhere in training would make TUAB numbers
    incomparable with published ones, so the split must hold them out globally.
    """
    subjects = {}
    for corpus in sorted({c for cs in index["corpora"] for c in cs.split(",")}):
        mask = index["corpora"].str.split(",").apply(lambda cs, c=corpus: c in cs)
        subjects[corpus] = set(index.loc[mask, "subject"])
    names = sorted(subjects)
    overlap = {a: {b: len(subjects[a] & subjects[b]) for b in names} for a in names}
    tuab_eval = set(index.loc[index["source_split"] == "eval", "subject"])
    tuab_train = set(index.loc[index["source_split"] == "train", "subject"])
    eval_recs = index[index["subject"].isin(tuab_eval)]
    return {
        "subject_overlap": overlap,
        "tuab_eval_subjects": len(tuab_eval),
        "tuab_train_eval_subject_overlap": len(tuab_eval & tuab_train),
        # Recordings of TUAB-eval subjects that are NOT TUAB-eval recordings:
        # other sessions in TUEG/TUEP which must stay out of training.
        "tuab_eval_subjects_other_recordings": int((eval_recs["source_split"] != "eval").sum()),
        "duplicate_stems_merged": int((index["corpora"].str.contains(",")).sum()),
        "age_conflicts": int(index["age_conflict"].sum()),
    }


# ------------------------------------------------------------------- commands
def write_frame(df, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"wrote {len(df)} rows -> {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan", help="header-scan a local corpus")
    s.add_argument("--root", required=True)
    s.add_argument("--corpus", required=True)
    s.add_argument("--pattern", default="**/*.edf")
    s.add_argument("--out", required=True)
    m = sub.add_parser("merge-scans", help="merge download_tuh per-chunk scans")
    m.add_argument("--scans", required=True, help="directory of <chunk>.json scans")
    m.add_argument("--out", required=True)
    b = sub.add_parser("build", help="one row per recording + report + leakage audit")
    b.add_argument("--corpus-scans", nargs="+", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--report", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "scan":
        write_frame(rows_to_frame(scan_recordings(args.root, corpus=args.corpus,
                                                  pattern=args.pattern)), args.out)
    elif args.command == "merge-scans":
        write_frame(merge_scans(glob.glob(os.path.join(args.scans, "*.json"))), args.out)
    else:
        index = build_index([pd.read_parquet(f) for f in args.corpus_scans])
        write_frame(index, args.out)
        report = {"summary": summarize(index), "leakage": leakage_audit(index)}
        text = json.dumps(report, indent=1, default=_json_default)
        if args.report:
            with open(args.report, "w") as fh:
                fh.write(text)
        print(json.dumps(report["leakage"], indent=1, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


if __name__ == "__main__":
    sys.exit(main())
