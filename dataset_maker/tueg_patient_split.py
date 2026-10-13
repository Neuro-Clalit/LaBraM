#!/usr/bin/env python
"""One train/val assignment per TUEG patient, shared by every TUH cohort.

TUAB patients keep their TUAB age split (``make_cohort_dataset --tuab-split``).
Every other TUEG patient gets one split here, written to a JSON file that
``make_cohort_dataset --patient-split`` reads, so a patient is in val for every
cohort or for none -- a model trained on one cohort can be scored on another
cohort's val without patient leakage.

* Anchors (``--anchor <age_split.json>``, repeatable): patients already split by
  an existing cohort keep that split, so models trained on it stay valid (v1:
  ``normal_tueg_v1``, whose TUEG split ``normal_both_v1`` shares).
* Everyone else: ``val`` iff ``sha256("<seed>:<subject>")`` / 2^256 <
  ``val_fraction`` -- stable under adding or removing other patients.

    python -m dataset_maker.tueg_patient_split \\
        --anchor /data/.../cohorts/normal_tueg_v1/age_split.json \\
        --out /data/.../cohorts/tueg_patient_split_v2.json
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import pandas as pd

DEFAULT_INDEX = "/data/datasets/EEG-public/TUH_EEG/index/tuh_index.parquet"
DEFAULT_TUAB_SPLIT = "/data/datasets/EEG-public/TUH_EEG/tuab/v3.0.1/edf/processed_npy/age_split.json"


def hash_split(subject: str, seed: int, val_fraction: float) -> str:
    h = int.from_bytes(hashlib.sha256(f"{seed}:{subject}".encode()).digest(), "big")
    return "val" if h / 2 ** 256 < val_fraction else "train"


def split_subjects(path: str) -> dict:
    """subject -> split of the windows listed in an age_split.json."""
    out = {}
    for split, names in json.load(open(path))["files"].items():
        for name in names:
            subj = os.path.basename(name).split("_")[0]
            if out.setdefault(subj, split) != split:
                raise ValueError(f"{path}: patient {subj} is in both {out[subj]} and {split}")
    return out


def build_patient_split(subjects, anchors, *, seed: int, val_fraction: float):
    """subject -> train/val for ``subjects``; anchored patients keep their split."""
    out, source = {}, Counter()
    for subj in sorted(subjects):
        if subj in anchors:
            out[subj] = anchors[subj]
            source["anchor"] += 1
        else:
            out[subj] = hash_split(subj, seed, val_fraction)
            source["hash"] += 1
    return out, dict(source)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--anchor", action="append", default=[], help="existing cohort age_split.json")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--tuab-split", default=DEFAULT_TUAB_SPLIT)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--val-fraction", type=float, default=0.1)
    args = p.parse_args(argv)
    tuab = set(split_subjects(args.tuab_split))
    subjects = set(pd.read_parquet(args.index, columns=["subject"]).subject.dropna()) - tuab
    anchors = {}
    for path in args.anchor:
        for subj, split in split_subjects(path).items():
            if subj in tuab:
                continue
            if split not in ("train", "val"):
                raise ValueError(f"{path}: TUEG patient {subj} has split {split!r}")
            if anchors.setdefault(subj, split) != split:
                raise ValueError(f"anchors disagree on patient {subj}")
    split, source = build_patient_split(subjects, anchors, seed=args.seed, val_fraction=args.val_fraction)
    counts = dict(Counter(split.values()))
    json.dump({"version": 1, "seed": args.seed, "val_fraction": args.val_fraction,
               "anchors": args.anchor, "source": source, "counts": counts, "subjects": split},
              open(args.out, "w"))
    print(f"{args.out}: {len(split)} TUEG patients outside TUAB {counts}; {source}")


if __name__ == "__main__":
    sys.exit(main())
