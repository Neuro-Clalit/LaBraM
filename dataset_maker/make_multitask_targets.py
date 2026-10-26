#!/usr/bin/env python
"""Per-recording classification targets for the multi-task brain-age head.

``model.head_type=multitask`` with ``data.multitask_targets`` predicts, next to
the age, the recording's clinical labels -- as **outputs**, never as inputs:

* ``status``: abnormal = 1 / normal = 0 (the TUAB folder label, else the NEDC
  report finding, as in ``make_condition_features.py``); NaN when the
  recording has neither (no label);
* ``pathology/<group>``: one 0/1 target per NEDC diagnosis group of the
  condition vector (clinical seizure, ..., other; ``PATHOLOGY`` without its
  "healthy or not available" entry); NaN for every group when the session is
  not in the NEDC clinical sheet, so absence from the sheet never counts as
  "no diagnosis".

Written as ``<root>/multitask_targets_v1.npz``: ``stems`` [N], ``targets``
[N, 1 + n_groups] float32 (status first, NaN = unknown) and ``names``. The loss
ignores the NaN entries (``losses/regression.py::MultiTaskAgeCriterion``), so
recordings with no label still train the age head.

    python -m dataset_maker.make_multitask_targets /data/.../cohorts/all_both_v3_anymiss
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from dataset_maker import tuh_clinical as tc
from dataset_maker.make_condition_features import DEFAULT_INDEX, DEFAULT_SHEET, PATHOLOGY

DEFAULT_NAME = "multitask_targets_v1.npz"
GROUPS = [k for k in PATHOLOGY if k != "healthy_or_not_available"]


def targets_for(stems, meta, clinical):
    """``[N, 1 + len(GROUPS)]`` float32 targets (see the module doc)."""
    info = clinical.drop_duplicates("stem").set_index("stem")
    out = np.full((len(stems), 1 + len(GROUPS)), np.nan, dtype=np.float32)
    for i, stem in enumerate(stems):
        label = (meta.get(stem) or {}).get("label")
        if label in ("normal", "abnormal"):
            out[i, 0] = float(label == "abnormal")
        if stem in info.index and bool(info.at[stem, "in_sheet"]):
            row = info.loc[stem]
            for k, group in enumerate(GROUPS, start=1):
                out[i, k] = float(any(bool(row[c]) if pd.notna(row[c]) else False
                                      for c in PATHOLOGY[group]))
    return out


def build(root, clinical, out_name=DEFAULT_NAME):
    rows = json.load(open(os.path.join(root, "manifest.json")))["recordings"]
    meta_path = os.path.join(root, "age_metadata.json")
    meta = json.load(open(meta_path))["recordings"] if os.path.exists(meta_path) else {}
    stems = [r["stem"] for r in rows]
    targets = targets_for(stems, meta, clinical)
    out = os.path.join(root, out_name)
    np.savez(out, stems=np.array(stems), targets=targets,
             names=np.array(["status_abnormal"] + [f"pathology/{g}" for g in GROUPS]))
    return out, targets


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="dataset dirs with manifest.json")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--sheet", default=DEFAULT_SHEET)
    p.add_argument("--name", default=DEFAULT_NAME)
    args = p.parse_args(argv)
    clinical = tc.join_index(pd.read_parquet(args.index), tc.load_sheet(args.sheet))
    for root in args.roots:
        out, t = build(root, clinical, args.name)
        known = np.isfinite(t)
        print(f"{out}: {len(t)} recordings; status known {int(known[:, 0].sum())} "
              f"(abnormal {int(np.nansum(t[:, 0]))}); pathology known {int(known[:, 1].sum())}; "
              + ", ".join(f"{g} {int(np.nansum(t[:, k]))}" for k, g in enumerate(GROUPS, start=1)))


if __name__ == "__main__":
    sys.exit(main())
