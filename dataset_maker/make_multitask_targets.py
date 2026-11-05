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

A joint TUH + HEEDB cohort (``make_joint_cohort.py``) takes the HEEDB labels
with ``--extra-clinical <heedb cohort>/labels.parquet`` and is written as
``multitask_targets_v2.npz`` (``--name``). For HEEDB recordings (stems
``sub-...``) the diagnosis groups come from the TUH-aligned ``dx_*`` flags,
which are per-patient ICD-10 codes:

* a patient without any health record has every group unknown (NaN), never 0;
* a group the neurology ICD extract never codes (no positive in the whole
  file, e.g. TBI or the psychiatric groups) is unknown for every HEEDB
  recording;
* ``--extra-groups`` adds HEEDB ICD-10 groups (``icd_*`` columns, e.g.
  headache) as further targets, unknown for TUH recordings.

    python -m dataset_maker.make_multitask_targets /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1 \\
        --extra-clinical /data/datasets/HEEDB/v4.1/cohorts/heedb_routine_v1/labels.parquet \\
        --extra-groups headache_disorders sleep_disorders intracranial_and_spinal_tumors \\
                       movement_and_cerebellar_disorders --name multitask_targets_v2.npz
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


def heedb_targets_for(stems, meta, labels, extra_groups=()):
    """``[N, 1 + len(GROUPS) + len(extra_groups)]`` targets of HEEDB recordings
    from a HEEDB cohort's ``labels.parquet`` (see the module doc)."""
    info = labels.drop_duplicates("stem").set_index("stem")
    # The ICD-based dx flags are known exactly for patients with a health record
    # (heedb_index.health_record); dx_electrographic_seizure comes from the report.
    record_cols = [c for c in info.columns if c.startswith("dx_") and c != "dx_electrographic_seizure"]
    coded = {g: any(c in info and info[c].astype("boolean").fillna(False).any()
                    for c in PATHOLOGY[g]) for g in GROUPS}
    out = np.full((len(stems), 1 + len(GROUPS) + len(extra_groups)), np.nan, dtype=np.float32)
    for i, stem in enumerate(stems):
        label = (meta.get(stem) or {}).get("label")
        if label in ("normal", "abnormal"):
            out[i, 0] = float(label == "abnormal")
        if stem not in info.index:
            continue
        row = info.loc[stem]
        if not any(pd.notna(row[c]) for c in record_cols):       # no health record
            continue
        for k, group in enumerate(GROUPS, start=1):
            if coded[group]:
                out[i, k] = float(any(bool(row[c]) for c in PATHOLOGY[group]
                                      if c in row and pd.notna(row[c])))
        for k, group in enumerate(extra_groups, start=1 + len(GROUPS)):
            v = row.get(f"icd_{group}")
            out[i, k] = float(bool(v)) if pd.notna(v) else 0.0
    return out


def build(root, clinical, out_name=DEFAULT_NAME, extra=None, extra_groups=()):
    rows = json.load(open(os.path.join(root, "manifest.json")))["recordings"]
    meta_path = os.path.join(root, "age_metadata.json")
    meta = json.load(open(meta_path))["recordings"] if os.path.exists(meta_path) else {}
    stems = [r["stem"] for r in rows]
    is_heedb = np.array([s.startswith("sub-") for s in stems]) if extra is not None else \
        np.zeros(len(stems), bool)
    targets = np.full((len(stems), 1 + len(GROUPS) + len(extra_groups)), np.nan, np.float32)
    tuh = [s for s, h in zip(stems, is_heedb) if not h]
    if tuh:
        targets[~is_heedb, :1 + len(GROUPS)] = targets_for(tuh, meta, clinical)
    if is_heedb.any():
        targets[is_heedb] = heedb_targets_for([s for s, h in zip(stems, is_heedb) if h], meta,
                                              extra, extra_groups)
    out = os.path.join(root, out_name)
    np.savez(out, stems=np.array(stems), targets=targets,
             names=np.array(["status_abnormal"] + [f"pathology/{g}" for g in GROUPS]
                            + [f"pathology/{g}" for g in extra_groups]))
    return out, targets


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="dataset dirs with manifest.json")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--sheet", default=DEFAULT_SHEET)
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--extra-clinical", default=None,
                   help="a HEEDB cohort's labels.parquet, for the HEEDB recordings of a joint cohort")
    p.add_argument("--extra-groups", nargs="*", default=[],
                   help="HEEDB ICD-10 groups (icd_<name> columns) added as targets")
    args = p.parse_args(argv)
    clinical = tc.join_index(pd.read_parquet(args.index), tc.load_sheet(args.sheet))
    extra = pd.read_parquet(args.extra_clinical) if args.extra_clinical else None
    for root in args.roots:
        out, t = build(root, clinical, args.name, extra, tuple(args.extra_groups))
        names = list(np.load(out)["names"])
        known = np.isfinite(t)
        print(f"{out}: {len(t)} recordings")
        for k, name in enumerate(names):
            print(f"  {name:45s} known {int(known[:, k].sum()):6d}  positive {int(np.nansum(t[:, k])):6d}")


if __name__ == "__main__":
    sys.exit(main())
