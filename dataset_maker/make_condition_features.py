#!/usr/bin/env python
"""Per-recording clinical condition vectors (model.session_features=head).

Two one-/multi-hot groups per recording, in the session-feature file format
(``labram/data/session_features.py``), so the loader hands them to the model:

* ``status`` [3]: normal / abnormal / not available -- the recording's label
  in the cohort's ``age_metadata.json`` (TUAB folder label, else the NEDC
  report finding; no label -> not available).
* ``pathology`` [14], multi-hot: index 0 = healthy or not available (no
  diagnosis flagged, or the session is not in the NEDC sheet); then one entry
  per diagnosis of the NEDC clinical sheet (``dataset_maker/tuh_clinical.py``):
  clinical seizure, electrographic seizure, epilepsy, stroke, TBI, substance
  use, bipolar, anxiety, depression, schizophrenia, dementia (incl.
  Alzheimer's), multiple sclerosis, other (Parkinson's, ADHD -- under 50
  recordings each).

Nothing is standardized: the vectors are categorical. Written for each dataset
dir given (its ``manifest.json`` lists the recordings):

    python -m dataset_maker.make_condition_features \\
        /data/.../cohorts/all_both_v2 /data/.../tuab/v3.0.1/edf/processed_npy_interp50-60
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from dataset_maker import tuh_clinical as tc

DEFAULT_INDEX = "/data/datasets/EEG-public/TUH_EEG/index/tuh_index.parquet"
DEFAULT_SHEET = "/data/datasets/EEG-public/TUH_EEG/index/nedc/metadata_v12.xlsm.xlsx"
DEFAULT_NAME = "conditions_v1.npz"
STATUS = ("normal", "abnormal", "not_available")
PATHOLOGY = {
    "healthy_or_not_available": (),
    "clinical_seizure": ("dx_clinical_seizure",),
    "electrographic_seizure": ("dx_electrographic_seizure",),
    "epilepsy": ("dx_epilepsy",),
    "stroke": ("dx_stroke",),
    "tbi": ("dx_tbi",),
    "substance": ("dx_substance",),
    "bipolar": ("dx_bipolar",),
    "anxiety": ("dx_anxiety",),
    "depression": ("dx_depression",),
    "schizophrenia": ("dx_schizophrenia",),
    "dementia": ("dx_dementia", "dx_alzheimers"),
    "ms": ("dx_ms",),
    "other": ("dx_parkinsons", "dx_adhd"),
}


def status_vector(label):
    v = np.zeros(len(STATUS), np.float32)
    v[STATUS.index(label if label in ("normal", "abnormal") else "not_available")] = 1
    return v


def pathology_vector(flags):
    """``flags``: dx column -> True/False/<NA>; <NA> (not in the sheet) counts as absent."""
    v = np.zeros(len(PATHOLOGY), np.float32)
    for k, cols in enumerate(PATHOLOGY.values()):
        if k and any(bool(flags.get(c)) if pd.notna(flags.get(c)) else False for c in cols):
            v[k] = 1
    if not v[1:].any():
        v[0] = 1
    return v


def build(root, clinical, out_name=DEFAULT_NAME):
    rows = json.load(open(os.path.join(root, "manifest.json")))["recordings"]
    meta_path = os.path.join(root, "age_metadata.json")
    meta = json.load(open(meta_path))["recordings"] if os.path.exists(meta_path) else {}
    stems = [r["stem"] for r in rows]
    dx_cols = [c for cols in PATHOLOGY.values() for c in cols]
    flags = clinical.set_index("stem")[dx_cols]
    flags = flags[~flags.index.duplicated()]
    status = np.stack([status_vector((meta.get(s) or {}).get("label")) for s in stems])
    pathology = np.stack([pathology_vector(flags.loc[s].to_dict() if s in flags.index else {})
                          for s in stems])
    out = os.path.join(root, out_name)
    np.savez(out, stems=np.array(stems), groups=np.array(["status", "pathology"]),
             **{"tokens/status": status[:, None, :], "tokens/pathology": pathology[:, None, :]},
             meta=np.array(json.dumps({"status": list(STATUS), "pathology": list(PATHOLOGY)})))
    return out, status, pathology


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="dataset dirs with manifest.json")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--sheet", default=DEFAULT_SHEET)
    p.add_argument("--name", default=DEFAULT_NAME)
    args = p.parse_args(argv)
    clinical = tc.join_index(pd.read_parquet(args.index), tc.load_sheet(args.sheet))
    for root in args.roots:
        out, status, pathology = build(root, clinical, args.name)
        print(f"{out}: {len(status)} recordings; status "
              + ", ".join(f"{n} {int(c)}" for n, c in zip(STATUS, status.sum(0)))
              + "; pathology " + ", ".join(f"{n} {int(c)}" for n, c in zip(PATHOLOGY, pathology.sum(0))))


if __name__ == "__main__":
    sys.exit(main())
