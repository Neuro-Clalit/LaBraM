"""Multi-task targets of a joint TUH + HEEDB cohort: HEEDB groups from ICD-based
dx flags, unknown (NaN) without a health record or for groups never coded."""
import json

import numpy as np
import pandas as pd

from dataset_maker import make_multitask_targets as mt


def test_joint_targets_semantics(tmp_path):
    tuh, h1, h2 = "aaaa_s001_t000", "sub-S00011_ses-1_task-EEG", "sub-S00012_ses-1_task-EEG"
    (tmp_path / "manifest.json").write_text(json.dumps({"recordings": [{"stem": s} for s in (tuh, h1, h2)]}))
    (tmp_path / "age_metadata.json").write_text(json.dumps({"recordings": {
        tuh: {"label": "abnormal"}, h1: {"label": "normal"}, h2: {"label": None}}}))
    clinical = pd.DataFrame({"stem": [tuh], "in_sheet": [True],
                             **{c: [c == "dx_epilepsy"] for cols in mt.PATHOLOGY.values() for c in cols}})
    b = pd.array
    labels = pd.DataFrame({"stem": [h1, h2],
                           "dx_epilepsy": b([True, pd.NA], dtype="boolean"),
                           "dx_tbi": b([False, pd.NA], dtype="boolean"),        # never coded in HEEDB
                           "dx_stroke": b([False, pd.NA], dtype="boolean"),
                           "icd_headache_disorders": b([True, pd.NA], dtype="boolean"),
                           "icd_sleep_disorders": b([False, pd.NA], dtype="boolean")})
    out, t = mt.build(str(tmp_path), clinical, "mt_v2.npz", labels, ("headache_disorders",))
    names = list(np.load(out)["names"])
    col = {n: i for i, n in enumerate(names)}
    assert names[-1] == "pathology/headache_disorders" and t.shape == (3, 1 + len(mt.GROUPS) + 1)
    assert t[0, col["status_abnormal"]] == 1 and t[0, col["pathology/epilepsy"]] == 1
    assert np.isnan(t[0, col["pathology/headache_disorders"]])          # TUH: not coded
    assert t[1, col["status_abnormal"]] == 0 and t[1, col["pathology/epilepsy"]] == 1
    assert t[1, col["pathology/headache_disorders"]] == 1
    assert np.isnan(t[1, col["pathology/tbi"]])                          # never coded in HEEDB
    assert np.isnan(t[2, 1:]).all() and np.isnan(t[2, 0])                # no record, no label
