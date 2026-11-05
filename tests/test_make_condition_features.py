"""Clinical condition vectors (dataset_maker/make_condition_features.py) for a
joint TUH + HEEDB cohort: TUH flags from the NEDC sheet, HEEDB flags from its
labels.parquet."""
import json

import numpy as np
import pandas as pd
import pytest

from dataset_maker import make_condition_features as mc


def test_build_joint_cohort_status_and_pathology(tmp_path):
    stems = ["aaaa_s001_t000", "sub-S00011_ses-1_task-EEG", "sub-S00012_ses-1_task-EEG"]
    (tmp_path / "manifest.json").write_text(json.dumps({"recordings": [{"stem": s} for s in stems]}))
    (tmp_path / "age_metadata.json").write_text(json.dumps({"recordings": {
        stems[0]: {"label": "abnormal"}, stems[1]: {"label": "normal"}, stems[2]: {"label": None}}}))
    tuh = pd.DataFrame({"stem": [stems[0]], "dx_epilepsy": [True], "dx_stroke": [False]})
    heedb = pd.DataFrame({"stem": stems[1:], "dx_stroke": pd.array([True, pd.NA], dtype="boolean"),
                          "dx_dementia": pd.array([True, pd.NA], dtype="boolean"), "site": ["S0001"] * 2})
    clinical = mc.merge_clinical(tuh, [heedb])
    _, status, path = mc.build(str(tmp_path), clinical)
    names = list(mc.PATHOLOGY)
    assert status.tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    assert path[0, names.index("epilepsy")] == 1 and path[0].sum() == 1
    assert path[1, names.index("stroke")] == 1 and path[1, names.index("dementia")] == 1
    assert path[2, 0] == 1 and path[2].sum() == 1          # no health record -> healthy/NA
    z = np.load(tmp_path / mc.DEFAULT_NAME)
    assert z["tokens/status"].shape == (3, 1, 3) and z["tokens/pathology"].shape == (3, 1, 14)


def test_merge_refuses_a_stem_in_two_sources():
    tuh = pd.DataFrame({"stem": ["x"], "dx_epilepsy": [True]})
    with pytest.raises(ValueError, match="more than one"):
        mc.merge_clinical(tuh, [pd.DataFrame({"stem": ["x"], "dx_stroke": [True]})])
