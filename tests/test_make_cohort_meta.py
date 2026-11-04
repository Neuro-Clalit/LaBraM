"""Tests for dataset_maker.make_cohort_meta: joining a cohort's meta sources."""
import pandas as pd

from dataset_maker import tuh_clinical as tc
from dataset_maker.make_cohort_meta import build_table


def _sources():
    cohort = pd.DataFrame({
        "stem": ["aaaaaaaa_s001_t000", "aaaaaaab_s003_t001"],
        "subject": ["aaaaaaaa", "aaaaaaab"], "split": ["train", "val"],
        "age": [37.0, 82.0], "sex": ["M", "F"], "corpora": ["tueg", "tuab,tueg"],
        "n_windows": [120, 60], "channel_mask": [[True] * 23, [True] * 21 + [False] * 2],
        "duration_sec": [1200.0, 600.0], "label": ["normal", "abnormal"],
    })
    index = pd.DataFrame({
        "stem": cohort["stem"], "session": ["s001", "s003"], "token": ["t000", "t001"],
        "year": [2015, 2004], "montage": ["01_tcp_ar", "03_tcp_ar_a"], "sfreq": [256.0, 250.0],
        "mixed_sfreq": [False, False], "n_signals": [31, 27],
        "missing_channels": [[], ["T1", "T2"]],
        "abnormal": pd.array([pd.NA, 1], dtype="Int8"), "epilepsy": pd.array([pd.NA, pd.NA], dtype="Int8"),
    })
    # Only the first session is in the clinical sheet.
    sheet = pd.DataFrame({"subject": ["aaaaaaaa"], "session": ["s001"],
                          **{col: [False] for col in tc.FLAGS}})
    sheet["dx_stroke"] = [True]
    headers = pd.DataFrame({"stem": ["aaaaaaaa_s001_t000"], "equipment_family": ["Compact_32ch"],
                            "sfreq": [256.0], "n_signals": [31], "duration_min": [20.0],
                            "has_ekg": [True]})
    return cohort, index, sheet, headers


def test_build_table_keeps_every_row_and_marks_missing_sources():
    df = build_table(*_sources()).set_index("stem")
    assert len(df) == 2
    first, second = df.loc["aaaaaaaa_s001_t000"], df.loc["aaaaaaab_s003_t001"]
    assert first["in_sheet"] and not second["in_sheet"]
    assert bool(first["dx_stroke"]) and pd.isna(second["dx_stroke"])     # unknown, not False
    assert first["in_header_dump"] and not second["in_header_dump"]
    assert first["sfreq_header"] == 256.0 and "channel_mask" not in df
    assert second["session_number"] == 3 and second["token_number"] == 1
    assert second["n_missing_channels"] == 2 and second["missing_channels"] == "T1,T2"
    assert second["in_tuab"] and not first["in_tuab"]
    assert first["duration_min"] == 20.0
