"""Tests for eval.meta_analysis on a synthetic meta-feature table."""
import numpy as np
import pandas as pd
import pytest

from labram.eval.meta_analysis import (
    COHORT_FEATURE_GROUPS,
    add_derived_columns,
    available_features,
    metadata_age_baseline,
    metadata_baseline,
    report_reason_flags,
    signature_by_condition,
    signature_flags,
    univariate_auc,
)


@pytest.fixture
def table():
    rng = np.random.default_rng(0)
    n = 400
    abnormal = rng.random(n) < 0.5
    return add_derived_columns(pd.DataFrame({
        "label": np.where(abnormal, "abnormal", "normal"),
        "sex": rng.choice(["M", "F"], n),
        "age": rng.integers(18, 90, n),
        # Slowing separates the classes; PDR is lower in abnormals; HR is noise.
        "qeeg_slowing_ratio": rng.normal(0, 1, n) + 2.0 * abnormal,
        "qeeg_pdr_freq": rng.normal(10, 0.7, n) - 2.0 * abnormal,
        "ekg_hr_bpm": np.where(rng.random(n) < 0.1, np.nan, rng.normal(70, 10, n)),
        "has_emg": rng.random(n) < 0.6,
    }))


def test_derived_columns(table):
    assert set(table["is_abnormal"]) == {0, 1}
    assert table["has_emg"].dtype == float
    assert table["age_group"].notna().all()


def test_univariate_auc_ranks_informative_features_and_orients_direction(table):
    res = univariate_auc(table, ["qeeg_slowing_ratio", "qeeg_pdr_freq", "ekg_hr_bpm"]).set_index("feature")
    assert res.loc["qeeg_slowing_ratio", "auc"] > 0.8
    assert res.loc["qeeg_slowing_ratio", "direction"] == "higher=abnormal"
    assert res.loc["qeeg_pdr_freq", "direction"] == "lower=abnormal"
    assert res.loc["ekg_hr_bpm", "auc"] < 0.6
    assert res.loc["ekg_hr_bpm", "coverage"] < 1.0


def test_signature_flags_hit_few_normals_and_many_abnormals(table):
    flags = signature_flags(table, table["is_abnormal"] == 0)
    normal = table["is_abnormal"] == 0
    assert flags.loc[normal, "diffuse_slowing"].mean() <= 0.10
    assert flags.loc[~normal, "diffuse_slowing"].mean() > 0.5
    assert flags.loc[~normal, "slow_pdr"].mean() > 0.5
    assert "asymmetry" not in flags                        # column absent -> skipped


def test_metadata_baseline_learns_signal(table):
    feats = available_features(table, ["demographics", "qeeg", "physiology"])
    metrics, _ = metadata_baseline(table.iloc[:300], table.iloc[300:], feats)
    assert metrics["auroc"] > 0.85


def test_cohort_columns_keep_unlabelled_as_nan_and_encode_setup():
    df = add_derived_columns(pd.DataFrame({
        "label": ["normal", "abnormal", None],
        "sex": ["M", "F", "M"],
        "age": [30, 60, 45],
        "montage": ["01_tcp_ar", "02_tcp_le", "03_tcp_ar_a"],
        "equipment_family": ["Compact_32ch", "Intution/Ultrasom", "unknown"],
        "dx_stroke": pd.array([False, True, pd.NA], dtype="boolean"),
    }))
    assert table_is_nan(df["is_abnormal"].iloc[2]) and list(df["is_abnormal"].iloc[:2]) == [0, 1]
    assert list(df["montage_le"]) == [0, 1, 0] and list(df["montage_reduced"]) == [0, 0, 1]
    assert df["equip_compact_32ch"].tolist() == [1, 0, 0]
    assert df["equip_intution_ultrasom"].tolist() == [0, 1, 0]
    assert df["dx_stroke"].iloc[1] == 1.0 and table_is_nan(df["dx_stroke"].iloc[2])
    feats = available_features(df, ["acquisition", "clinical"], COHORT_FEATURE_GROUPS)
    assert {"montage_le", "equip_bravo", "dx_stroke"} <= set(feats)


def table_is_nan(value):
    return value != value


def test_signature_by_condition_reports_lift_and_skips_rare_conditions():
    rng = np.random.default_rng(1)
    n = 600
    stroke = rng.random(n) < 0.3
    flags = pd.DataFrame({"asymmetry": np.where(stroke, rng.random(n) < 0.6, rng.random(n) < 0.1),
                          "slow_pdr": rng.random(n) < 0.2})
    conditions = pd.DataFrame({"dx_stroke": stroke.astype(float), "dx_rare": (np.arange(n) < 5).astype(float)})
    conditions.loc[:9, "dx_stroke"] = np.nan              # unknown rows are left out
    res = signature_by_condition(flags, conditions).set_index(["condition", "signature"])
    assert "dx_rare" not in res.index.get_level_values(0)
    assert res.loc[("dx_stroke", "asymmetry"), "lift"] > 3
    assert 0.5 < res.loc[("dx_stroke", "slow_pdr"), "lift"] < 2
    assert res.loc[("dx_stroke", "asymmetry"), "n_positive"] == int(stroke[10:].sum())


def test_metadata_age_baseline_beats_mean_predictor():
    rng = np.random.default_rng(2)
    n = 500
    age = rng.uniform(18, 90, n)
    df = pd.DataFrame({"age": age, "qeeg_pdr_freq": 11 - age / 30 + rng.normal(0, 0.2, n),
                       "noise": rng.normal(0, 1, n)})
    metrics, _ = metadata_age_baseline(df.iloc[:400], df.iloc[400:], ["qeeg_pdr_freq", "noise"])
    assert metrics["mae"] < 0.5 * metrics["mae_mean"]
    assert metrics["pearson_r"] > 0.9 and metrics["n_features"] == 2


def test_report_reason_flags_are_multi_label():
    text = pd.Series([
        "Generalized background slowing. Leftt temporal sharp waves.",
        "Intermittent focal slowing in the right midtemporal region.",
        "Absence of cerebral activity.",
        "Excess beta.",
        "Adult onset nocturnal spells.",
        None,
    ])
    flags = report_reason_flags(text)
    assert flags.loc[0, "diffuse_slowing"] and flags.loc[0, "epileptiform"]
    assert flags.loc[1, "focal_slowing"] and not flags.loc[1, "epileptiform"]
    assert flags.loc[2, "suppression"] and flags.loc[3, "excess_beta"]
    assert flags.loc[4, "reason_unmatched"]
    assert not flags.loc[5].any()                          # no text: nothing, not "unmatched"
