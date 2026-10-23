"""Tests for eval.meta_analysis on a synthetic meta-feature table."""
import numpy as np
import pandas as pd
import pytest

from labram.eval.meta_analysis import (
    add_derived_columns,
    available_features,
    metadata_baseline,
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
