"""scripts/pathology_brain_age.py (adjusted group gap, small-cell suppression) and
scripts/window_confidence_figures.py (window-level quantities)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _cohort(rng, n=4000, effect=2.0):
    """Recordings whose gap falls with age (compression) and where group 'g' adds ``effect`` years."""
    age = rng.uniform(18, 90, n)
    in_group = rng.random(n) < np.clip((age - 30) / 80, 0.05, 0.6)          # the group is older
    gap = -0.2 * (age - 50) + effect * in_group + 4 * rng.standard_normal(n)
    return pd.DataFrame({"subject": [f"s{i // 2}" for i in range(n)], "age": age, "pred": age + gap, "gap": gap,
                         "sex": rng.choice(["M", "F"], n), "g": in_group.astype(float),
                         "label": np.where(in_group & (rng.random(n) < 0.5), "abnormal", "normal"),
                         "reference": ~in_group})


def test_adjusted_gap_recovers_a_group_effect_that_the_raw_gap_hides():
    import pathology_brain_age as pb
    d = _cohort(np.random.default_rng(0))
    raw = d.loc[d["g"] == 1, "gap"].mean() - d.loc[d["reference"], "gap"].mean()
    out = pb.adjusted(d, d["g"] == 1, d["reference"])
    assert raw < 1.0                                    # the older group's raw gap is pulled down by the age bias
    assert out["gap_adj_lo"] < 2.0 < out["gap_adj_hi"]
    assert abs(out["gap_adj"] - 2.0) < 0.5 and out["gap_adj_p"] < 1e-6
    d0 = _cohort(np.random.default_rng(1), effect=0.0)
    null = pb.adjusted(d0, d0["g"] == 1, d0["reference"])
    assert null["gap_adj_lo"] < 0.0 < null["gap_adj_hi"]


def test_cluster_robust_errors_are_wider_than_naive_ones_for_repeated_patients():
    import pathology_brain_age as pb
    rng = np.random.default_rng(0)
    subj = np.repeat(np.arange(200), 5)
    x = np.repeat(rng.integers(0, 2, 200), 5).astype(float)                 # the group is a property of the patient
    y = np.repeat(rng.standard_normal(200), 5) + 0.3 * rng.standard_normal(1000)
    X = np.column_stack([x, np.ones(1000)])
    beta, se, g = pb.cluster_ols(y, X, subj)
    resid = y - X @ beta
    naive = np.sqrt(resid @ resid / 998 * np.linalg.inv(X.T @ X)[0, 0])
    assert g == 200 and se[0] > 1.8 * naive


def test_benjamini_hochberg():
    import pathology_brain_age as pb
    q = pb.bh([0.01, 0.04, 0.03, np.nan, 0.5])
    np.testing.assert_allclose(q[[0, 1, 2, 4]], [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])
    assert np.isnan(q[3])


def test_small_cells_are_suppressed_and_reference_rows_have_no_gap():
    import pathology_brain_age as pb
    d = _cohort(np.random.default_rng(0), n=600)
    d["rare"] = 0.0
    d.loc[d.index[:12], "rare"] = 1.0                   # 12 recordings of 6 patients
    t = pb.table(d, ["g", "rare"], min_n=30, min_subjects=10)
    rare = t[(t["kind"] == "pathology") & (t["group"] == "rare")].iloc[0]
    assert rare["suppressed"] and np.isnan(rare["n"]) and np.isnan(rare.get("mae", np.nan))
    ref = t[(t["kind"] == "report") & (t["group"] == "reference")].iloc[0]
    assert ref["n"] == int(d["reference"].sum()) and np.isnan(ref["gap_adj"])
    assert np.isfinite(t[(t["kind"] == "pathology") & (t["group"] == "g")].iloc[0]["gap_adj_q"])


def _fold_windows(tmp_path, members, rng, bad_low_attention):
    """30 recordings x 12 windows; in each recording three windows are off by 15 years and, when
    ``bad_low_attention``, get a tenth of the attention of the others."""
    age = {f"r{i}": float(rng.uniform(20, 80)) for i in range(30)}
    for m in members:
        rows = []
        for r, y in age.items():
            bad = np.arange(12) < 3
            a = np.where(bad & bad_low_attention, 0.1, 1.0) * (1 + 0.05 * rng.random(12))
            rows.append(pd.DataFrame({"recording": r, "slot": range(12), "minute": 0.0, "alpha": 3 * a / a.sum(),
                                      "window_pred": y + 15.0 * bad + rng.standard_normal(12), "age": y}))
        pd.concat(rows).sample(frac=1, random_state=0).to_parquet(tmp_path / f"{m}_test.parquet")   # any row order
    return age


def test_window_quantities_and_within_recording_correlations(tmp_path):
    import window_confidence_figures as wc
    members = ["M_f0", "M_f1"]
    age = _fold_windows(tmp_path, members, np.random.default_rng(0), bad_low_attention=True)
    d = wc.load_windows(str(tmp_path), members, set(age))
    assert len(d) == 360 and (d["n"] == 12).all()
    np.testing.assert_allclose(d.groupby("recording")["alpha"].sum(), 1.0)          # weights re-normalised
    np.testing.assert_allclose(d.groupby("recording")["att"].mean(), 1.0)           # 1 = the uniform weight
    np.testing.assert_allclose(d["err"], (d["pred"] - d["age"]).abs())
    row = wc.summary("x", d, min_windows=8)
    assert row["rho_att_err_within_median"] < -0.3 and row["rho_att_err_within_share_negative"] == 1.0
    assert row["rho_dev_err_within_median"] > 0.3                                  # the deviant windows are the wrong ones
    assert row["recording_mae_attention"] < row["recording_mae_plain_mean"]         # down-weighting them helps
    assert row["window_mae_attention_weighted"] < row["window_mae_unweighted_per_recording"]


def test_uninformative_attention_gives_no_within_recording_correlation(tmp_path):
    import window_confidence_figures as wc
    members = ["M_f0", "M_f1"]
    age = _fold_windows(tmp_path, members, np.random.default_rng(1), bad_low_attention=False)
    row = wc.summary("x", wc.load_windows(str(tmp_path), members, set(age)), min_windows=8)
    assert abs(row["rho_att_err_within_median"]) < 0.25
    assert abs(row["recording_mae_attention"] - row["recording_mae_plain_mean"]) < 0.2


def test_chbp_fix_rows_come_from_the_result_files():
    import chbp_variant_bars as cv
    t = cv.collect().set_index("label")
    assert len(t) == 7 and t["chbp_mae"].between(5, 11).all()
    assert t["chbp_mae"].idxmin() == "band + amplitude normalisation"             # the device-robust model
    assert (t.loc[t["kind"] != "train", "in_domain_mae"] == t.loc["E2L, as recorded", "in_domain_mae"]).all()
    assert (t.loc[t["kind"] == "train", "in_domain_mae"] > t.loc["E2L, as recorded", "in_domain_mae"]).all()
