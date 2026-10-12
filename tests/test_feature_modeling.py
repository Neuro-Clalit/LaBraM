"""labram.features.modeling: model zoo, CV and group importance on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from labram.features import modeling as M


def _synthetic(n=240, seed=0):
    rng = np.random.default_rng(seed)
    informative = rng.standard_normal((n, 4))
    noise = rng.standard_normal((n, 6))
    X = np.concatenate([informative, noise], axis=1)
    groups = {"signal": np.arange(4), "noise": np.arange(4, 10)}
    y_age = 40 + 8 * informative.sum(axis=1) + rng.standard_normal(n)
    y_abn = (informative[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(float)
    meta = pd.DataFrame({
        "stem": [f"r{i}" for i in range(n)], "subject": [f"s{i // 2}" for i in range(n)],
        "source_split": ["train"] * (n - 40) + ["test"] * 40,
        "label": np.where(y_abn == 1, "abnormal", "normal"), "age": y_age,
    })
    meta["abnormal"] = y_abn.astype(int)
    return M.FeatureSet(X, groups, X.shape[1], False, [f"f{i}" for i in range(10)]), meta


@pytest.mark.parametrize("kind", ["linear", "hgb", "rf"])
def test_cross_validate_and_group_permutation_importance(kind):
    fs, meta = _synthetic()
    sp = M.make_splits(meta)
    res = M.cross_validate(fs, "abn", kind, sp, n_jobs=1, importance=True, n_repeats=2)
    assert res["cv_AUROC"] > 0.75 and "test_AUROC" in res
    perm = res["perm_importance"]
    assert perm["signal"][0] > perm["noise"][0] + 0.05
    if kind != "hgb":
        intr = res["intrinsic_importance"]
        assert intr["signal"] > intr["noise"] and abs(sum(intr.values()) - 1) < 1e-6
        assert res["top_features"][0][0] in {"f0", "f1", "f2", "f3"}


def test_age_task_reports_abnormal_stress_test():
    fs, meta = _synthetic()
    sp = M.make_splits(meta)
    res = M.cross_validate(fs, "age", "linear", sp, n_jobs=1, importance=True, n_repeats=2)
    assert res["cv_MAE"] < 3 and "test_abn_MAE" in res
    assert res["perm_importance"]["signal"][0] > 1.0  # MAE increase in years


def test_xgboost_model_if_available():
    pytest.importorskip("xgboost")
    fs, meta = _synthetic()
    sp = M.make_splits(meta)
    res = M.cross_validate(fs, "age", "xgb", sp, n_jobs=1, importance=True, n_repeats=1)
    assert res["intrinsic_importance"]["signal"] > 0.5


def test_fb_tangent_space_groups_and_names():
    rng = np.random.default_rng(0)
    c, nb, n = 5, 2, 30
    a = rng.standard_normal((n, nb, c, 3 * c))
    cov = a @ np.swapaxes(a, -1, -2)
    h = np.eye(c) - 1.0 / c  # common-average reference -> rank c-1
    cov = h @ cov @ h
    X = np.concatenate([rng.standard_normal((n, 3)), cov.reshape(n, -1)], axis=1)
    t = M.FBTangentSpace(n_passthrough=3, n_channels=c, n_bands=nb).fit(X)
    Z = t.transform(X)
    m = c - 1
    assert Z.shape == (n, 3 + nb * m * (m + 1) // 2)
    groups = t.output_groups({"flat": np.arange(3), "riemann": np.arange(3, X.shape[1])})
    assert len(groups["riemann"]) == Z.shape[1] - 3
    assert len(t.output_names(["a", "b", "c"])) == Z.shape[1]


def test_params_train_scores_and_learning_curve_fraction():
    fs, meta = _synthetic()
    sp = M.make_splits(meta)
    res = M.cross_validate(fs, "age", "rf", sp, n_jobs=1, params={"min_samples_leaf": 1},
                           return_train=True, run_test=False)
    assert res["train_MAE"] < res["cv_MAE"]  # a deep forest fits its training data closely
    small = M.cross_validate(fs, "age", "linear", sp, n_jobs=1, params={"alpha": 1.0},
                             train_fraction=0.3, run_test=False)
    assert np.isfinite(small["cv_MAE"])


def test_residual_boost_and_nested_tune():
    pytest.importorskip("xgboost")
    fs, meta = _synthetic()
    sp = M.make_splits(meta)
    stack = M.cross_validate(fs, "age", "stack", sp, n_jobs=1, params={"n_estimators": 50}, run_test=False)
    assert stack["cv_MAE"] < 5
    res = M.nested_tune(fs, "abn", "rf", sp, {"min_samples_leaf": [1, 5], "max_features": ["sqrt", 0.5]},
                        n_iter=2, n_inner=2, n_jobs=1)
    assert len(res["chosen_params"]) == 5 and "final_params" in res and res["cv_AUROC"] > 0.7


@pytest.mark.parametrize("scaler", M.SCALERS)
def test_linear_model_with_each_scaler(scaler):
    fs, meta = _synthetic()
    sp = M.make_splits(meta)
    res = M.cross_validate(fs, "age", "linear", sp, n_jobs=1, params={"scaler": scaler}, run_test=False)
    assert res["cv_MAE"] < 6


def test_quadratic_weighted_kappa_bounds():
    assert M.quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3], 4) == pytest.approx(1.0)
    assert M.quadratic_weighted_kappa([0, 1, 2, 3], [3, 2, 1, 0], 4) < 0


@pytest.mark.parametrize("method", ["logistic", "ordinal", "rf"])
def test_age_classification_cv(method):
    fs, meta = _synthetic(n=300)
    sp = M.make_splits(meta)
    res = M.age_classification_cv(fs, method, 5, sp, n_jobs=1)
    k = len(M.age_grid(5)[1])
    assert res["oof_proba"].shape == (len(sp.dev["age"]), k)
    assert np.allclose(res["oof_proba"].sum(axis=1), 1, atol=1e-5)
    assert res["cv_MAE"] < 12 and 0 <= res["cv_acc"] <= 1 and res["cv_qwk"] > 0.3


def test_age_classification_ridge_binned_reference():
    fs, meta = _synthetic(n=300)
    sp = M.make_splits(meta)
    ridge = M.cross_validate(fs, "age", "linear", sp, n_jobs=1)
    res = M.age_classification_cv(fs, "ridge_binned", 2, sp, ridge_result=ridge)
    assert res["cv_MAE"] == pytest.approx(ridge["cv_MAE"], rel=0.05)  # same point estimates (pooled vs fold-mean)


@pytest.mark.parametrize("kind", ["lasso", "enet"])
def test_sparse_linear_models_select_informative_group(kind):
    fs, meta = _synthetic(n=300)
    sp = M.make_splits(meta)
    res = M.cross_validate(fs, "age", kind, sp, n_jobs=1, importance=True, n_repeats=1)
    assert res["cv_MAE"] < 3
    assert res["n_selected"]["signal"] == 4
    # CV-optimal L1 penalties keep some noise columns, but with negligible weight.
    assert res["intrinsic_importance"]["signal"] > 0.95


def test_l1_logistic_runs():
    fs, meta = _synthetic(n=300)
    sp = M.make_splits(meta)
    res = M.cross_validate(fs, "abn", "lasso", sp, n_jobs=1)
    assert res["cv_AUROC"] > 0.8 and "n_selected" in res
