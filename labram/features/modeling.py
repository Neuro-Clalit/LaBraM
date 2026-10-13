# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Model zoo, cross-validated evaluation and group-level feature importance for handcrafted EEG features.
# ---------------------------------------------------------
"""Shared modelling utilities for the handcrafted-feature notebooks.

* :func:`load_feature_table` merges the ``tuab.py`` caches into one dict of
  recording-level feature arrays plus a metadata frame.
* :class:`FBTangentSpace` maps filter-bank covariances to Riemannian tangent
  vectors inside a pipeline (fitted per training fold).
* :func:`make_model` builds linear / HistGradientBoosting / XGBoost /
  random-forest models for the two tasks (``"abn"`` classification,
  ``"age"`` regression).
* :func:`cross_validate` runs grouped CV + a held-out test and, optionally,
  **group permutation importance** (all columns of a feature group permuted
  jointly on each validation fold) and the model's **intrinsic** importance
  (|standardised coefficient|, XGBoost gain, RF impurity) summed per group.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import null_space
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import (balanced_accuracy_score, mean_absolute_error, r2_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from labram.features import extractors as fx
from labram.features import tuab
from labram.features.riemann import mean_riemann, tangent_space

MODEL_KINDS = ("linear", "hgb", "xgb", "rf")
MODEL_LABELS = {"linear": "Linear (logistic / ridge)", "hgb": "HistGradientBoosting",
                "xgb": "XGBoost", "rf": "Random forest", "lasso": "Lasso / L1-logistic",
                "enet": "Elastic net"}


# ------------------------------------------------------------------ data
def load_feature_table(npy_root: str, kinds=("main", "scattering", "spectrogram")):
    """Merge the feature caches; returns ``(D, names, meta)``.

    ``D`` maps ``rec/<family>`` to ``[N, C, F]`` arrays (plus ``fbcov`` and the
    raw quality arrays), ``names`` maps family -> per-channel feature names,
    ``meta`` is one row per recording (stem, subject, split, label, age, sex).
    """
    D, stems = {}, None
    for kind in kinds:
        z = np.load(tuab.default_cache_path(npy_root, kind))
        if stems is None:
            stems = z["meta/stem"]
            D.update({k: z[k] for k in z.files})
        else:
            assert (z["meta/stem"] == stems).all(), f"{kind} cache is not aligned"
            D.update({k: z[k] for k in z.files if k.startswith("rec/")})
    names = tuab.feature_names()
    if "scattering" in kinds:
        names.update(tuab.scattering_names())
    if "spectrogram" in kinds:
        names.update(tuab.spectrogram_names())
    meta = pd.DataFrame({k[len("meta/"):]: D[k] for k in D
                         if k.startswith("meta/") and np.ndim(D[k]) == 1})
    meta["abnormal"] = (meta.label == "abnormal").astype(int)
    return D, names, meta


def nan_to_median(X: np.ndarray) -> np.ndarray:
    """Replace non-finite entries with the column median."""
    X = np.array(X, dtype=np.float64)
    bad = ~np.isfinite(X)
    if bad.any():
        med = np.nanmedian(np.where(bad, np.nan, X), axis=0)
        X[bad] = np.take(np.nan_to_num(med), np.where(bad)[1])
    return X


@dataclass
class FeatureSet:
    """A design matrix with named column groups (and an optional Riemann block)."""

    X: np.ndarray
    groups: dict  # group name -> np.ndarray of column indices into X
    n_flat: int   # columns before the raw-covariance block (Riemann), if any
    has_riemann: bool = False
    feature_names: list = field(default_factory=list)


def build_feature_set(D: dict, names: dict, families, riemann: bool = False,
                      channels=tuab.CHANNELS_1020) -> FeatureSet:
    """Concatenate recording-level families (and, optionally, the covariance block)."""
    n = D["meta/stem"].shape[0]
    blocks, groups, fnames, start = [], {}, [], 0
    for fam in families:
        a = D[f"rec/{fam}"].reshape(n, -1)
        blocks.append(a)
        groups[fam] = np.arange(start, start + a.shape[1])
        fnames += [f"{fam}:{f}@{c}" for c in channels for f in names[fam]]
        start += a.shape[1]
    X = nan_to_median(np.concatenate(blocks, axis=1)) if blocks else np.zeros((n, 0))
    n_flat = X.shape[1]
    if riemann:
        cov = D["fbcov"].reshape(n, -1).astype(np.float64)
        groups["riemann"] = np.arange(n_flat, n_flat + cov.shape[1])
        X = np.concatenate([X, cov], axis=1)
    return FeatureSet(X, groups, n_flat, riemann, fnames)


# ------------------------------------------------------------------ Riemann transformer
_REF_CACHE: dict | None = None


def cache_riemann_refs(enabled: bool = True) -> None:
    """Memoise :class:`FBTangentSpace` reference means across fits.

    The Riemannian mean of a training fold depends only on the covariance
    block of its rows, so subset searches that refit the same folds with
    different flat feature groups can reuse it (it is ~80 % of a ridge fit).
    Keyed by a hash of that block; results are identical with or without it.
    """
    global _REF_CACHE
    _REF_CACHE = {} if enabled else None


class FBTangentSpace(BaseEstimator, TransformerMixin):
    """Filter-bank covariances (last ``n_bands*C*C`` columns) -> tangent vectors.

    The first ``n_passthrough`` columns pass through unchanged. Covariances of a
    common-average-referenced montage have rank ``C-1``; they are projected onto
    an orthonormal basis of the complement of the all-ones vector, shrunk
    slightly, and mapped at the Riemannian mean of the training data.
    """

    def __init__(self, n_passthrough: int = 0, n_channels: int = 19, n_bands: int = None,
                 shrink: float = 1e-3):
        self.n_passthrough, self.n_channels, self.n_bands, self.shrink = (
            n_passthrough, n_channels, n_bands, shrink)

    def _covs(self, X):
        c = self.n_channels
        nb = self.n_bands or len(fx.FILTERBANK)
        cov = X[:, self.n_passthrough:].reshape(len(X), nb, c, c)
        u = null_space(np.ones((1, c)))
        cov = u.T @ cov @ u
        tr = np.trace(cov, axis1=-2, axis2=-1)[..., None, None] / cov.shape[-1]
        return (1 - self.shrink) * cov + self.shrink * tr * np.eye(cov.shape[-1])

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        key = None
        if _REF_CACHE is not None:
            block = np.ascontiguousarray(X[:, self.n_passthrough:])
            key = (hashlib.sha1(block.tobytes()).hexdigest(), self.n_channels, self.n_bands, self.shrink)
            if key in _REF_CACHE:
                self.refs_ = _REF_CACHE[key]
                return self
        cov = self._covs(X)
        self.refs_ = [mean_riemann(cov[:, b]) for b in range(cov.shape[1])]
        if key is not None:
            _REF_CACHE[key] = self.refs_
        return self

    def transform(self, X):
        cov = self._covs(np.asarray(X, dtype=np.float64))
        ts = [tangent_space(cov[:, b], self.refs_[b]) for b in range(cov.shape[1])]
        return np.concatenate([np.asarray(X)[:, : self.n_passthrough]] + ts, axis=1)

    def output_names(self, flat_names: list) -> list:
        """Column names after the transform: flat names + ``riemann:<band>:<i>-<j>``."""
        nb = self.n_bands or len(fx.FILTERBANK)
        m = self.n_channels - 1
        iu = np.triu_indices(m)
        bands = list(fx.FILTERBANK)[:nb]
        return list(flat_names) + [f"riemann:{bands[b]}:u{i}-u{j}" for b in range(nb)
                                   for i, j in zip(*iu)]

    def output_groups(self, groups: dict) -> dict:
        """Column groups after the transform (Riemann block -> its tangent columns)."""
        nb = self.n_bands or len(fx.FILTERBANK)
        m = self.n_channels - 1
        out = {g: idx for g, idx in groups.items() if g != "riemann"}
        if "riemann" in groups:
            out["riemann"] = np.arange(self.n_passthrough, self.n_passthrough + nb * m * (m + 1) // 2)
        return out


# ------------------------------------------------------------------ models
SCALERS = ("none", "standard", "robust", "minmax", "power", "quantile")


def make_scaler(name: str):
    """Feature scaling for linear models (trees are invariant to monotone per-feature maps).

    ``standard``: z-score; ``robust``: (x - median) / IQR; ``minmax``: to [0, 1];
    ``power``: Yeo-Johnson then z-score (Gaussianises skewed features);
    ``quantile``: rank -> standard normal (outlier-proof); ``none``: raw values.
    """
    from sklearn.preprocessing import (MinMaxScaler, PowerTransformer, QuantileTransformer,
                                       RobustScaler)

    return {"none": None, "standard": StandardScaler(), "robust": RobustScaler(),
            "minmax": MinMaxScaler(), "power": PowerTransformer(method="yeo-johnson"),
            "quantile": QuantileTransformer(n_quantiles=500, output_distribution="normal",
                                            subsample=10_000, random_state=0)}[name]


def make_model(task: str, kind: str, n_jobs: int = 6, random_state: int = 0, params: dict = None):
    """Model factory. ``task`` is ``"abn"`` (binary) or ``"age"`` (regression).

    ``params`` overrides the defaults below. For ``linear`` it can fix the
    regularisation instead of choosing it by inner CV (``{"C": ...}`` for the
    logistic model, ``{"alpha": ...}`` for ridge) and pick the feature scaling
    (``{"scaler": name}``, see :func:`make_scaler`).
    """
    params = dict(params or {})
    if kind == "linear":
        from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV

        scaler = make_scaler(params.pop("scaler", "standard"))
        if task == "abn" and "C" in params:
            est = LogisticRegression(C=params["C"], max_iter=5000)
        elif task == "age" and "alpha" in params:
            est = Ridge(alpha=params["alpha"])
        elif task == "abn":
            est = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=3, scoring="roc_auc",
                                       max_iter=5000, n_jobs=n_jobs)
        else:
            est = RidgeCV(alphas=np.logspace(-1, 5, 25))
        return make_pipeline(scaler, est) if scaler is not None else make_pipeline(est)
    if kind in ("lasso", "enet"):
        # Sparse linear models. Age: LassoCV / ElasticNetCV (alpha, and the L1 ratio for
        # the elastic net, chosen by an inner 3-fold CV). Pathology: L1-penalised logistic
        # regression (liblinear), C chosen by inner CV on AUROC; elastic net uses saga.
        from sklearn.linear_model import ElasticNetCV, LassoCV, LogisticRegressionCV

        scaler = make_scaler(params.pop("scaler", "standard"))
        if task == "age":
            est = (LassoCV(n_alphas=30, cv=3, max_iter=5000, n_jobs=n_jobs, random_state=random_state)
                   if kind == "lasso" else
                   ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=30, cv=3, max_iter=5000, n_jobs=n_jobs,
                                random_state=random_state))
        elif kind == "lasso":
            est = LogisticRegressionCV(Cs=np.logspace(-3, 0, 7), cv=3, penalty="l1", solver="liblinear",
                                       scoring="roc_auc", max_iter=2000, n_jobs=n_jobs)
        else:
            est = LogisticRegressionCV(Cs=np.logspace(-3, 0, 5), cv=3, penalty="elasticnet", solver="saga",
                                       l1_ratios=[0.5], scoring="roc_auc", max_iter=2000, n_jobs=n_jobs)
        return make_pipeline(scaler, est)
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        p = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                 l2_regularization=1.0, early_stopping=False, random_state=random_state)
        if task == "abn":
            p.update(params)
            return HistGradientBoostingClassifier(**p)
        p["loss"] = "absolute_error"
        p.update(params)
        return HistGradientBoostingRegressor(**p)
    if kind == "xgb":
        import xgboost as xgb

        p = dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
                 colsample_bytree=0.3, min_child_weight=2, reg_lambda=1.0, tree_method="hist",
                 n_jobs=n_jobs, random_state=random_state, importance_type="gain")
        if task == "abn":
            p["eval_metric"] = "logloss"
            p.update(params)
            return xgb.XGBClassifier(**p)
        p["objective"] = "reg:squarederror"
        p.update(params)
        return xgb.XGBRegressor(**p)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        p = dict(n_estimators=400, max_features="sqrt", min_samples_leaf=2, n_jobs=n_jobs,
                 random_state=random_state)
        p.update(params)
        return RandomForestClassifier(**p) if task == "abn" else RandomForestRegressor(**p)
    if kind == "stack":
        return ResidualBoostRegressor(make_model("age", "linear", n_jobs),
                                      make_model("age", "xgb", n_jobs, params=params))
    raise ValueError(f"unknown model kind {kind!r}")


def predict(model, X, task: str) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if task == "abn" else model.predict(X)


def scores(task: str, y, p) -> dict:
    """Metrics: AUROC/BAcc/Acc/Sens/Spec (threshold 0.5) or MAE/R2/r/bias."""
    y, p = np.asarray(y), np.asarray(p)
    if task == "abn":
        yhat = p >= 0.5
        return {"AUROC": roc_auc_score(y, p), "BAcc": balanced_accuracy_score(y, yhat),
                "Acc": np.mean(yhat == (y == 1)), "Sens": np.mean(yhat[y == 1]),
                "Spec": np.mean(~yhat[y == 0])}
    r = stats.pearsonr(y, p)[0] if np.std(p) > 0 and np.std(y) > 0 else np.nan
    return {"MAE": mean_absolute_error(y, p), "R2": r2_score(y, p), "r": r, "bias": np.mean(p - y)}


def primary(task: str) -> str:
    return "AUROC" if task == "abn" else "MAE"


# ------------------------------------------------------------------ importance
def _final_estimator(model):
    while isinstance(model, Pipeline):
        model = model.steps[-1][1]
    return model


def intrinsic_group_importance(model, groups: dict) -> dict | None:
    """Share of the model's own importance per group (sums to 1).

    Linear: |coefficient| on standardised inputs; XGBoost: total gain; RF:
    mean impurity decrease. ``None`` for models without one (HistGB).
    ``groups`` must index the columns the final estimator sees.
    """
    est = _final_estimator(model)
    if hasattr(est, "coef_"):
        w = np.abs(np.ravel(est.coef_))
    elif hasattr(est, "feature_importances_"):
        w = np.asarray(est.feature_importances_, dtype=float)
    else:
        return None
    w = np.nan_to_num(w)
    tot = w.sum() or 1.0
    return {g: float(w[idx].sum() / tot) for g, idx in groups.items()}


def group_permutation_importance(model, X, y, task: str, groups: dict, n_repeats: int = 3,
                                 rng=None) -> dict:
    """Drop in the primary score when all columns of a group are permuted jointly.

    Rows of the group's block are shuffled with one permutation (keeping the
    within-group structure but breaking its link to the target). Returned as
    ``base - permuted`` for AUROC and ``permuted - base`` for MAE, so positive
    always means "the model relies on this group".
    """
    rng = rng or np.random.default_rng(0)
    key = primary(task)
    base = scores(task, y, predict(model, X, task))[key]
    out = {}
    for g, idx in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, idx] = X[rng.permutation(len(X))][:, idx]
            s = scores(task, y, predict(model, Xp, task))[key]
            drops.append(base - s if task == "abn" else s - base)
        out[g] = float(np.mean(drops))
    return out


# ------------------------------------------------------------------ evaluation
@dataclass
class Splits:
    dev: dict
    test: dict
    folds: dict
    test_age_abn: np.ndarray
    y: dict
    subjects: np.ndarray = None


def make_splits(meta: pd.DataFrame, seed: int = 0) -> Splits:
    """Development pool = TUAB train (+val); test = official eval; subject-grouped folds."""
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    pool = meta.source_split.isin(["train", "val"]).to_numpy()
    test = meta.source_split.eq("test").to_numpy()
    normal = meta.label.eq("normal").to_numpy()
    age = meta.age.to_numpy(dtype=float)
    ok = np.isfinite(age)
    subj = meta.subject.to_numpy()
    y = {"abn": meta.abnormal.to_numpy().astype(float), "age": age}
    dev = {"abn": np.where(pool)[0], "age": np.where(pool & normal & ok)[0]}
    tst = {"abn": np.where(test)[0], "age": np.where(test & normal & ok)[0]}
    folds = {
        "abn": list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
            dev["abn"], y["abn"][dev["abn"]], subj[dev["abn"]])),
        "age": list(GroupKFold(5, shuffle=True, random_state=seed).split(
            dev["age"], groups=subj[dev["age"]])),
    }
    return Splits(dev, tst, folds, np.where(test & ~normal & ok)[0], y, subj)


def selected_per_group(model, groups: dict, tol: float = 1e-10) -> dict | None:
    """Number of non-zero coefficients per group (for sparse linear models)."""
    est = _final_estimator(model)
    if not hasattr(est, "coef_"):
        return None
    w = np.abs(np.ravel(est.coef_))
    return {g: int((w[idx] > tol).sum()) for g, idx in groups.items()}


def top_features(model, names: list, k: int = 20) -> list | None:
    """Top-``k`` individual inputs by the model's intrinsic importance (share of total)."""
    est = _final_estimator(model)
    if hasattr(est, "coef_"):
        w = np.abs(np.ravel(est.coef_))
    elif hasattr(est, "feature_importances_"):
        w = np.asarray(est.feature_importances_, dtype=float)
    else:
        return None
    w = np.nan_to_num(w)
    order = np.argsort(-w)[:k]
    tot = w.sum() or 1.0
    return [(names[i], float(w[i] / tot)) for i in order]


def cross_validate(fs: FeatureSet, task: str, kind: str, splits: Splits, n_jobs: int = 6,
                   importance: bool = False, n_repeats: int = 3, run_test: bool = True,
                   params: dict = None, train_fraction: float = None, return_train: bool = False,
                   seed: int = 0) -> dict:
    """Grouped 5-fold CV on the development pool, then refit -> held-out test.

    Returns a dict with ``cv_<metric>`` (mean), ``cv_<metric>_sd``,
    ``test_<metric>``, ``test_abn_<metric>`` (age only: the normal-trained model
    on abnormal test recordings), ``oof`` and ``test_pred`` arrays, and, with
    ``importance=True``, ``perm_importance`` / ``intrinsic_importance`` (dicts
    averaged over folds).
    """
    X, y = fs.X, splits.y[task]
    idx = splits.dev[task]

    def pipe():
        m = make_model(task, kind, n_jobs=n_jobs, params=params)
        if fs.has_riemann:
            return make_pipeline(FBTangentSpace(n_passthrough=fs.n_flat), m)
        return m

    out_groups = (FBTangentSpace(n_passthrough=fs.n_flat).output_groups(fs.groups)
                  if fs.has_riemann else fs.groups)
    oof = np.full(len(idx), np.nan)
    per_fold, perm, intr = [], [], []
    rng = np.random.default_rng(0)
    sub_rng = np.random.default_rng(seed)
    train_scores = []
    for tr, va in splits.folds[task]:
        if train_fraction is not None and train_fraction < 1:
            tr = np.sort(sub_rng.choice(tr, int(round(train_fraction * len(tr))), replace=False))
        m = pipe().fit(X[idx[tr]], y[idx[tr]])
        oof[va] = predict(m, X[idx[va]], task)
        per_fold.append(scores(task, y[idx[va]], oof[va]))
        if return_train:
            train_scores.append(scores(task, y[idx[tr]], predict(m, X[idx[tr]], task)))
        if importance:
            perm.append(group_permutation_importance(m, X[idx[va]], y[idx[va]], task, fs.groups,
                                                     n_repeats, rng))
            ii = intrinsic_group_importance(m, out_groups)
            if ii is not None:
                intr.append(ii)
    res = {"oof": oof}
    for k in (train_scores[0] if train_scores else {}):
        v = [s_[k] for s_ in train_scores]
        res[f"train_{k}"], res[f"train_{k}_sd"] = float(np.mean(v)), float(np.std(v))
    for k in per_fold[0]:
        v = [s[k] for s in per_fold]
        res[f"cv_{k}"], res[f"cv_{k}_sd"] = float(np.mean(v)), float(np.std(v))
    if importance:
        res["perm_importance"] = {g: (float(np.mean([p[g] for p in perm])),
                                      float(np.std([p[g] for p in perm]))) for g in fs.groups}
        res["intrinsic_importance"] = ({g: float(np.mean([p[g] for p in intr])) for g in fs.groups}
                                       if intr else None)
    if run_test:
        m = pipe().fit(X[idx], y[idx])
        tp = predict(m, X[splits.test[task]], task)
        res["test_pred"] = tp
        if importance and fs.feature_names:
            names = (FBTangentSpace(n_passthrough=fs.n_flat).output_names(fs.feature_names)
                     if fs.has_riemann else fs.feature_names)
            res["top_features"] = top_features(m, names)
        if kind in ("lasso", "enet"):
            res["n_selected"] = selected_per_group(m, out_groups)
        res.update({f"test_{k}": v for k, v in scores(task, y[splits.test[task]], tp).items()})
        if task == "age":
            pa = predict(m, X[splits.test_age_abn], task)
            res.update({f"test_abn_{k}": v for k, v in scores("age", y[splits.test_age_abn], pa).items()})
    return res


def results_cache(path: str):
    """A tiny pickle-backed dict for expensive notebook results (keyed by tuples)."""
    import pickle

    store = {}
    if path and os.path.exists(path):
        with open(path, "rb") as fh:
            store = pickle.load(fh)

    def save():
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(store, fh)
            os.replace(tmp, path)

    return store, save


def forward_group_selection(evaluate, groups, metric: str = "cv_MAE", lower_is_better: bool = True,
                            start=()) -> pd.DataFrame:
    """Greedy forward selection over feature *groups*.

    ``evaluate(list_of_groups) -> dict`` scores one subset (e.g. a cached
    :func:`cross_validate` call). Starting from ``start``, each step adds the
    group that gives the best ``metric``; the search runs until every group is
    in, so the returned path (one row per step: the group added, the subset and
    all numeric scores) also shows where adding more starts to hurt. Choose on
    CV scores only; held-out test scores are just carried along.
    """
    chosen, remaining = list(start), [g for g in groups if g not in start]
    sign = 1.0 if lower_is_better else -1.0
    rows = []
    while remaining:
        trial = {g: evaluate(chosen + [g]) for g in remaining}
        best = min(trial, key=lambda g: sign * trial[g][metric])
        chosen.append(best)
        remaining.remove(best)
        rows.append({"step": len(chosen), "added": best, "subset": tuple(chosen),
                     **{k: v for k, v in trial[best].items() if isinstance(v, (int, float, np.floating))}})
    return pd.DataFrame(rows)


def one_se_choice(path: pd.DataFrame, metric: str = "cv_MAE", n_folds: int = 5,
                  lower_is_better: bool = True) -> int:
    """Smallest step whose CV score is within one standard error of the best step."""
    sign = 1.0 if lower_is_better else -1.0
    best = (sign * path[metric]).idxmin()
    se = path[f"{metric}_sd"][best] / np.sqrt(n_folds)
    ok = sign * path[metric] <= sign * path[metric][best] + se
    return int(path.step[ok].min())


# ------------------------------------------------------------------ stacking + nested tuning
from sklearn.base import RegressorMixin  # noqa: E402


class ResidualBoostRegressor(BaseEstimator, RegressorMixin):
    """Ridge first, then a booster on ridge's cross-fitted residuals.

    The booster is trained on ``y - oof_base`` where ``oof_base`` are
    out-of-fold base predictions (inner K-fold), so it learns the base model's
    *generalisation* errors rather than its (smaller) in-sample residuals. The
    prediction is ``base(x) + booster(x)``: the linear trend (which extrapolates
    to the age extremes) plus non-linear corrections.
    """

    def __init__(self, base=None, booster=None, n_inner: int = 5, random_state: int = 0):
        self.base, self.booster, self.n_inner, self.random_state = base, booster, n_inner, random_state

    def fit(self, X, y):
        from sklearn.model_selection import KFold, cross_val_predict

        y = np.asarray(y, dtype=float)
        cv = KFold(self.n_inner, shuffle=True, random_state=self.random_state)
        oof = cross_val_predict(clone(self.base), X, y, cv=cv)
        self.base_ = clone(self.base).fit(X, y)
        self.booster_ = clone(self.booster).fit(X, y - oof)
        return self

    def predict(self, X):
        return self.base_.predict(X) + self.booster_.predict(X)


def nested_tune(fs: FeatureSet, task: str, kind: str, splits: Splits, space: dict, n_iter: int = 12,
                n_inner: int = 3, n_jobs: int = 6, seed: int = 0, run_test: bool = True) -> dict:
    """Nested CV: random search on each outer training fold, scored on its validation fold.

    The inner search samples ``n_iter`` configurations from ``space`` (a dict of
    lists), scores each by a subject-grouped ``n_inner``-fold CV *inside the
    outer training fold*, refits the best on the whole training fold and
    evaluates it on the untouched validation fold. The final model is tuned the
    same way on the whole development pool and scored once on the held-out test.
    """
    from sklearn.model_selection import GroupKFold, ParameterSampler

    X, y = fs.X, splits.y[task]
    idx = splits.dev[task]
    groups = splits.subjects
    configs = list(ParameterSampler(space, n_iter=n_iter, random_state=seed))
    key = primary(task)
    sign = 1 if task == "abn" else -1  # maximise AUROC, minimise MAE

    def pipe(p):
        m = make_model(task, kind, n_jobs=n_jobs, params=p)
        return make_pipeline(FBTangentSpace(n_passthrough=fs.n_flat), m) if fs.has_riemann else m

    def search(rows):
        inner = list(GroupKFold(n_inner, shuffle=True, random_state=seed).split(rows, groups=groups[rows]))
        best, best_s = None, -np.inf
        for p in configs:
            s_ = [scores(task, y[rows[v]], predict(pipe(p).fit(X[rows[t]], y[rows[t]]), X[rows[v]], task))[key]
                  for t, v in inner]
            if sign * np.mean(s_) > best_s:
                best, best_s = p, sign * np.mean(s_)
        return best

    oof, per_fold, chosen = np.full(len(idx), np.nan), [], []
    for tr, va in splits.folds[task]:
        p = search(idx[tr])
        chosen.append(p)
        m = pipe(p).fit(X[idx[tr]], y[idx[tr]])
        oof[va] = predict(m, X[idx[va]], task)
        per_fold.append(scores(task, y[idx[va]], oof[va]))
    res = {"oof": oof, "chosen_params": chosen}
    for k in per_fold[0]:
        v = [s_[k] for s_ in per_fold]
        res[f"cv_{k}"], res[f"cv_{k}_sd"] = float(np.mean(v)), float(np.std(v))
    if run_test:
        p = search(idx)
        res["final_params"] = p
        m = pipe(p).fit(X[idx], y[idx])
        tp = predict(m, X[splits.test[task]], task)
        res["test_pred"] = tp
        res.update({f"test_{k}": v for k, v in scores(task, y[splits.test[task]], tp).items()})
        if task == "age":
            pa = predict(m, X[splits.test_age_abn], task)
            res.update({f"test_abn_{k}": v for k, v in scores("age", y[splits.test_age_abn], pa).items()})
    return res


# ------------------------------------------------------------------ age as (ordinal) classification
AGE_CLASS_METHODS = ("ridge_binned", "logistic", "ordinal", "xgb", "rf")


def age_grid(width: float, lo: float = 0.0, hi: float = 100.0):
    """Bin edges ``[lo, lo+w, ...]`` and centres for age classes of ``width`` years."""
    edges = np.arange(lo, hi + width, width)
    return edges, (edges[:-1] + edges[1:]) / 2


def age_to_class(age, width: float, lo: float = 0.0) -> np.ndarray:
    return np.floor((np.asarray(age, dtype=float) - lo) / width).astype(int)


def quadratic_weighted_kappa(a, b, n_classes: int) -> float:
    """Cohen's kappa with weights ``(i-j)^2/(K-1)^2`` (ordinal agreement; 1 = perfect, 0 = chance)."""
    a, b = np.asarray(a), np.asarray(b)
    o = np.zeros((n_classes, n_classes))
    np.add.at(o, (a, b), 1)
    w = (np.subtract.outer(np.arange(n_classes), np.arange(n_classes)) ** 2) / max(n_classes - 1, 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / max(o.sum(), 1)
    den = (w * e).sum()
    return float(1 - (w * o).sum() / den) if den > 0 else 0.0


def age_class_scores(age, proba, width: float, point=None) -> dict:
    """Metrics of a class-probability matrix ``proba`` [n, K] over the global age grid.

    ``point`` (optional) overrides the point estimate (used for the binned
    regression reference); otherwise the expected age ``sum_k p_k c_k``.
    """
    edges, centres = age_grid(width)
    k = len(centres)
    yc = np.clip(age_to_class(age, width), 0, k - 1)
    exp_age = proba @ centres if point is None else np.asarray(point, dtype=float)
    pc = (np.clip(age_to_class(exp_age, width), 0, k - 1) if point is not None
          else proba.argmax(axis=1))
    err = np.abs(exp_age - age)
    return {"acc": float(np.mean(pc == yc)), "acc_pm1": float(np.mean(np.abs(pc - yc) <= 1)),
            "qwk": quadratic_weighted_kappa(yc, pc, k), "within5": float(np.mean(err <= 5)),
            "MAE": float(err.mean()), "MAE_argmax": float(np.mean(np.abs(centres[pc] - age)))}


def _ordinal_proba(Xtr, ytr_cls, Xte, k: int, C: float):
    """Frank & Hall (2001): one logistic model per threshold, P(class > j) -> class probabilities."""
    from sklearn.linear_model import LogisticRegression

    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    gt = np.ones((len(Xte), k - 1))
    for j in range(k - 1):
        lab = (ytr_cls > j).astype(int)
        if lab.min() == lab.max():  # threshold outside the training range
            gt[:, j] = float(lab[0])
        else:
            gt[:, j] = LogisticRegression(C=C, max_iter=2000).fit(Ztr, lab).predict_proba(Zte)[:, 1]
    gt = np.minimum.accumulate(gt, axis=1)  # enforce P(>j) non-increasing in j
    cum = np.concatenate([np.ones((len(Xte), 1)), gt, np.zeros((len(Xte), 1))], axis=1)
    return np.clip(cum[:, :-1] - cum[:, 1:], 0, None)


def _fit_proba(method: str, Xtr, age_tr, Xte, width: float, n_jobs: int, C: float = None,
               groups_tr=None, seed: int = 0):
    """Class probabilities over the global grid for one train/test split."""
    from sklearn.linear_model import LogisticRegression

    import warnings

    warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")
    _, centres = age_grid(width)
    k = len(centres)
    ytr = np.clip(age_to_class(age_tr, width), 0, k - 1)
    full = np.zeros((len(Xte), k))
    if method == "ordinal":
        return _ordinal_proba(Xtr, ytr, Xte, k, C)
    present = np.unique(ytr)
    enc = np.searchsorted(present, ytr)
    if method == "logistic":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Xtr, enc)
    elif method == "rf":
        from sklearn.ensemble import RandomForestClassifier

        m = RandomForestClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=2,
                                   n_jobs=n_jobs, random_state=seed).fit(Xtr, enc)
    elif method == "xgb":
        import xgboost as xgb

        m = xgb.XGBClassifier(objective="multi:softprob", n_estimators=250, learning_rate=0.1, max_depth=3,
                              subsample=0.8, colsample_bytree=0.3, min_child_weight=2, tree_method="hist",
                              n_jobs=n_jobs, random_state=seed).fit(Xtr, enc)
    else:
        raise ValueError(method)
    full[:, present] = m.predict_proba(Xte)
    return full


def _select_C(Xtr, age_tr, groups_tr, width: float, Cs=(1e-3, 1e-2, 1e-1), n_inner: int = 3, seed: int = 0):
    """Inner subject-grouped CV: pick the multinomial-logistic C with the lowest expected-age MAE."""
    from sklearn.model_selection import GroupKFold

    inner = list(GroupKFold(n_inner, shuffle=True, random_state=seed).split(Xtr, groups=groups_tr))
    best, best_mae = None, np.inf
    for C in Cs:
        maes = []
        for t, v in inner:
            p = _fit_proba("logistic", Xtr[t], age_tr[t], Xtr[v], width, 1, C=C)
            maes.append(age_class_scores(age_tr[v], p, width)["MAE"])
        if np.mean(maes) < best_mae:
            best, best_mae = C, np.mean(maes)
    return best


def age_classification_cv(fs: FeatureSet, method: str, width: float, splits: Splits, n_jobs: int = 6,
                          ridge_result: dict = None, seed: int = 0) -> dict:
    """Age as ``width``-year classes on normal recordings: grouped CV + held-out test.

    Uses the same development pool and folds as the age regression. ``logistic``
    and ``ordinal`` choose ``C`` by an inner grouped CV on expected-age MAE
    (``ordinal`` reuses the multinomial choice). ``ridge_binned`` re-scores a
    finished ridge regression (``ridge_result`` from :func:`cross_validate`) with
    the classification metrics. Returns pooled CV metrics (``cv_*``), test
    metrics (``test_*``) and the out-of-fold / test probability matrices.
    """
    if fs.has_riemann:
        raise ValueError("age classification expects a flat feature set")
    X, age = fs.X, splits.y["age"]
    idx, tst = splits.dev["age"], splits.test["age"]
    _, centres = age_grid(width)
    k = len(centres)
    res = {"width": width, "method": method}
    if method == "ridge_binned":
        oof, tp = ridge_result["oof"], ridge_result["test_pred"]
        cv = age_class_scores(age[idx], None, width, point=oof)
        te = age_class_scores(age[tst], None, width, point=tp)
        res.update({f"cv_{a}": b for a, b in cv.items()} | {f"test_{a}": b for a, b in te.items()})
        return res
    oof = np.zeros((len(idx), k))
    chosen = []
    for tr, va in splits.folds["age"]:
        C = None
        if method in ("logistic", "ordinal"):
            C = _select_C(X[idx[tr]], age[idx[tr]], splits.subjects[idx[tr]], width, seed=seed)
            chosen.append(C)
        oof[va] = _fit_proba(method, X[idx[tr]], age[idx[tr]], X[idx[va]], width, n_jobs, C=C)
    C = _select_C(X[idx], age[idx], splits.subjects[idx], width, seed=seed) if method in ("logistic", "ordinal") else None
    tp = _fit_proba(method, X[idx], age[idx], X[tst], width, n_jobs, C=C)
    cv = age_class_scores(age[idx], oof, width)
    te = age_class_scores(age[tst], tp, width)
    res.update({f"cv_{a}": b for a, b in cv.items()} | {f"test_{a}": b for a, b in te.items()})
    res.update({"oof_proba": oof.astype(np.float32), "test_proba": tp.astype(np.float32),
                "chosen_C": chosen, "final_C": C})
    return res
