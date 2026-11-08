# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Confidence-weighted window pooling without retraining (Proposal A of
# docs/window_aggregation_plan.md) and whether the confidences track errors.
# ---------------------------------------------------------
"""Window weights, recording confidences and their relation to the errors.

Input: one row per scored window with ``split, recording, window, age, cohort,
dataset, pred`` (the plain prediction, years), the test-time-augmented
predictions ``tta_1 .. tta_k`` and the quality descriptors ``emg, emg_max,
max_abs, amp_uv, min_std`` (``scripts/window_confidence_predict.py``).

Window weights, Eq. (3.0) of the plan pools ``sum w y / sum w`` per recording:

* **A1** -- inverse TTA variance, ``w = 1 / (var_tta + eps)``, pooling the TTA
  means (:func:`add_tta_stats`, :func:`inverse_variance_weights`);
* **A2** -- the Huber M-estimate's fixed-point weights
  ``w = min(1, delta / |pred - mu|)`` (:func:`huber_weights`);
* **A3** -- a softmax over a linear score of window descriptors, fitted on val
  (:func:`fit_softmax_pooling`, :func:`softmax_weights`).

Then :func:`weight_error_correlation` asks whether a window's weight tracks its
own error (pooled over windows and, what pooling can use, *within* a
recording), :func:`recording_confidence` turns the windows of a recording into
uncertainty scores, and :func:`confidence_error_table` /
:func:`risk_coverage` ask whether those track the recording's error.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

KEY = ["split", "recording"]
DESCRIPTORS = ("log_emg", "log_emg_max", "log_max_abs", "log_amp", "log_min_std",
               "log_tta_var", "dist_median", "minute")


# ------------------------------------------------------------------ weights
def tta_columns(df: pd.DataFrame) -> List[str]:
    """``pred`` and the TTA copies ``tta_1 ..`` (every scored copy)."""
    return ["pred"] + sorted((c for c in df if c.startswith("tta_") and c[4:].isdigit()),
                             key=lambda c: int(c[4:]))


def add_tta_stats(df: pd.DataFrame) -> pd.DataFrame:
    """``tta_mean`` and ``tta_var`` (population variance) over every copy, Eq. (3.1)."""
    p = df[tta_columns(df)].to_numpy(float)
    df["tta_mean"] = p.mean(1)
    df["tta_var"] = p.var(1)
    return df


def inverse_variance_weights(var, eps: float = 1.0) -> np.ndarray:
    """A1: ``1 / (var + eps)``; ``eps`` (years^2) caps an invariant window's weight."""
    return 1.0 / (np.asarray(var, float) + eps)


def huber_weights(df: pd.DataFrame, pred: str = "pred", delta: float = 5.0,
                  n_iter: int = 10) -> Tuple[np.ndarray, pd.Series]:
    """A2: fixed-point (IRLS) weights of the Huber M-estimate of each
    recording's location, Eq. (3.2), started at the median. Returns
    ``(window weights, location per window)``."""
    y = df[pred].to_numpy(float)
    g = df.groupby(KEY, sort=False)[pred]
    mu = g.transform("median").to_numpy(float)
    codes = g.ngroup().to_numpy()
    w = np.ones_like(y)
    for _ in range(n_iter):
        w = np.minimum(1.0, delta / np.maximum(np.abs(y - mu), 1e-9))
        mu = (np.bincount(codes, w * y) / np.bincount(codes, w))[codes]
    return w, pd.Series(mu, index=df.index)


def descriptor_matrix(df: pd.DataFrame, pred: str = "pred") -> pd.DataFrame:
    """A3 inputs ``q_i``: log EMG ratio (channel mean and max), log max |x|,
    log mean / minimum channel std, log TTA variance, distance of the window's
    prediction from its recording's median, and the minute of the window
    within the evaluated stretch."""
    g = df.groupby(KEY, sort=False)
    q = pd.DataFrame(index=df.index)
    q["log_emg"] = np.log(df["emg"].clip(lower=1e-6))
    q["log_emg_max"] = np.log(df["emg_max"].clip(lower=1e-6))
    q["log_max_abs"] = np.log1p(df["max_abs"])
    q["log_amp"] = np.log1p(df["amp_uv"])
    q["log_min_std"] = np.log1p(df["min_std"].clip(upper=1e4))
    q["log_tta_var"] = np.log1p(df["tta_var"])
    q["dist_median"] = (df[pred] - g[pred].transform("median")).abs()
    q["minute"] = (df["window"] - g["window"].transform("min")) / 6.0
    return q[list(DESCRIPTORS)]


def softmax_weights(q: np.ndarray, theta: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """``exp(theta . q_i)`` normalized over each recording (``codes``), Eq. (3.3)."""
    s = q @ theta
    s = s - pd.Series(s).groupby(codes).transform("max").to_numpy()
    e = np.exp(s)
    return e / np.bincount(codes, e)[codes]


def fit_softmax_pooling(q: np.ndarray, y_win: np.ndarray, age: np.ndarray, codes: np.ndarray,
                        l2: float = 1e-3, huber_delta: float = 1.0) -> np.ndarray:
    """A3: ``theta`` minimizing the mean Huber loss (``huber_delta`` years,
    a smooth MAE) of the softmax-pooled recording prediction, plus ``l2 |theta|^2``.
    ``q`` is standardized by the caller; ``age`` is per window (constant within
    a recording)."""
    import torch
    qt = torch.as_tensor(q, dtype=torch.float64)
    yt = torch.as_tensor(y_win, dtype=torch.float64)
    ct = torch.as_tensor(codes, dtype=torch.long)
    n = int(codes.max()) + 1
    age_rec = torch.as_tensor(pd.Series(age).groupby(codes).first().to_numpy(), dtype=torch.float64)
    theta = torch.zeros(q.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([theta], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        s = qt @ theta
        smax = torch.zeros(n, dtype=torch.float64).scatter_reduce(0, ct, s.detach(), "amax",
                                                                   include_self=False)
        e = torch.exp(s - smax[ct])
        den = torch.zeros(n, dtype=torch.float64).index_add_(0, ct, e)
        num = torch.zeros(n, dtype=torch.float64).index_add_(0, ct, e * yt)
        loss = torch.nn.functional.huber_loss(num / den, age_rec, delta=huber_delta)
        loss = loss + l2 * theta.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return theta.detach().numpy()


def a3_weights(df: pd.DataFrame, fit_split: str = "val", n_folds: int = 5,
               seed: int = 0) -> Tuple[np.ndarray, pd.Series]:
    """A3 weights for every window: ``fit_split`` recordings get cross-fitted
    weights (``theta`` fitted on the other folds of ``fit_split``), every other
    split the ``theta`` fitted on all of ``fit_split``. Descriptors are
    standardized with ``fit_split`` statistics. Returns ``(weights, theta)``."""
    q = descriptor_matrix(df)
    fit = (df["split"] == fit_split).to_numpy()
    mu, sd = q[fit].mean(), q[fit].std().replace(0, 1)
    qz = ((q - mu) / sd).to_numpy(float)
    codes = df.groupby(KEY, sort=False).ngroup().to_numpy()
    y, age = df["pred"].to_numpy(float), df["age"].to_numpy(float)

    def fit_on(mask):
        sub = codes[mask]
        _, local = np.unique(sub, return_inverse=True)
        return fit_softmax_pooling(qz[mask], y[mask], age[mask], local)

    theta = fit_on(fit)
    w = softmax_weights(qz, theta, codes)
    recs = np.unique(codes[fit])
    fold = np.random.default_rng(seed).permutation(len(recs)) % n_folds
    fold_of = dict(zip(recs, fold))
    win_fold = np.array([fold_of.get(c, -1) for c in codes])
    for k in range(n_folds):
        th = fit_on(fit & (win_fold != k))
        sel = fit & (win_fold == k)
        w[sel] = softmax_weights(qz[sel], th, np.unique(codes[sel], return_inverse=True)[1])
    return w, pd.Series(theta / sd.to_numpy(), index=q.columns)   # per raw descriptor unit


# ------------------------------------------------------------------ pooling
def pool(df: pd.DataFrame, weights=None, pred: str = "pred") -> pd.DataFrame:
    """Per-recording weighted mean of ``pred`` (Eq. (3.0)), with ``err`` /
    ``abs_err`` and the recording's ``age, cohort, dataset``."""
    w = np.ones(len(df)) if weights is None else np.asarray(weights, float)
    d = df[KEY + ["age", "cohort", "dataset"]].assign(_w=w, _wp=w * df[pred].to_numpy(float))
    rec = d.groupby(KEY, sort=False).agg(age=("age", "first"), cohort=("cohort", "first"),
                                         dataset=("dataset", "first"), _wp=("_wp", "sum"),
                                         _w=("_w", "sum")).reset_index()
    rec["pred"] = rec.pop("_wp") / rec.pop("_w")
    rec["err"] = rec["pred"] - rec["age"]
    rec["abs_err"] = rec["err"].abs()
    return rec


def trimmed_pool(df: pd.DataFrame, frac: float = 0.1, pred: str = "pred") -> pd.DataFrame:
    """Per-recording mean after dropping the ``frac`` lowest and highest windows."""
    r = df.groupby(KEY, sort=False)[pred].rank(pct=True, method="first")
    n = df.groupby(KEY, sort=False)[pred].transform("size")
    keep = ((r > frac) & (r <= 1 - frac)) | (n < 3)
    return pool(df, keep.astype(float).to_numpy(), pred)


def window_weights(df: pd.DataFrame, eps: float = 1.0, delta: float = 5.0) -> pd.DataFrame:
    """Every pooling of the comparison as ``(weights, pooled column)`` columns:
    ``w_<method>`` per window. A1 pools ``tta_mean``, the others ``pred``."""
    w = pd.DataFrame(index=df.index)
    w["w_A1"] = inverse_variance_weights(df["tta_var"], eps)
    w["w_A2"], _ = huber_weights(df, "pred", delta)
    w["w_A3"], _ = a3_weights(df)
    return w


POOLED_ON = {"A1": "tta_mean", "A2": "pred", "A3": "pred"}


# ---------------------------------------------- weights vs. window errors
def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(stats.spearmanr(a, b).statistic)


def weight_error_correlation(df: pd.DataFrame, weight: str, err: str,
                             min_windows: int = 5) -> Dict[str, float]:
    """Does a window's weight track its own absolute error?

    ``rho_pooled`` -- Spearman over all windows (mixes recordings);
    ``rho_within`` -- mean over recordings of the within-recording Spearman,
    the part pooling can exploit (weights only move a recording's estimate
    among its own windows); ``frac_neg`` -- share of recordings where it is
    negative (the larger the weight, the smaller the error);
    ``err_low_w`` / ``err_high_w`` -- mean |error| of the windows in the
    lowest / highest within-recording weight quintile, minus the recording's
    mean |error| (excess error, years)."""
    rho_rec = []
    for _, g in df.groupby(KEY, sort=False):
        if len(g) >= min_windows:
            rho_rec.append(_spearman(g[weight], g[err]))
    rho_rec = np.asarray(rho_rec)
    rho_rec = rho_rec[np.isfinite(rho_rec)]
    rank = df.groupby(KEY, sort=False)[weight].rank(pct=True, method="average")
    excess = df[err] - df.groupby(KEY, sort=False)[err].transform("mean")
    return {"rho_pooled": _spearman(df[weight], df[err]),
            "rho_within": float(rho_rec.mean()) if len(rho_rec) else np.nan,
            "frac_neg": float((rho_rec < 0).mean()) if len(rho_rec) else np.nan,
            "err_low_w": float(excess[rank <= 0.2].mean()),
            "err_high_w": float(excess[rank > 0.8].mean()),
            "n_rec": int(len(rho_rec))}


def excess_error_by_weight_rank(df: pd.DataFrame, weight: str, err: str,
                                n_bins: int = 10) -> pd.Series:
    """Mean excess |error| (window minus its recording's mean, years) by
    within-recording weight-rank bin (1 = lowest weight)."""
    rank = df.groupby(KEY, sort=False)[weight].rank(pct=True, method="average")
    excess = df[err] - df.groupby(KEY, sort=False)[err].transform("mean")
    b = np.minimum((rank * n_bins).apply(np.ceil).astype(int), n_bins).clip(lower=1)
    return excess.groupby(b).mean()


def error_variance_split(df: pd.DataFrame, pred: str = "pred") -> Dict[str, float]:
    """Signed window error ``e = pred - age`` split into the part shared by a
    recording's windows (variance of the recording mean of ``e``) and the part
    that varies within a recording; ``share_within`` bounds what any
    re-weighting of a recording's own windows can remove."""
    e = df[pred] - df["age"]
    m = e.groupby([df[k] for k in KEY], sort=False).transform("mean")
    between, within = float(m.var()), float((e - m).var())
    return {"var_between": between, "var_within": within,
            "share_within": within / (between + within)}


# ------------------------------------------ recording confidence vs. error
def recording_confidence(df: pd.DataFrame, weights: pd.DataFrame, eps: float = 1.0) -> pd.DataFrame:
    """Per-recording *uncertainty* scores (higher = less confident):

    ``u_A1_se``  -- standard error of the A1 pooled estimate, ``(sum w)^-1/2``;
    ``u_A1_tta`` -- mean TTA standard deviation of the windows;
    ``u_A2_mad`` -- 1.4826 x median absolute deviation of the window predictions;
    ``u_A2_down`` -- share of Huber weight removed, ``1 - mean w``;
    ``u_A3_conc`` -- concentration of the A3 weights, ``1 - ESS / n`` with
    ``ESS = 1 / sum w^2``;
    ``u_sd`` -- standard deviation of the window predictions;
    ``u_emg`` -- mean EMG ratio of the windows."""
    d = df[KEY].copy()
    d["_tta_sd"] = np.sqrt(df["tta_var"])
    d["_w1"] = inverse_variance_weights(df["tta_var"], eps)
    d["_w2"] = weights["w_A2"].to_numpy()
    d["_w3sq"] = weights["w_A3"].to_numpy() ** 2
    d["pred"], d["emg"] = df["pred"], df["emg"]
    g = d.groupby(KEY, sort=False)
    out = g.agg(n=("pred", "size"), u_A1_tta=("_tta_sd", "mean"), _w1=("_w1", "sum"),
                u_A2_down=("_w2", "mean"), _w3=("_w3sq", "sum"), u_sd=("pred", "std"),
                u_emg=("emg", "mean")).reset_index()
    med = g["pred"].transform("median")
    out["u_A2_mad"] = 1.4826 * (d["pred"] - med).abs().groupby([d[k] for k in KEY], sort=False).median().to_numpy()
    out["u_A1_se"] = 1.0 / np.sqrt(out.pop("_w1"))
    out["u_A2_down"] = 1.0 - out["u_A2_down"]
    out["u_A3_conc"] = 1.0 - (1.0 / out.pop("_w3")) / out["n"]
    return out


UNCERTAINTIES = ("u_A1_se", "u_A1_tta", "u_A2_mad", "u_A2_down", "u_A3_conc", "u_sd", "u_emg")


def _age_stratified_spearman(u, err, age, n_bins: int = 10) -> float:
    """Spearman of ``u`` and ``err`` within age deciles, averaged with the bin
    sizes as weights: removes the part of the association that only says
    'older recordings have larger errors and noisier windows'."""
    bins = pd.qcut(age, n_bins, labels=False, duplicates="drop")
    rs, ns = [], []
    for b in np.unique(bins):
        m = bins == b
        r = _spearman(u[m], err[m])
        if np.isfinite(r):
            rs.append(r)
            ns.append(m.sum())
    return float(np.average(rs, weights=ns)) if rs else np.nan


def risk_coverage(u, err, coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.5)) -> Dict[str, float]:
    """MAE of the recordings kept when the least confident (largest ``u``) are
    deferred, per coverage; ``aurc`` = area under the full risk-coverage curve,
    ``aurc_oracle`` = the same deferring by the true error (the floor)."""
    u, err = np.asarray(u, float), np.asarray(err, float)
    order = np.argsort(u, kind="stable")
    risk = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    oracle = np.cumsum(np.sort(err)) / np.arange(1, len(err) + 1)
    out = {f"mae@{int(round(c * 100))}": float(risk[max(int(round(c * len(err))) - 1, 0)])
           for c in coverages}
    out["aurc"], out["aurc_oracle"] = float(risk.mean()), float(oracle.mean())
    return out


def confidence_error_table(rec: pd.DataFrame, conf: pd.DataFrame,
                           measures: Sequence[str] = UNCERTAINTIES,
                           groups: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """One row per ``group x measure``: Spearman of the uncertainty with the
    recording's |error| (plain and within age deciles) and the selective-
    prediction MAEs of :func:`risk_coverage`. ``rec`` is a :func:`pool` output
    (its ``abs_err`` is the error being predicted)."""
    d = rec.merge(conf, on=KEY, validate="one_to_one")
    groups = groups or {"all": np.ones(len(d), bool)}
    rows = []
    for name, mask in groups.items():
        mask = np.asarray(mask(d) if callable(mask) else mask, bool)
        g = d[mask]
        if len(g) < 20:
            continue
        for m in measures:
            rows.append({"group": name, "measure": m, "n": len(g),
                         "rho": _spearman(g[m], g["abs_err"]),
                         "rho_age_strat": _age_stratified_spearman(
                             g[m].to_numpy(), g["abs_err"].to_numpy(), g["age"].to_numpy()),
                         **risk_coverage(g[m], g["abs_err"])})
    return pd.DataFrame(rows)


def regression_row(rec: pd.DataFrame) -> Dict[str, float]:
    e = rec["err"].to_numpy(float)
    return {"n": len(e), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "r": float(np.corrcoef(rec["pred"], rec["age"])[0, 1])}
