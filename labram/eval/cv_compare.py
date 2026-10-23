"""Compare cross-validation fold models by their per-recording predictions.

Input is the ``--preds-out`` JSON of ``scripts/eval_cv_folds.py``:
``{exp: {fold: {split: {stem: [pred_years, true_years]}}}}`` with splits
``val`` / ``test`` (the fold's own held-out folds) and ``heldout`` (the cohort's
test set, the same recordings for every fold model). Everything here works on
the long frame of :func:`long_frame` (one row per exp x fold x split x stem).

* :func:`metrics` -- n / MAE / RMSE / R² / mean error (bias) in years;
* :func:`model_matrix` -- recordings x models wide frame for one split;
* :func:`ensemble` -- median (or mean) over fold models, per experiment;
* :func:`out_of_fold` -- the one prediction each recording gets from the fold
  that held it out (``test`` split);
* :func:`pairwise` -- prediction / error correlation and agreement rates for
  every pair of models;
* :func:`bootstrap_delta_mae` -- paired bootstrap of the MAE difference;
* :func:`bias_correct` -- the linear age-bias correction of the brain-age gap.
"""
import itertools
import json
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def load_predictions(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def long_frame(preds: dict) -> pd.DataFrame:
    """One row per (exp, fold, split, stem): ``pred`` / ``true`` in years,
    ``err = pred - true``."""
    rows = [(exp, int(fold), split, stem, p, t)
            for exp, folds in preds.items() for fold, splits in folds.items()
            for split, recs in splits.items() for stem, (p, t) in recs.items()]
    d = pd.DataFrame(rows, columns=["exp", "fold", "split", "stem", "pred", "true"])
    d["err"] = d.pred - d.true
    return d


def metrics(pred, true) -> Dict[str, float]:
    """n, MAE, RMSE, R² (1 - SSE/SST around the set's own mean age) and the
    mean error (bias), all in years."""
    p, t = np.asarray(pred, float), np.asarray(true, float)
    e = p - t
    sst = ((t - t.mean()) ** 2).sum()
    return {"n": int(len(e)), "MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "R2": float(1 - (e ** 2).sum() / sst) if sst > 0 else float("nan"),
            "bias": float(e.mean())}


def model_matrix(d: pd.DataFrame, split: str, value: str = "pred",
                 exps: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Recordings x models (columns ``<exp>/f<fold>``), restricted to the
    recordings every model scored."""
    s = d[d.split == split]
    if exps is not None:
        s = s[s.exp.isin(list(exps))]
    w = s.assign(model=s.exp + "/f" + s.fold.astype(str)).pivot_table(
        index="stem", columns="model", values=value, aggfunc="first")
    return w.dropna()


def ensemble(d: pd.DataFrame, split: str = "heldout", how: str = "median") -> pd.DataFrame:
    """Per experiment and recording, the median (or mean) of the fold models'
    predictions; only recordings every fold model of the experiment scored."""
    s = d[d.split == split]
    folds = s.groupby("exp").fold.nunique()
    g = s.groupby(["exp", "stem"])
    out = g.agg(pred=("pred", how), true=("true", "first"), n_models=("fold", "nunique"),
                spread=("pred", "std")).reset_index()
    out = out[out.n_models == out.exp.map(folds)]
    out["err"] = out.pred - out.true
    return out


def out_of_fold(d: pd.DataFrame) -> pd.DataFrame:
    """The ``test``-split prediction of each recording (each recording is held
    out by exactly one fold); raises when a recording appears twice."""
    s = d[d.split == "test"]
    dup = s.duplicated(["exp", "stem"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} recordings are in more than one test fold")
    return s.reset_index(drop=True)


def agreement(a, b, tau: float) -> float:
    """Share of recordings whose two predictions differ by at most ``tau`` years."""
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b)) <= tau))


def sign_agreement(ea, eb) -> float:
    """Share of recordings on which two models' errors (brain-age gaps) have
    the same sign (both older or both younger than the true age)."""
    return float(np.mean(np.sign(ea) == np.sign(eb)))


def pairwise(pred: pd.DataFrame, true: pd.Series, taus=(2.0, 5.0)) -> pd.DataFrame:
    """For every pair of columns of ``pred`` (recordings x models): Pearson r
    of the predictions and of the errors, the agreement rate within each tau,
    the sign agreement of the errors and the MAE difference (a - b)."""
    t = true.loc[pred.index].to_numpy(float)
    rows = []
    for a, b in itertools.combinations(pred.columns, 2):
        pa, pb = pred[a].to_numpy(float), pred[b].to_numpy(float)
        ea, eb = pa - t, pb - t
        row = {"a": a, "b": b, "r_pred": float(np.corrcoef(pa, pb)[0, 1]),
               "r_err": float(np.corrcoef(ea, eb)[0, 1]),
               "sign_agree": sign_agreement(ea, eb),
               "dMAE": float(np.abs(ea).mean() - np.abs(eb).mean())}
        for tau in taus:
            row[f"agree_{tau:g}y"] = agreement(pa, pb, tau)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_delta_mae(err_a, err_b, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap over recordings of MAE(a) - MAE(b): the point estimate,
    a 95% percentile interval and the share of resamples with a < b."""
    ea, eb = np.abs(np.asarray(err_a, float)), np.abs(np.asarray(err_b, float))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ea), size=(n_boot, len(ea)))
    deltas = ea[idx].mean(1) - eb[idx].mean(1)
    return {"dMAE": float(ea.mean() - eb.mean()), "lo": float(np.percentile(deltas, 2.5)),
            "hi": float(np.percentile(deltas, 97.5)), "p_a_better": float(np.mean(deltas < 0))}


def bias_correct(err, true, fit_err=None, fit_true=None) -> np.ndarray:
    """Brain-age gap with the linear age dependence removed: fit
    ``err ~ beta * true + alpha`` (OLS, ``np.polyfit``) on the reference set
    (``fit_*``, default the data itself) and subtract it."""
    err, true = np.asarray(err, float), np.asarray(true, float)
    fe = err if fit_err is None else np.asarray(fit_err, float)
    ft = true if fit_true is None else np.asarray(fit_true, float)
    beta, alpha = np.polyfit(ft, fe, 1)
    return err - (beta * true + alpha)


def mean_ci(x, z: float = 1.96) -> Dict[str, float]:
    """Mean and the normal-approximation 95% interval of the mean."""
    x = np.asarray(x, float)
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return {"n": int(len(x)), "mean": float(x.mean()), "lo": float(x.mean() - z * se),
            "hi": float(x.mean() + z * se)}
