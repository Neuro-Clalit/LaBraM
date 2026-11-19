#!/usr/bin/env python
"""Brain-age gap of the clinical groups of an external cohort against its own controls.

For a cohort whose manifest has a ``healthy`` flag and a ``clinical_group``
column (AHEPA: alzheimer / ftd / control; the Parkinson's cohorts: parkinson /
control), per model and group: recordings, mean age, MAE, mean gap
``g = prediction - age``, and the group's gap against the controls at the same
age and sex,

    g_r = b0 + b1 * 1[r in group] + b2 * age_r + b3 * male_r + e_r

fitted on the group and the controls (``gap_adj`` = b1, with a 95 % interval
and p from heteroscedasticity-robust HC3 standard errors). The age term absorbs
the models' age bias (predictions are compressed toward the mean age), which
would otherwise show as a gap difference between groups of different age.

With ``--scores`` (default: every clinical score column present: MMSE, MoCA,
UPDRS, LEDD, disease duration) it also reports, inside each patient group, the
partial correlation of the gap with the score given age: the Pearson r of the
two residuals after a linear fit on age.

Ages only score; nothing is fitted for the models. One recording per person is
assumed (true for these cohorts).

    python scripts/clinical_cohort_gap.py --cohort <cohort dir> --eval <eval json> \\
        --models E2L P3ens P4ens --out docs/data/external_eval/ahepa/gap_by_group.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

SCORES = ["mmse", "moca", "updrs", "ledd", "disease_duration"]


def ols_hc3(y, X):
    """OLS coefficients and HC3 standard errors."""
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    lev = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    w = (resid / (1 - lev)) ** 2
    cov = xtx_inv @ (X * w[:, None]).T @ X @ xtx_inv
    return beta, np.sqrt(np.diag(cov))


def adjusted_gap(d, in_group, in_ref):
    x = d[(in_group | in_ref) & d["sex"].isin(["M", "F"])]
    flag = in_group.loc[x.index].to_numpy(float)
    if flag.sum() < 8 or (1 - flag).sum() < 8:
        return {}
    X = np.column_stack([np.ones(len(x)), flag, (x["age"] - x["age"].mean()).to_numpy(float),
                         (x["sex"] == "M").to_numpy(float)])
    beta, se = ols_hc3(x["gap"].to_numpy(float), X)
    t = beta[1] / se[1]
    return {"gap_adj": beta[1], "gap_adj_lo": beta[1] - 1.96 * se[1], "gap_adj_hi": beta[1] + 1.96 * se[1],
            "gap_adj_p": 2 * stats.t.sf(abs(t), len(x) - X.shape[1]), "controls": int((1 - flag).sum())}


def partial_r(x, y, z):
    """Pearson r of x and y after removing a linear trend on z from both, with its p-value."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 10:
        return np.nan, np.nan, int(ok.sum())
    x, y, z = x[ok], y[ok], z[ok]
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    r = np.corrcoef(rx, ry)[0, 1]
    t = r * np.sqrt((len(x) - 3) / max(1 - r ** 2, 1e-12))
    return r, 2 * stats.t.sf(abs(t), len(x) - 3), int(ok.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--scores", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    meta = pd.DataFrame(json.load(open(os.path.join(args.cohort, "manifest.json")))["recordings"]).set_index("stem")
    if "clinical_group" not in meta or not (~meta["healthy"].astype(bool)).any():
        print(f"{args.cohort}: no clinical groups, nothing to compare")
        return
    scores = [s for s in (args.scores if args.scores is not None else SCORES) if s in meta]
    res = json.load(open(args.eval))
    rows = []
    for model in args.models:
        d = meta.join(pd.Series(res[model]["splits"]["test"]["predictions"], dtype=float).rename("pred"), how="inner")
        d["gap"] = d["pred"] - d["age"]
        ref = d["healthy"].astype(bool)
        groups = ["control"] + sorted(g for g in d["clinical_group"].unique() if g != "control")
        if len(groups) > 2:
            groups.append("all patients")
        for g in groups:
            mask = ref if g == "control" else (~ref if g == "all patients" else d["clinical_group"] == g)
            x = d[mask]
            row = {"model": model, "group": g, "n": len(x), "age_mean": x["age"].mean(), "age_sd": x["age"].std(),
                   "mae": x["gap"].abs().mean(), "gap_mean": x["gap"].mean(), "gap_sd": x["gap"].std()}
            if g != "control":
                row.update(adjusted_gap(d, mask, ref))
                for s in scores:
                    r, p, n = partial_r(x["gap"].to_numpy(float), pd.to_numeric(x[s], errors="coerce").to_numpy(float),
                                        x["age"].to_numpy(float))
                    row.update({f"r_gap_{s}": r, f"p_gap_{s}": p, f"n_{s}": n})
            rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
