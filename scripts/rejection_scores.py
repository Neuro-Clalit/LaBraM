#!/usr/bin/env python
"""Compare label-free confidence scores as recording-rejection rules for the
attention (ABMIL) brain-age ensembles.

For fold model ``k`` and recording ``r`` with windows ``i`` (prediction
``p_i``, attention weight ``a_i``, ``sum_i a_i = 1``, ``n`` windows), from the
``eval_age_by_cohort.py --save-windows`` parquets:

* ``spread``      attention-weighted SD of the window predictions,
                  ``sqrt(sum a_i (p_i - yhat)^2)``, ``yhat = sum a_i p_i`` (the
                  rule used so far, ``keep80r``);
* ``spread_unw``  the plain SD of the window predictions (attention ignored);
* ``sem``         SD of the pooled prediction if windows were independent,
                  ``spread * sqrt(sum a_i^2)``;
* ``neff``        effective share of windows used, ``1 / (n sum a_i^2)`` (1 =
                  uniform attention, small = attention on few windows);
* ``entropy``     ``H(a) / log n``;
* ``amax``        the largest relative weight, ``n * max a_i``;
* ``shift``       how far attention moves the answer, ``|yhat - mean_i p_i|``;
* ``low_att_dev`` attention-weighted mean of ``|p_i - yhat|`` over the windows
                  with below-median attention minus that over the others (do
                  the down-weighted windows disagree with the answer?).

Ensemble scores are the fold means of the above, plus

* ``fold_sd``     SD over the fold models of ``yhat`` (model disagreement);
* ``total``       ``sqrt(mean_k sem^2 + fold_sd^2)``;
* ``rank_spread_fold``  mean of the within-cohort ranks of ``spread`` and ``fold_sd``;
* ``pred_age``    the predicted age itself (errors grow with age, so rejecting
                  the oldest predictions is the confound to beat);
* ``spread|age``, ``total|age``  the score ranked within deciles of the
                  predicted age (the part not explained by predicted age);
* ``learned``     a gradient-boosted regressor of ``|error|`` on all the scores
                  above, fitted on the in-domain cohorts (``--fit-on``,
                  cross-fitted there by subject) and applied unchanged elsewhere.

A unit (recording; TDBRAIN: session = mean over its files) is rejected when its
score is among the ``1 - coverage`` most uncertain of its cohort. The uncertain
direction of each score is the sign of its Spearman correlation with ``|error|``
on the ``--fit-on`` cohorts (so it is never chosen on the cohort it is applied
to, except for those cohorts themselves). Reported per cohort and score:
Spearman rho with ``|error|``, MAE of the kept 80 %, their mean age, and the
area under the risk-coverage curve (mean kept MAE over coverages 0.5 .. 1.0),
with ``random`` (no information: the MAE of all units at every coverage) and
``oracle`` (reject by true ``|error|``). ``d_vs_spread`` is the kept-80 % MAE
minus that of ``spread`` with a 95 % interval from 500 bootstrap resamples of
the units (the rejection threshold is re-set in every resample).

    python scripts/rejection_scores.py --models P4ens P3ens \\
        --set harvard:<cohort>:<eval.json>:<windows>:harvard ... \\
        --fit-on temple harvard --out docs/data/external_eval/rejection_scores.csv \\
        --figure docs/figures/tdbrain/rejection_curves.png
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from external_cohort_table import load_meta  # noqa: E402

FOLD_SCORES = ["spread", "spread_unw", "sem", "neff", "entropy", "amax", "shift", "low_att_dev"]
FEATURES = FOLD_SCORES + ["fold_sd", "pred_age"]
COVERAGES = np.round(np.arange(0.5, 1.0001, 0.05), 2)


def fold_scores(path):
    """Per recording of one fold model: yhat and the window-level scores."""
    w = pd.read_parquet(path, columns=["recording", "alpha", "window_pred"])
    g = w.groupby("recording", sort=False)
    a = w["alpha"] / g["alpha"].transform("sum")
    n = g["alpha"].transform("size")
    p = w["window_pred"]
    yhat = (a * p).groupby(w["recording"], sort=False).transform("sum")
    dev = (p - yhat).abs()
    low = a < g["alpha"].transform("median") / g["alpha"].transform("sum")
    rec = w["recording"]

    def wsum(x, mask=None):
        x = x if mask is None else x.where(mask, 0.0)
        return x.groupby(rec, sort=False).sum()

    sum_a2 = wsum(a ** 2)
    spread = np.sqrt(wsum(a * (p - yhat) ** 2))
    a_low, a_high = wsum(a, low), wsum(a, ~low)
    out = pd.DataFrame({
        "yhat": wsum(a * p), "spread": spread, "spread_unw": g["window_pred"].std(ddof=0),
        "sem": spread * np.sqrt(sum_a2), "neff": 1.0 / (g.size() * sum_a2),
        "entropy": -wsum(a * np.log(a.clip(lower=1e-12))) / np.log(g.size().clip(lower=2)),
        "amax": (a * n).groupby(rec, sort=False).max(),
        "shift": (wsum(a * p) - g["window_pred"].mean()).abs(),
        "low_att_dev": (wsum(a * dev, low) / a_low.clip(lower=1e-12)).where(a_low > 0, 0.0)
                       - wsum(a * dev, ~low) / a_high.clip(lower=1e-12),
    })
    return out


def cohort_units(meta, wdir, members):
    """One row per scoring unit with age, ensemble prediction and every score."""
    per = [fold_scores(os.path.join(wdir, f"{m}_test.parquet")) for m in members]
    idx = per[0].index
    stack = {c: np.stack([f.reindex(idx)[c].to_numpy() for f in per]) for c in ["yhat"] + FOLD_SCORES}
    d = pd.DataFrame({c: stack[c].mean(axis=0) for c in FOLD_SCORES}, index=idx)
    d["pred"] = stack["yhat"].mean(axis=0)
    d["fold_sd"] = stack["yhat"].std(axis=0)
    d["sem2"] = (stack["sem"] ** 2).mean(axis=0)
    cols = ["subject", "age"] + (["session"] if "task" in meta else [])
    d = meta[cols].join(d, how="inner")
    if "task" in meta:                                   # session = mean over its files
        d = d.groupby(["subject", "session"]).mean(numeric_only=True).reset_index()
    d["pred_age"] = d["pred"]
    d["total"] = np.sqrt(d["sem2"] + d["fold_sd"] ** 2)
    d["abs_err"] = (d["pred"] - d["age"]).abs()
    return d.reset_index(drop=True)


def add_cohort_scores(d):
    """Scores that are ranks within the cohort (label-free)."""
    d["rank_spread_fold"] = (d["spread"].rank(pct=True) + d["fold_sd"].rank(pct=True)) / 2
    dec = pd.qcut(d["pred_age"], 10, labels=False, duplicates="drop")
    d["spread|age"] = d.groupby(dec)["spread"].rank(pct=True)
    d["total|age"] = d.groupby(dec)["total"].rank(pct=True)
    return d


def risk_coverage(score, abs_err, age):
    """Kept MAE per coverage (most certain first), kept mean age at 0.8."""
    order = np.argsort(score, kind="stable")
    e, a = abs_err[order], age[order]
    mae = [e[:max(1, int(round(c * len(e))))].mean() for c in COVERAGES]
    k = max(1, int(round(0.8 * len(e))))
    return np.array(mae), a[:k].mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True, metavar="NAME:COHORT:EVAL.json:WINDOWS_DIR[:FAMILY]")
    ap.add_argument("--models", nargs="+", default=["P4ens"])
    ap.add_argument("--fit-on", nargs="+", default=["temple", "harvard"])
    ap.add_argument("--min-age", type=float, default=18.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figure", default=None)
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingRegressor
    rows, curves = [], {}
    for model in args.models:
        data = {}
        for spec in args.set:
            name, cdir, ev, wdir, *family = spec.split(":")
            members = json.load(open(ev))[model]["members"]
            data[name] = add_cohort_scores(cohort_units(load_meta(cdir, family[0] if family else None, args.min_age),
                                                        wdir, members))
        fit = pd.concat([data[c].assign(cohort=c) for c in args.fit_on], ignore_index=True)
        # learned score: cross-fitted on the fit cohorts (5 subject-grouped folds), full fit elsewhere
        subs = fit["subject"].astype(str) + "@" + fit["cohort"]
        fold = pd.Series(np.random.default_rng(0).permutation(subs.nunique()) % 5, index=subs.unique())[subs].to_numpy()
        gbm = dict(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0, loss="absolute_error")
        oof = np.empty(len(fit))
        for k in range(5):
            m = HistGradientBoostingRegressor(**gbm, random_state=0).fit(fit.loc[fold != k, FEATURES], fit.loc[fold != k, "abs_err"])
            oof[fold == k] = m.predict(fit.loc[fold == k, FEATURES])
        full = HistGradientBoostingRegressor(**gbm, random_state=0).fit(fit[FEATURES], fit["abs_err"])
        for c, d in data.items():
            d["learned"] = oof[(fit["cohort"] == c).to_numpy()] if c in args.fit_on else full.predict(d[FEATURES])
        fit = pd.concat([data[c] for c in args.fit_on], ignore_index=True)
        scores = FOLD_SCORES + ["fold_sd", "total", "rank_spread_fold", "pred_age", "spread|age", "total|age", "learned"]
        sign = {s: (1.0 if stats.spearmanr(fit[s], fit["abs_err"])[0] >= 0 else -1.0) for s in scores}
        for c, d in data.items():
            e, age = d["abs_err"].to_numpy(), d["age"].to_numpy()
            rng = np.random.default_rng(0)
            specials = {"random": None, "oracle": e}
            signed = {s: (specials[s] if s in specials else sign[s] * d[s].to_numpy()) for s in scores + ["oracle"]}
            k80 = max(1, int(round(0.8 * len(e))))
            boot = {s: [] for s in signed}
            for _ in range(500):                              # paired: the same resample for every score
                i = rng.integers(0, len(e), len(e))
                for s, sc in signed.items():
                    boot[s].append(e[i][np.argsort(sc[i], kind="stable")[:k80]].mean())
            for s in scores + list(specials):
                if s == "random":
                    mae, kept_age, rho, ci = np.full(len(COVERAGES), e.mean()), age.mean(), np.nan, (np.nan, np.nan)
                else:
                    mae, kept_age = risk_coverage(signed[s], e, age)
                    rho = np.nan if s in specials else stats.spearmanr(d[s], e)[0]
                    ci = np.percentile(np.array(boot[s]) - np.array(boot["spread"]), [2.5, 97.5])
                rows.append({"model": model, "cohort": c, "score": s, "n": len(d),
                             "uncertain_when": "" if s in specials else ("high" if sign[s] > 0 else "low"),
                             "spearman_abs_err": rho, "mae_all": e.mean(),
                             "mae_keep80": mae[list(COVERAGES).index(0.8)], "age_all": age.mean(),
                             "age_keep80": kept_age, "aurc_50_100": mae.mean(),
                             "d_vs_spread_lo": ci[0], "d_vs_spread_hi": ci[1]})
                curves[(model, c, s)] = mae
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    for model in args.models:
        t = df[df.model == model]
        with pd.option_context("display.width", 250, "display.max_rows", 500):
            print(f"== {model}: MAE of the kept 80 % (all = no rejection)")
            piv = t.pivot(index="score", columns="cohort", values="mae_keep80")
            piv.loc["(no rejection)"] = t.groupby("cohort")["mae_all"].first()
            print(piv[[c.split(":")[0] for c in args.set]].round(2).to_string())
            print("-- Spearman rho(score, |error|)")
            print(t.pivot(index="score", columns="cohort", values="spearman_abs_err")[[c.split(":")[0] for c in args.set]].round(2).to_string())
            print("-- kept-80 % MAE minus spread's, 95 % bootstrap interval")
            ci = t.assign(ci=t.d_vs_spread_lo.round(2).astype(str) + " .. " + t.d_vs_spread_hi.round(2).astype(str))
            print(ci[ci.score.isin(["fold_sd", "total", "learned", "neff", "amax", "total|age"])].pivot(
                index="score", columns="cohort", values="ci")[[c.split(":")[0] for c in args.set]].to_string())
            print("-- mean age of the kept 80 %")
            piv = t.pivot(index="score", columns="cohort", values="age_keep80")
            piv.loc["(no rejection)"] = t.groupby("cohort")["age_all"].first()
            print(piv[[c.split(":")[0] for c in args.set]].round(1).to_string())
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from labram.eval.age_plots import assert_max_columns, panel_labels
        model = args.models[0]
        names = [c.split(":")[0] for c in args.set]
        show = [("spread", "#2a78d6", "-"), ("fold_sd", "#eb6834", "-"), ("total", "#1baf7a", "-"),
                ("learned", "#6250d6", "-"), ("neff", "#eda100", "-"), ("pred_age", "#6b6a64", "--"),
                ("random", "#b0afa8", ":"), ("oracle", "#1f1f1e", ":")]
        fig, axes = plt.subplots((len(names) + 1) // 2, 2, figsize=(13, 4.2 * ((len(names) + 1) // 2)), squeeze=False)
        for ax, c in zip(axes.flat, names):
            for s, color, ls in show:
                ax.plot(COVERAGES * 100, curves[(model, c, s)], color=color, ls=ls, lw=2, label=s)
            ax.set_title(f"{c} (n = {int(df[(df.model == model) & (df.cohort == c)].n.iloc[0]):,})", fontsize=11)
            ax.set_xlabel("coverage: share of recordings kept (%)")
            ax.set_ylabel("MAE of the kept recordings (y)")
            ax.invert_xaxis()
            ax.grid(color="#e4e3df", lw=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        for ax in axes.flat[len(names):]:
            ax.set_visible(False)
        axes.flat[0].legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        panel_labels(list(axes.flat[:len(names)]))
        assert_max_columns(fig)
        os.makedirs(os.path.dirname(os.path.abspath(args.figure)), exist_ok=True)
        fig.savefig(args.figure, dpi=140)
        print(args.figure)


if __name__ == "__main__":
    main()
