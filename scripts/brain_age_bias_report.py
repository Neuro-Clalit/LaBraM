#!/usr/bin/env python
"""Brain-age bias correction report: real vs predicted age, and the brain-age
gap before / after correction, per model and cohort.

Inputs are ``scripts/eval_age_by_cohort.py --splits val test --out <json>``
results (one per model and corpus). Every correction is fitted on the
**validation** recordings with a normal report (the non-pathological reference
of Gemein et al. 2024) and applied to the test recordings
(``labram/eval/age_analysis.py::fit_bias_correction`` / ``apply_bias_correction``):

* ``none``          -- raw prediction;
* ``cole``          -- (pred - b) / a, with pred ~ a*age + b (Cole et al. 2018);
* ``age_level``     -- pred - ((a - 1)*age + b): linear gap removed (de Lange & Cole 2020);
* ``quadratic_gap`` -- pred - (c0 + c1*age + c2*age^2): quadratic gap removed
  (Gemein et al. 2024, *Brain age revisited*).

``age_level`` and ``quadratic_gap`` use the true age, so they give a corrected
brain-age gap (BAG), not a new age estimate.

    python scripts/brain_age_bias_report.py \\
        --corpus tuab=/data/datasets/EEG-public/TUH_EEG/cohorts/all_both_v3_anymiss \\
        --corpus heedb=/data/datasets/HEEDB/v4.1/cohorts/heedb_routine_v1 \\
        --scores /data/runs/heedb/scores --models E0 E1 E2 --ensemble E1 E2 \\
        --csv docs/data/brain_age_bias_correction.csv --fig-dir docs/figures

HEEDB is plotted as a density with cells of fewer than 10 recordings
suppressed (DUA: aggregates only); TUAB as points.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from labram.eval.age_analysis import apply_bias_correction, fit_bias_correction  # noqa: E402

METHODS = ("none", "cole", "age_level", "quadratic_gap")
LABELS = {"E0": "TUH only (exp4noear)", "E1": "E1 HEEDB only", "E2": "E2 TUH+HEEDB",
          "E2c": "E2c TUH+HEEDB + conditions", "E1+E2": "E1+E2 mean"}
COHORTS = ("all", "normal", "abnormal", "unlabelled")
SMALL_CELL = 10


def load(scores, model, corpus, meta):
    """``{split: DataFrame(stem, age, pred, cohort)}`` for one model."""
    res = json.load(open(os.path.join(scores, f"{model}_{corpus}_valtest.json")))[model]["splits"]
    out = {}
    for split, r in res.items():
        rows = [{"stem": s, "pred": float(p), "age": float(meta[s]["age"]),
                 "cohort": meta[s].get("label") if meta[s].get("label") in ("normal", "abnormal")
                 else "unlabelled"} for s, p in r["predictions"].items()]
        out[split] = pd.DataFrame(rows)
    return out


def ensemble(parts):
    out = {}
    for split in parts[0]:
        base = parts[0][split].set_index("stem")
        base["pred"] = np.mean([p[split].set_index("stem").loc[base.index, "pred"] for p in parts], axis=0)
        out[split] = base.reset_index()
    return out


def auroc(score, positive):
    """P(score of a positive > score of a negative), ties counted half."""
    from scipy.stats import rankdata
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if not n1 or not n0:
        return float("nan")
    r = rankdata(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(data, model, corpus):
    """Rows of the report for one model on one corpus."""
    from scipy.stats import pearsonr, ttest_ind
    val, test = data["val"], data["test"]
    ref = val[val.cohort == "normal"]
    params = fit_bias_correction(ref.pred, ref.age)
    rows = []
    for method in METHODS:
        corr = apply_bias_correction(test.pred, params, method, age=test.age)
        t = test.assign(corr=corr, bag=corr - test.age)
        for cohort in COHORTS:
            g = t if cohort == "all" else t[t.cohort == cohort]
            if len(g) < SMALL_CELL:
                continue
            r = pearsonr(g.age, g.bag)[0] if g.age.nunique() > 2 else float("nan")
            rows.append({"corpus": corpus, "model": model, "method": method, "cohort": cohort,
                         "n": len(g), "MAE": g.bag.abs().mean(), "BAG_mean": g.bag.mean(),
                         "BAG_sd": g.bag.std(ddof=1), "r_BAG_age": r,
                         "slope_BAG_on_age": np.polyfit(g.age, g.bag, 1)[0]})
        nrm, abn = t[t.cohort == "normal"].bag, t[t.cohort == "abnormal"].bag
        if len(nrm) >= SMALL_CELL and len(abn) >= SMALL_CELL:
            rows.append({"corpus": corpus, "model": model, "method": method,
                         "cohort": "abnormal-normal", "n": len(nrm) + len(abn),
                         "BAG_mean": abn.mean() - nrm.mean(),
                         "p_welch": ttest_ind(abn, nrm, equal_var=False).pvalue,
                         "AUROC_BAG_abnormal": auroc(np.r_[abn, nrm], np.r_[np.ones(len(abn)), np.zeros(len(nrm))])})
    return rows, params


def plot_scatter(datas, corpus, path):
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import assert_max_columns, panel_labels
    models = list(datas)
    rows = (len(models) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 6.2 * rows), squeeze=False)
    axes = axes.ravel()
    for ax, m in zip(axes, models):
        t = datas[m]["test"]
        lim = (10, 95)
        if corpus == "heedb":                       # density only (DUA small-cell rule)
            hb = ax.hexbin(t.age, t.pred, gridsize=28, extent=(*lim, *lim), mincnt=SMALL_CELL,
                           cmap="viridis", bins="log")
            fig.colorbar(hb, ax=ax, label="recordings per cell (cells < 10 hidden)")
        else:
            for c, col in (("normal", "tab:green"), ("abnormal", "tab:red")):
                g = t[t.cohort == c]
                ax.scatter(g.age, g.pred, s=14, alpha=.6, c=col, label=f"{c} (n={len(g)})")
        ax.plot(lim, lim, "k--", lw=1, label="identity")
        a, b = np.polyfit(t.age, t.pred, 1)
        xs = np.array(lim)
        ax.plot(xs, a * xs + b, c="tab:orange", lw=2, label=f"fit: slope {a:.2f}")
        mae = (t.pred - t.age).abs().mean()
        ax.set_title(f"{LABELS[m]}: MAE {mae:.2f}, bias {np.mean(t.pred - t.age):+.2f}")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("chronological age (years)")
        ax.set_ylabel("predicted age (years)"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
    for ax in axes[len(models):]:
        ax.remove()
    fig.tight_layout()
    panel_labels([a for a in axes[:len(models)]])
    assert_max_columns(fig)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_bag_by_age(datas, params, corpus, path):
    """Mean BAG per age decade, raw vs quadratic-corrected (a, b), and the
    corrected BAG of normal vs abnormal recordings (c, d)."""
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import assert_max_columns, panel_labels
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    a = axes.ravel()
    colors = {"E0": "tab:gray", "E1": "tab:blue", "E2": "tab:orange", "E2c": "tab:purple",
              "E1+E2": "tab:green"}
    edges = [18, 30, 40, 50, 60, 70, 80, 100]
    names = ["18-29", "30s", "40s", "50s", "60s", "70s", "80+"]
    for m, d in datas.items():
        t = d["test"]
        dec = pd.cut(t.age, edges, right=False, labels=names)
        for ax, method in ((a[0], "none"), (a[1], "quadratic_gap")):
            bag = apply_bias_correction(t.pred, params[m], method, age=t.age) - t.age
            g = pd.DataFrame({"dec": dec, "bag": bag}).groupby("dec", observed=True)["bag"]
            stats = g.agg(["mean", "count", "sem"])
            stats = stats[stats["count"] >= SMALL_CELL]
            ax.errorbar(range(len(stats)), stats["mean"], yerr=1.96 * stats["sem"], marker="o",
                        capsize=3, c=colors[m], label=LABELS[m])
            ax.set_xticks(range(len(stats)), stats.index)
    a[0].set_title("Brain-age gap by age, raw"); a[1].set_title("after quadratic correction (Gemein 2024)")
    for ax in a[:2]:
        ax.axhline(0, c="k", lw=1); ax.set_ylabel("mean BAG = pred - age (years, 95% CI)")
        ax.set_xlabel("age group"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    lo = min(ax.get_ylim()[0] for ax in a[:2]); hi = max(ax.get_ylim()[1] for ax in a[:2])
    for ax in a[:2]:
        ax.set_ylim(lo, hi)
    for ax, method, title in ((a[2], "none", "BAG by report, raw"),
                              (a[3], "quadratic_gap", "BAG by report, quadratic-corrected")):
        data, ticks = [], []
        for m, d in datas.items():
            t = d["test"]
            bag = apply_bias_correction(t.pred, params[m], method, age=t.age) - t.age
            for c in ("normal", "abnormal"):
                data.append(bag[(t.cohort == c).to_numpy()]); ticks.append(f"{m}\n{c}")
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=.6)
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor("tab:green" if i % 2 == 0 else "tab:red"); box.set_alpha(.5)
        ax.set_xticks(range(1, len(ticks) + 1), ticks, fontsize=8)
        ax.axhline(0, c="k", lw=1); ax.set_title(title); ax.set_ylabel("BAG (years)"); ax.grid(alpha=.3)
    fig.tight_layout()
    panel_labels(list(a))
    assert_max_columns(fig)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", action="append", required=True, metavar="NAME=COHORT_DIR")
    ap.add_argument("--scores", required=True)
    ap.add_argument("--models", nargs="+", default=["E0", "E1", "E2"])
    ap.add_argument("--ensemble", nargs="*", default=["E1", "E2"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fig-dir", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    rows = []
    for spec in args.corpus:
        corpus, root = spec.split("=", 1)
        meta = json.load(open(os.path.join(root, "age_metadata.json")))["recordings"]
        datas = {m: load(args.scores, m, corpus, meta) for m in args.models}
        if args.ensemble:
            datas["+".join(args.ensemble)] = ensemble([datas[m] for m in args.ensemble])
        params = {}
        for m, d in datas.items():
            r, params[m] = evaluate(d, m, corpus)
            rows += r
        plot_scatter(datas, corpus, os.path.join(args.fig_dir, f"brain_age_scatter_{corpus}.png"))
        plot_bag_by_age(datas, params, corpus, os.path.join(args.fig_dir, f"brain_age_bias_{corpus}.png"))
    df = pd.DataFrame(rows)
    out = df.round(4)
    if "p_welch" in df:
        out["p_welch"] = df["p_welch"].map(lambda v: f"{v:.2e}" if pd.notna(v) else "")
    out.to_csv(args.csv, index=False)
    gap = df[df.cohort == "abnormal-normal"][["corpus", "model", "method", "BAG_mean", "p_welch",
                                              "AUROC_BAG_abnormal"]]
    print(gap.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
