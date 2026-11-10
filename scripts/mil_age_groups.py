#!/usr/bin/env python
"""Error per age group of the aggregation experiments, before and after age-bias correction.

Recording predictions of each experiment (collected as in
``scripts/mil_cohort_summary.py``) are scored on test per age group (overall,
TUH = TUAB eval set, HEEDB): MAE and mean signed error (brain-age gap). Three
bias corrections are fitted on each experiment's own **val** predictions of
report-normal recordings (as Gemein et al. fit on normals) and applied to its
test predictions (``labram/eval/age_analysis.py``):

* ``recalibrate``: age ~ a * pred + b (deployable; uses only the prediction);
* ``cole``: pred ~ a * age + c, inverted: (pred - c) / a (deployable);
* ``quadratic_gap``: gap ~ c0 + c1 age + c2 age^2 subtracted (uses the true
  age; a brain-age-gap quantity, not a prediction).

    python scripts/mil_age_groups.py --csv-dir docs/data/mil_age --fig-dir docs/figures
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import mil_cohort_summary as mcs  # noqa: E402
from labram.eval.age_analysis import apply_bias_correction, fit_bias_correction  # noqa: E402

AGE_BINS = [0, 20, 30, 40, 50, 60, 70, 80, 120]
AGE_LABELS = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
CORRECTIONS = ["none", "recalibrate", "cole", "quadratic_gap"]
SMALL_CELL = 10
# experiment label -> key in mil_cohort_summary.collect() (E2 is added here)
EXPERIMENTS = {
    "E2 (benchmark)": "E2",
    "E2L (mean of window predictions)": "E2L (reference)",
    "E2L mean pooling + head refit (S1 M0)": "S1 M0",
    "A3 post-hoc weights": "A3 post-hoc",
    "Frozen attention S1 M4-hq": "S1 M4-hq",
    "L1 ungated ABMIL": "S3 L1",
    "W4 gated ABMIL": "S3 W4",
    "W7 gated + retention": "S3 W7",
    "X1 gated + retention, 20 ep": "X1",
    "X3 gated hq + retention, 20 ep": "X3",
    "X4 ungated hq, 20 ep": "X4",
    "X2 ungated, 20 ep (local)": "X2 local",
}


def collect():
    preds = mcs.collect()
    for split in ("val", "test"):
        p = {}
        for f in ("E2_tuab_valtest.json", "E2_heedb_valtest.json"):
            p.update(json.load(open(os.path.join(mcs.SCORES, f)))["E2"]["splits"][split]["predictions"])
        preds.setdefault("E2", {})[split] = pd.Series(p)
    return {label: preds[key] for label, key in EXPERIMENTS.items()}


def frame(pred: pd.Series, age: pd.Series, recs: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"pred": pred}).join(recs, how="inner")
    d["age"] = age.reindex(d.index)
    d = d.dropna(subset=["age"])
    d["dataset"] = np.where(d.source == "HEEDB", "HEEDB", "TUH")
    d["age_group"] = pd.cut(d.age, AGE_BINS, right=False, labels=AGE_LABELS)
    return d


def corrected(val: pd.DataFrame, test: pd.DataFrame):
    """Test frame per correction method, fitted on val normals."""
    fit = val[val.cohort == "normal"]
    params = fit_bias_correction(fit.pred, fit.age)
    out = {}
    for m in CORRECTIONS:
        t = test.copy()
        t["pred"] = apply_bias_correction(t.pred, params, m, age=t.age)
        t["err"] = t.pred - t.age
        out[m] = t
    return out, params


def crossfit_val(val: pd.DataFrame, subjects: pd.Series, folds: int = 5) -> dict:
    """Val frame per correction method, cross-fitted: each subject-grouped fold is
    corrected with parameters fitted on the other folds' normal recordings."""
    from sklearn.model_selection import GroupKFold
    groups = subjects.reindex(val.index).fillna(pd.Series(val.index, index=val.index))
    parts = {m: [] for m in CORRECTIONS}
    for fit_idx, app_idx in GroupKFold(folds).split(val, groups=groups):
        fit, app = val.iloc[fit_idx], val.iloc[app_idx]
        part, _ = corrected(fit, app)
        for m in CORRECTIONS:
            parts[m].append(part[m])
    return {m: pd.concat(v) for m, v in parts.items()}


def cohort_table(frames: dict, split: str, exp: str) -> list:
    """MAE and mean signed error per source (TUAB / TUEG / HEEDB) x report cohort."""
    rows = []
    for m, t in frames.items():
        for src in ("TUAB", "TUEG", "HEEDB"):
            for c in ("all", "normal", "abnormal", "unlabelled"):
                g = t[(t.source == src) & ((t.cohort == c) if c != "all" else np.ones(len(t), bool))]
                if len(g) < SMALL_CELL:
                    continue
                rows.append({"experiment": exp, "split": split, "correction": m, "source": src, "cohort": c,
                             "n": len(g), "mae": g.err.abs().mean(), "mean_err": g.err.mean()})
    return rows


def subjects() -> pd.Series:
    m = json.load(open(os.path.join(mcs.DATA, "manifest.json")))["recordings"]
    return pd.Series({r["stem"]: f"{r.get('corpus')}:{r.get('subject')}" for r in m})


def summary(t: pd.DataFrame) -> dict:
    slope = np.polyfit(t.age, t.err, 1)[0]
    return {"n": len(t), "mae": t.err.abs().mean(), "mean_err": t.err.mean(),
            "gap_age_slope": slope, "gap_age_r": np.corrcoef(t.age, t.err)[0, 1]}


def separation(t: pd.DataFrame) -> dict:
    """Corrected brain-age gap of abnormal vs normal recordings: mean gaps, Cohen's d, ROC AUC."""
    from sklearn.metrics import roc_auc_score
    t = t[t.cohort.isin(["normal", "abnormal"])]
    a, n = t.err[t.cohort == "abnormal"], t.err[t.cohort == "normal"]
    sd = np.sqrt(((len(a) - 1) * a.var() + (len(n) - 1) * n.var()) / (len(a) + len(n) - 2))
    return {"gap_normal": n.mean(), "gap_abnormal": a.mean(), "cohen_d": (a.mean() - n.mean()) / sd,
            "auc": roc_auc_score(t.cohort == "abnormal", t.err), "n_normal": len(n), "n_abnormal": len(a)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--fig-dir", default=None)
    args = ap.parse_args()
    recs, age, preds = mcs.recording_table(), mcs.ages(), collect()
    groups, overall, params_rows, test_frames, sep, cohorts = [], [], [], {}, [], []
    subj = subjects()
    for exp, by in preds.items():
        val, test = frame(by["val"], age["val"], recs), frame(by["test"], age["test"], recs)
        corr, params = corrected(val, test)
        params_rows.append({"experiment": exp, **params, "fit_n": int((val.cohort == "normal").sum())})
        test_frames[exp] = corr
        cohorts += cohort_table(corr, "test", exp)
        vcf = crossfit_val(val, subj)
        cohorts += cohort_table(vcf, "val (cross-fitted)", exp)
        for m, t in corr.items():
            for ds in ("all", "TUH", "HEEDB"):
                s = t if ds == "all" else t[t.dataset == ds]
                overall.append({"experiment": exp, "correction": m, "dataset": ds, **summary(s)})
                sep.append({"experiment": exp, "correction": m, "dataset": ds, **separation(s)})
                g = s.groupby("age_group", observed=False)
                for ag, gg in g:
                    n = len(gg)
                    groups.append({"experiment": exp, "correction": m, "dataset": ds, "age_group": ag, "n": n,
                                   "mae": gg.err.abs().mean() if n >= SMALL_CELL else np.nan,
                                   "mean_err": gg.err.mean() if n >= SMALL_CELL else np.nan})
    os.makedirs(args.csv_dir, exist_ok=True)
    groups, overall = pd.DataFrame(groups), pd.DataFrame(overall)
    groups.to_csv(os.path.join(args.csv_dir, "age_groups_test.csv"), index=False)
    overall.to_csv(os.path.join(args.csv_dir, "overall_test.csv"), index=False)
    cohorts = pd.DataFrame(cohorts)
    cohorts.to_csv(os.path.join(args.csv_dir, "corrections_by_cohort.csv"), index=False)
    for split in ("val (cross-fitted)", "test"):
        print(f"== MAE per source x cohort, {split} (none / recalibrate / cole / quadratic_gap)")
        c = cohorts[cohorts.split == split]
        w = c.pivot_table(index=["source", "cohort", "experiment"], columns="correction", values="mae")
        print(w[CORRECTIONS].round(2).to_string())
    sep = pd.DataFrame(sep)
    sep.to_csv(os.path.join(args.csv_dir, "gap_separation_test.csv"), index=False)
    pd.DataFrame(params_rows).to_csv(os.path.join(args.csv_dir, "correction_params.csv"), index=False)
    pd.set_option("display.width", 250)
    for ds in ("all", "TUH", "HEEDB"):
        for col in ("mae", "mean_err"):
            print(f"== test {col}, {ds}, no correction")
            print(groups[(groups.dataset == ds) & (groups.correction == "none")].pivot(
                index="experiment", columns="age_group", values=col).reindex(list(preds)).round(2).to_string())
    print("== n per age group (all / TUH / HEEDB)")
    print(groups[(groups.correction == "none") & (groups.experiment == list(preds)[1])].pivot(
        index="dataset", columns="age_group", values="n").to_string())
    print("== overall")
    print(overall.pivot_table(index=["experiment"], columns=["dataset", "correction"],
                              values="mae").reindex(list(preds)).round(2).to_string())
    print(overall.pivot_table(index=["experiment"], columns=["dataset", "correction"],
                              values="gap_age_slope").reindex(list(preds)).round(3).to_string())
    print("== abnormal vs normal separation by the gap (AUC / Cohen d)")
    for col in ("auc", "cohen_d", "gap_normal", "gap_abnormal"):
        print(col)
        print(sep.pivot_table(index="experiment", columns=["dataset", "correction"], values=col)
              .reindex(list(preds)).round(3).to_string())
    for m in CORRECTIONS[1:]:
        for col in ("mae", "mean_err"):
            print(f"== test {col}, all, {m}")
            print(groups[(groups.dataset == "all") & (groups.correction == m)].pivot(
                index="experiment", columns="age_group", values=col).reindex(list(preds)).round(2).to_string())
    if args.fig_dir:
        plot(groups, list(preds), args.fig_dir)


def plot(groups, exps, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import assert_max_columns, panel_labels
    show = ["E2 (benchmark)", "E2L (mean of window predictions)", "A3 post-hoc weights",
            "L1 ungated ABMIL", "W4 gated ABMIL", "W7 gated + retention"]
    x = np.arange(len(AGE_LABELS))
    for name, corr in (("none", "none"), ("recalibrate", "recalibrate")):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for row, ds in enumerate(("TUH", "HEEDB")):
            g = groups[(groups.dataset == ds) & (groups.correction == corr)]
            for col, (metric, label) in enumerate((("mae", "MAE (y)"), ("mean_err", "mean signed error (y)"))):
                ax = axes[row][col]
                for e in show:
                    v = g[g.experiment == e].set_index("age_group").reindex(AGE_LABELS)[metric]
                    ax.plot(x, v.to_numpy(), "-o", ms=4, label=e)
                if metric == "mean_err":
                    ax.axhline(0, color="0.6", lw=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(AGE_LABELS)
                ax.set_xlabel("true age group (y)")
                ax.set_ylabel(label)
                ax.set_title(f"{ds} test, {label.split(' (')[0]}")
                if row == 0 and col == 0:
                    ax.legend(fontsize=7)
        fig.tight_layout()
        panel_labels([axes[0][0], axes[0][1], axes[1][0], axes[1][1]])
        assert_max_columns(fig)
        path = os.path.join(fig_dir, f"mil_age_groups_{name}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
