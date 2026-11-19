#!/usr/bin/env python
"""Do window weights track window errors, and recording confidences recording errors?

Reads the ``<NAME>_windows.parquet`` caches of
``scripts/window_confidence_predict.py`` and, per run:

1. pools every recording with the mean, median, 10 %-trimmed mean, the TTA
   mean and the three confidence-weighted estimators of Proposal A
   (``docs/window_aggregation_plan.md``): A1 inverse TTA variance, A2 Huber,
   A3 learned softmax over window descriptors. ``eps`` (A1) and ``delta`` (A2)
   are chosen on val; A3 is fitted on val (cross-fitted there);
2. correlates each window's weight with its own absolute error, over all
   windows and within recordings, and splits the window error variance into
   its between- and within-recording parts;
3. builds per-recording uncertainty scores and correlates them with the
   recording's absolute error (plain and within age deciles), with
   selective-prediction (risk-coverage) curves.

    python scripts/window_confidence_report.py --in-dir /data/runs/heedb/window_conf \\
        --run E2L --run E2 --out-dir /data/runs/heedb/window_conf/report \\
        --fig-dir docs/figures
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from labram.eval import window_confidence as wc

EPS_GRID = (0.25, 1.0, 4.0, 16.0, 64.0)
DELTA_GRID = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)
GROUPS = {
    "all": lambda d: np.ones(len(d), bool),
    "temple": lambda d: d["dataset"] == "temple",
    "harvard": lambda d: d["dataset"] == "harvard",
    "normal": lambda d: d["cohort"] == "normal",
    "abnormal": lambda d: d["cohort"] == "abnormal",
}
# dataset x report cohort, e.g. "harvard_normal"
GROUPS.update({f"{ds}_{c}": (lambda d, ds=ds, c=c: (d["dataset"] == ds) & (d["cohort"] == c))
               for ds in ("temple", "harvard") for c in ("normal", "abnormal", "unlabelled")})


def val_mae(rec):
    return float(rec.loc[rec.split == "val", "abs_err"].mean())


def select(df, make, grid):
    """(best value on val, {value: val MAE})."""
    scores = {v: val_mae(make(v)) for v in grid}
    return min(scores, key=scores.get), scores


def poolings(df):
    """{method: per-recording frame}, the window weights and the selected settings."""
    eps, eps_scores = select(df, lambda e: wc.pool(df, wc.inverse_variance_weights(df.tta_var, e),
                                                   "tta_mean"), EPS_GRID)
    delta, delta_scores = select(df, lambda d: wc.pool(df, wc.huber_weights(df, "pred", d)[0]),
                                 DELTA_GRID)
    w = pd.DataFrame(index=df.index)
    w["w_A1"] = wc.inverse_variance_weights(df.tta_var, eps)
    w["w_A2"] = wc.huber_weights(df, "pred", delta)[0]
    w["w_A3"], theta = wc.a3_weights(df)
    w["w_mean"] = 1.0
    med = df.groupby(wc.KEY, sort=False)["pred"].transform("median")
    recs = {
        "mean": wc.pool(df),
        "median": wc.pool(df.assign(pred=med)),
        "trimmed10": wc.trimmed_pool(df),
        "tta_mean": wc.pool(df, None, "tta_mean"),
        "A1": wc.pool(df, w["w_A1"], "tta_mean"),
        "A2": wc.pool(df, w["w_A2"]),
        "A3": wc.pool(df, w["w_A3"]),
    }
    settings = {"eps": eps, "eps_val_mae": eps_scores, "delta": delta,
                "delta_val_mae": delta_scores, "theta": theta.to_dict()}
    return recs, w, settings


def mae_table(recs):
    rows = []
    for method, rec in recs.items():
        for split, d in rec.groupby("split"):
            for g, f in GROUPS.items():
                m = np.asarray(f(d), bool)
                if m.sum() >= 2:
                    rows.append({"method": method, "split": split, "group": g,
                                 **wc.regression_row(d[m])})
    return pd.DataFrame(rows)


def curve_split(df):
    """Split the figures show: test, else (smoke runs) val."""
    return "test" if (df["split"] == "test").any() else "val"


def bootstrap_delta(recs, base="mean", n_boot=2000, seed=0):
    """Paired bootstrap over recordings of MAE(method) - MAE(base), per split x
    group: point estimate and 95 % percentile interval (years)."""
    rng = np.random.default_rng(seed)
    rows = []
    ref = recs[base]
    for method, rec in recs.items():
        if method == base:
            continue
        d = rec.merge(ref[wc.KEY + ["abs_err"]], on=wc.KEY, suffixes=("", "_base"))
        for split, ds in d.groupby("split"):
            for g, f in GROUPS.items():
                m = np.asarray(f(ds), bool)
                diff = (ds["abs_err"] - ds["abs_err_base"]).to_numpy()[m]
                if len(diff) < 20:
                    continue
                boots = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(1)
                lo, hi = np.percentile(boots, [2.5, 97.5])
                rows.append({"method": method, "split": split, "group": g, "n": len(diff),
                             "delta_mae": float(diff.mean()), "ci_lo": float(lo), "ci_hi": float(hi)})
    return pd.DataFrame(rows)


def tta_profile(df, split):
    """Windows by within-recording TTA-variance decile (1 = most invariant):
    median TTA SD, signal descriptors and the mean |error| of the TTA mean."""
    d = df[df["split"] == split]
    rank = d.groupby(wc.KEY, sort=False)["tta_var"].rank(pct=True, method="average")
    dec = np.ceil(rank * 10).clip(1, 10).astype(int).rename("decile")
    err = (d["tta_mean"] - d["age"]).abs()
    return d.assign(tta_sd=np.sqrt(d["tta_var"]), abs_err=err).groupby(dec).agg(
        tta_sd=("tta_sd", "median"), amp_uv=("amp_uv", "median"), max_abs=("max_abs", "median"),
        min_std=("min_std", "median"), emg=("emg", "median"), abs_err=("abs_err", "mean"))


SKIPS = (0, 1, 3, 6, 12)


def position_tables(df):
    """Window position effects: mean |error| by window index within the
    evaluated stretch (0 = first window after the start trim), per split x
    dataset, and the mean / median pooled MAE when each recording's first
    ``k`` windows are skipped (``k`` in ``SKIPS``)."""
    k = (df["window"] - df.groupby(wc.KEY, sort=False)["window"].transform("min")).rename("k")
    ae = (df["pred"] - df["age"]).abs()
    by_pos = ae.groupby([df["split"], df["dataset"], k]).mean().rename("abs_err").reset_index()
    rows = []
    for skip in SKIPS:
        sub = df[k >= skip]
        med = sub.groupby(wc.KEY, sort=False)["pred"].transform("median")
        for pooling, rec in (("mean", wc.pool(sub)), ("median", wc.pool(sub.assign(pred=med)))):
            for (split, ds), g in rec.groupby(["split", "dataset"]):
                rows.append({"skip": skip, "pooling": pooling, "split": split, "dataset": ds,
                             "mae": float(g["abs_err"].mean())})
    return by_pos, pd.DataFrame(rows)


def window_tables(df, w):
    """Weight-vs-own-error correlations per split x method, plus the raw
    descriptors' correlation with the error (sign: larger descriptor = less
    reliable, so a positive rho is the useful direction)."""
    rows, rank_curves = [], {}
    err = {"pred": (df.pred - df.age).abs(), "tta_mean": (df.tta_mean - df.age).abs()}
    d = df.assign(err_pred=err["pred"], err_tta=err["tta_mean"], **{c: w[c] for c in w})
    for split, g in d.groupby("split"):
        for method in ("A1", "A2", "A3"):
            e = "err_tta" if wc.POOLED_ON[method] == "tta_mean" else "err_pred"
            rows.append({"split": split, "method": method, "kind": "weight",
                         **wc.weight_error_correlation(g, f"w_{method}", e)})
            if split == curve_split(df):
                rank_curves[method] = wc.excess_error_by_weight_rank(g, f"w_{method}", e).to_dict()
        for desc in ("emg", "emg_max", "max_abs", "tta_var"):
            r = wc.weight_error_correlation(g.assign(_neg=-g[desc]), "_neg", "err_pred")
            rows.append({"split": split, "method": desc, "kind": "descriptor (as -x)", **r})
        rows.append({"split": split, "method": "variance split", "kind": "error",
                     **wc.error_variance_split(g)})
    return pd.DataFrame(rows), rank_curves


def confidence_tables(df, w, recs):
    conf = wc.recording_confidence(df, w)
    out, curves = [], {}
    for split in sorted(set(df["split"]), key=["val", "test"].index):
        rec = recs["mean"][recs["mean"].split == split]
        groups = {k: f for k, f in GROUPS.items()}
        t = wc.confidence_error_table(rec, conf, groups=groups).assign(split=split, error_of="mean")
        out.append(t)
        # each estimator's own error against its own confidence
        for method, m in (("A1", "u_A1_se"), ("A2", "u_A2_mad"), ("A3", "u_A3_conc")):
            r = recs[method][recs[method].split == split]
            out.append(wc.confidence_error_table(r, conf, measures=[m])
                       .assign(split=split, error_of=method))
        if split == curve_split(df):
            d = rec.merge(conf, on=wc.KEY)
            for m in wc.UNCERTAINTIES + ("oracle",):
                u = d["abs_err"] if m == "oracle" else d[m]
                order = np.argsort(np.asarray(u), kind="stable")
                risk = np.cumsum(d["abs_err"].to_numpy()[order]) / np.arange(1, len(d) + 1)
                cov = np.arange(1, len(d) + 1) / len(d)
                curves[m] = (cov, risk)
            curves["_deciles"] = {m: d.groupby(pd.qcut(d[m].rank(method="first"), 10, labels=False))
                                  ["abs_err"].mean().to_list() for m in wc.UNCERTAINTIES}
    return pd.concat(out, ignore_index=True), curves, conf


LABELS = {"A1": "A1 inverse TTA variance", "A2": "A2 Huber", "A3": "A3 learned softmax",
          "u_A1_se": "A1 pooled SE", "u_A1_tta": "mean TTA SD", "u_A2_mad": "A2 MAD of windows",
          "u_A2_down": "A2 down-weighted share", "u_A3_conc": "A3 weight concentration",
          "u_sd": "SD of windows", "u_emg": "mean EMG ratio", "oracle": "oracle (true error)"}


COHORTS = [(ds, c) for c in ("normal", "abnormal", "unlabelled") for ds in ("temple", "harvard")]
DATASET_NAME = {"temple": "TUH", "harvard": "HEEDB"}
COHORT_MEASURES = ("u_A1_tta", "u_A2_down", "u_A3_conc", "u_sd", "u_emg")


def cohort_tables(df, w, recs, conf, n_bins=5):
    """Per dataset x report cohort, over val + test together (A3 val weights
    are cross-fitted): the window-level excess error by within-recording
    weight-rank bin and the within-recording Spearman of each weight with its
    window's |error|; the recording-level mean |error| (mean pooling) by
    uncertainty bin and the Spearman of each uncertainty score with it."""
    err = {"pred": (df.pred - df.age).abs(), "tta_mean": (df.tta_mean - df.age).abs()}
    d = df.assign(err_pred=err["pred"], err_tta=err["tta_mean"], **{c: w[c] for c in w})
    rec = recs["mean"].merge(conf, on=wc.KEY)
    curves, rows = {}, []
    for ds, c in COHORTS:
        g = d[(d.dataset == ds) & (d.cohort == c)]
        r = rec[(rec.dataset == ds) & (rec.cohort == c)]
        if len(r) < 20:
            continue
        cur = curves[(ds, c)] = {"n_rec": len(r), "window": {}, "record": {}}
        for method in ("A1", "A2", "A3"):
            e = "err_tta" if wc.POOLED_ON[method] == "tta_mean" else "err_pred"
            cur["window"][method] = wc.excess_error_by_weight_rank(g, f"w_{method}", e, n_bins).to_dict()
            st = wc.weight_error_correlation(g, f"w_{method}", e)
            rows.append({"dataset": ds, "cohort": c, "level": "window", "score": method,
                         "n_rec": len(r), "rho": st["rho_within"], "frac_neg": st["frac_neg"]})
        for m in COHORT_MEASURES:
            b = pd.qcut(r[m].rank(method="first"), n_bins, labels=False) + 1
            cur["record"][m] = r.groupby(b)["abs_err"].mean().to_dict()
            rows.append({"dataset": ds, "cohort": c, "level": "recording", "score": m,
                         "n_rec": len(r), "rho": wc.spearman(r[m], r["abs_err"]),
                         "rho_age_strat": wc.age_stratified_spearman(
                             r[m].to_numpy(), r["abs_err"].to_numpy(), r["age"].to_numpy(), 5)})
    return curves, pd.DataFrame(rows)


def cohort_figures(results, fig_dir):
    """Figures 3.2 / 4.3: one panel per dataset x cohort (TUH left, HEEDB
    right; rows normal / abnormal / unlabelled). Window level: E2L solid, the
    benchmark dashed. Recording level: the first run only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import assert_max_columns, panel_labels
    names = list(results)[:2]
    colors = {"A1": "C0", "A2": "C1", "A3": "C2", "u_A1_tta": "C0", "u_A2_down": "C1",
              "u_A3_conc": "C2", "u_sd": "C3", "u_emg": "C4"}
    paths = {}
    for level, keys, xlabel, ylabel, fname in (
            ("window", ("A1", "A2", "A3"), "within-recording weight quintile (1 = lowest weight)",
             "excess window |error| (y)", "window_confidence_weight_rank_cohorts.png"),
            ("record", COHORT_MEASURES, "recording uncertainty quintile (5 = least confident)",
             "recording MAE, mean pooling (y)", "window_confidence_deciles_cohorts.png")):
        fig, axes = plt.subplots(3, 2, figsize=(14, 13.5), sharex=True)
        flat = []
        for ax, (ds, c) in zip(axes.ravel(), COHORTS):
            flat.append(ax)
            first = results[names[0]]["cohort_curves"].get((ds, c))
            if first is None:
                ax.set_title(f"{DATASET_NAME[ds]} {c}: no recordings")
                ax.axis("off")
                continue
            runs = names if level == "window" else names[:1]
            for k in keys:
                for name, ls in zip(runs, ("-", "--")):
                    curve = results[name]["cohort_curves"][(ds, c)][level][k]
                    ax.plot(list(curve), list(curve.values()), ls=ls, marker="o", ms=4,
                            color=colors[k], label=f"{LABELS[k]} ({name})" if level == "window" else LABELS[k])
            if level == "window":
                ax.axhline(0, color="0.6", lw=0.8)
            ax.set_title(f"{DATASET_NAME[ds]} {c}, n = {first['n_rec']} recordings")
            ax.set_ylabel(ylabel)
        for ax in axes[-1]:
            ax.set_xlabel(xlabel)
            ax.set_xticks(range(1, 6))
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        panel_labels([a for a in flat if a.axison])
        assert_max_columns(fig)
        paths[level] = os.path.join(fig_dir, fname)
        fig.savefig(paths[level], dpi=110)
        plt.close(fig)
    return paths


def figures(results, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import panel_labels
    names = list(results)[:2]
    paths = {}

    fig, axes = plt.subplots(1, len(names), figsize=(14, 4.5), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        for method, curve in results[name]["rank_curves"].items():
            ax.plot(list(curve), list(curve.values()), marker="o", label=LABELS[method])
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("within-recording weight decile (1 = lowest weight)")
        ax.set_title(f"{name}, test")
    axes[0][0].set_ylabel("window |error| minus recording mean |error| (y)")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    panel_labels(axes[0])
    paths["weight_rank"] = os.path.join(fig_dir, "window_confidence_weight_rank.png")
    fig.savefig(paths["weight_rank"], dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(14, 4.5), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        for m in ("u_A1_tta", "u_A2_down", "u_A3_conc", "u_emg", "oracle"):
            cov, risk = results[name]["curves"][m]
            ax.plot(cov * 100, risk, label=LABELS[m], ls="--" if m == "oracle" else "-")
        full = results[name]["curves"]["oracle"][1][-1]
        ax.axhline(full, color="0.5", lw=0.8, ls=":", label="random deferral (= all recordings)")
        ax.set_ylim(0, full * 1.25)
        ax.set_xlabel("coverage (% of recordings kept)")
        ax.set_title(f"{name}, test, mean pooling")
        ax.set_xlim(10, 100)
    axes[0][0].set_ylabel("MAE of kept recordings (y)")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    panel_labels(axes[0])
    paths["risk_coverage"] = os.path.join(fig_dir, "window_confidence_risk_coverage.png")
    fig.savefig(paths["risk_coverage"], dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(14, 4.5), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        for m, vals in results[name]["curves"]["_deciles"].items():
            if m in ("u_A1_tta", "u_A2_down", "u_A3_conc", "u_sd", "u_emg"):
                ax.plot(range(1, len(vals) + 1), vals, marker="o", label=LABELS[m])
        ax.set_xlabel("recording uncertainty decile (10 = least confident)")
        ax.set_title(f"{name}, test, mean pooling")
    axes[0][0].set_ylabel("recording MAE (y)")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    panel_labels(axes[0])
    paths["deciles"] = os.path.join(fig_dir, "window_confidence_deciles.png")
    fig.savefig(paths["deciles"], dpi=130)
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fig-dir", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for name in args.run:
        df = pd.read_parquet(os.path.join(args.in_dir, f"{name}_windows.parquet"))
        df = wc.add_tta_stats(df.reset_index(drop=True))
        recs, w, settings = poolings(df)
        mae = mae_table(recs).assign(run=name)
        win, rank_curves = window_tables(df, w)
        conf_t, curves, conf = confidence_tables(df, w, recs)
        tta_sd = float(np.sqrt(df.tta_var).median())
        cohort_curves, cohort_stats = cohort_tables(df, w, recs, conf)
        cohort_stats.to_csv(os.path.join(args.out_dir, f"{name}_cohort_corr.csv"), index=False)
        boot = bootstrap_delta(recs).assign(run=name)
        boot.to_csv(os.path.join(args.out_dir, f"{name}_pooling_delta_ci.csv"), index=False)
        prof = tta_profile(df, curve_split(df))
        prof.to_csv(os.path.join(args.out_dir, f"{name}_tta_profile.csv"))
        by_pos, skips = position_tables(df)
        by_pos.to_csv(os.path.join(args.out_dir, f"{name}_error_by_position.csv"), index=False)
        skips.to_csv(os.path.join(args.out_dir, f"{name}_skip_first.csv"), index=False)
        results[name] = {"settings": settings, "rank_curves": rank_curves, "curves": curves,
                         "cohort_curves": cohort_curves,
                         "mae": mae, "window": win.assign(run=name), "conf": conf_t.assign(run=name),
                         "median_tta_sd": tta_sd, "n_windows": len(df)}
        mae.to_csv(os.path.join(args.out_dir, f"{name}_pooling_mae.csv"), index=False)
        win.to_csv(os.path.join(args.out_dir, f"{name}_window_corr.csv"), index=False)
        conf_t.to_csv(os.path.join(args.out_dir, f"{name}_recording_conf.csv"), index=False)
        conf.to_parquet(os.path.join(args.out_dir, f"{name}_recording_conf.parquet"))
        with pd.option_context("display.width", 200, "display.max_rows", 200,
                               "display.max_columns", 30):
            print(f"\n===== {name}: {len(df)} windows, median TTA SD {tta_sd:.2f} y; settings "
                  f"eps={settings['eps']} delta={settings['delta']}")
            print({k: round(v, 3) for k, v in settings["theta"].items()})
            print(mae[mae.group.isin(["all", "temple", "harvard"])]
                  .pivot_table(index="method", columns=["split", "group"], values="mae").round(3))
            print(boot[boot.group.isin(["all", "temple", "harvard"])].round(3).to_string(index=False))
            print(prof.round(2))
            print(by_pos[by_pos.k < 12].pivot_table(index=["split", "dataset"], columns="k",
                                                     values="abs_err").round(1))
            print(skips.pivot_table(index=["pooling", "skip"], columns=["split", "dataset"],
                                    values="mae").round(3))
            print(win.round(3))
            print(conf_t[conf_t.group.isin(["all", "temple", "harvard"]) & (conf_t.split == curve_split(df))]
                  .round(3).to_string(index=False))
    summary = {n: {"settings": r["settings"], "median_tta_sd": r["median_tta_sd"],
                   "n_windows": r["n_windows"]} for n, r in results.items()}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    if args.fig_dir:
        print(figures(results, args.fig_dir))
        print(cohort_figures(results, args.fig_dir))


if __name__ == "__main__":
    main()
