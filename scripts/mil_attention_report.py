#!/usr/bin/env python
"""Attention weight vs. per-window error of end-to-end MIL runs.

Reads ``analysis/{split}_bag_weights_epoch{E}.parquet`` written by
``labram/train/train_mil.py::evaluate_bags`` (one row per window: recording,
slot, minute, alpha, window_pred through the run's head, age) and reports:

* window |error| by within-recording attention decile, and its excess over
  the recording's mean window |error|;
* window |error| by relative weight ``r = alpha * n`` (1 = uniform weight);
* the within-recording Spearman of alpha with the window |error|, overall and
  per dataset x cohort;
* the mean relative weight by minute of the recording.

    python scripts/mil_attention_report.py \\
        --run L1=/data/runs/heedb/mil_local/L1_abmil_.../analysis \\
        --run W2=/data/runs/heedb/mil_sagemaker/artifacts/w2/finetune/analysis \\
        --data-path /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1 --fig-dir docs/figures
"""
import argparse
import os

import numpy as np
import pandas as pd

from labram.eval import window_confidence as wc

R_BINS = [0, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0, np.inf]
R_LABELS = ["<0.5", "0.5–0.8", "0.8–1.0", "1.0–1.25", "1.25–1.5", "1.5–2", ">2"]


def load(path, split, epoch, cohorts):
    w = pd.read_parquet(os.path.join(path, f"{split}_bag_weights_epoch{epoch:02d}.parquet"))
    w["split"] = split
    n = w.groupby("recording")["alpha"].transform("size")
    w["r"] = w["alpha"] * n
    w["abs_err"] = (w["window_pred"] - w["age"]).abs()
    rank = w.groupby("recording")["alpha"].rank(pct=True, method="average")
    w["decile"] = np.ceil(rank * 10).clip(1, 10).astype(int)
    w["excess"] = w["abs_err"] - w.groupby("recording")["abs_err"].transform("mean")
    w = w.merge(cohorts, on="recording", how="left")
    return w


def cohort_table(data_path):
    """recording -> dataset family (temple / harvard) and report cohort."""
    import json
    from labram.data.dataset_groups import dataset_family
    from labram.data.tuh_metadata import load_label_lookup_for
    m = json.load(open(os.path.join(data_path, "manifest.json")))["recordings"]
    labels = load_label_lookup_for(data_path)
    return pd.DataFrame([{"recording": r["stem"], "dataset": dataset_family(r.get("corpus")),
                          "cohort": labels.get(r["stem"]) if labels.get(r["stem"]) in ("normal", "abnormal")
                          else "unlabelled"} for r in m])


def summarize(name, w):
    out = {"run": name}
    c = wc.weight_error_correlation(w.rename(columns={"alpha": "w"}), "w", "abs_err")
    out.update(rho_within=c["rho_within"], share_neg=c["frac_neg"])
    out["window_mae"] = w["abs_err"].mean()
    out["max_r_mean"] = w.groupby("recording")["r"].max().mean()
    out["share_r_below_0.5"] = (w["r"] < 0.5).mean()
    out["share_r_above_1.5"] = (w["r"] > 1.5).mean()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, metavar="NAME=ANALYSIS_DIR")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--epoch", type=int, default=3)
    ap.add_argument("--fig-dir", default=None)
    ap.add_argument("--csv-dir", default=None)
    args = ap.parse_args()
    cohorts = cohort_table(args.data_path)
    runs = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        runs[name] = load(path, args.split, args.epoch, cohorts)

    pd.set_option("display.width", 220)
    summ = pd.DataFrame([summarize(n, w) for n, w in runs.items()])
    print("== summary (%s)" % args.split); print(summ.round(3).to_string(index=False))
    dec = pd.DataFrame({n: w.groupby("decile")["abs_err"].mean() for n, w in runs.items()})
    exc = pd.DataFrame({n: w.groupby("decile")["excess"].mean() for n, w in runs.items()})
    print("== window |error| by attention decile (1 = lowest weight)"); print(dec.round(2).T.to_string())
    print("== excess over recording mean"); print(exc.round(2).T.to_string())
    rb = {}
    for n, w in runs.items():
        b = pd.cut(w["r"], R_BINS, labels=R_LABELS, right=False)
        g = w.groupby(b, observed=False)
        rb[n] = pd.DataFrame({"share": g.size() / len(w), "window_mae": g["abs_err"].mean(),
                              "excess": g["excess"].mean()})
        print(f"== {n}: by relative weight r = alpha * n"); print(rb[n].round(3).T.to_string())
    coh = []
    for n, w in runs.items():
        for (ds, c), g in w.groupby(["dataset", "cohort"]):
            if g["recording"].nunique() < 20:
                continue
            cc = wc.weight_error_correlation(g.rename(columns={"alpha": "w"}), "w", "abs_err")
            low = g[g.decile <= 2]["abs_err"].mean()
            high = g[g.decile >= 9]["abs_err"].mean()
            coh.append({"run": n, "group": f"{ds}_{c}", "recordings": g["recording"].nunique(),
                        "rho_within": cc["rho_within"], "mae_low20": low, "mae_high20": high,
                        "window_mae": g["abs_err"].mean()})
    coh = pd.DataFrame(coh)
    print("== per cohort"); print(coh.round(3).to_string(index=False))
    minute = pd.DataFrame({n: w.groupby(np.floor(w["minute"]).astype(int))["r"].mean() for n, w in runs.items()})
    print("== mean relative weight by minute"); print(minute.round(3).T.to_string())
    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
        summ.to_csv(os.path.join(args.csv_dir, "summary.csv"), index=False)
        dec.to_csv(os.path.join(args.csv_dir, "window_mae_by_decile.csv"))
        coh.to_csv(os.path.join(args.csv_dir, "per_cohort.csv"), index=False)
        pd.concat(rb, names=["run"]).to_csv(os.path.join(args.csv_dir, "by_relative_weight.csv"))
    if args.fig_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from labram.eval.age_plots import panel_labels
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        ax = axes[0][0]
        for n in runs:
            ax.plot(dec.index, dec[n], "-o", ms=4, label=n)
        ax.set_xlabel("within-recording attention decile (1 = lowest weight)")
        ax.set_ylabel("window |error| (y)")
        ax.set_title("window MAE by attention decile")
        ax.legend(fontsize=8)
        ax = axes[0][1]
        for n in runs:
            ax.plot(minute.index, minute[n], "-o", ms=4, label=n)
        ax.axhline(1, color="0.6", lw=0.8)
        ax.set_xlabel("minute within the evaluated 5 min")
        ax.set_ylabel("mean relative weight (1 = uniform)")
        ax.set_title("where the attention goes")
        name0 = list(runs)[0]
        w0 = runs[name0]
        ax = axes[1][0]
        hb = ax.hexbin(np.clip(w0["r"], 0, 3), np.clip(w0["abs_err"], 0, 40), gridsize=50, bins="log",
                       cmap="viridis", mincnt=1)
        ax.plot([(R_BINS[i] + min(R_BINS[i + 1], 3)) / 2 for i in range(len(R_LABELS))],
                rb[name0]["window_mae"].to_numpy(), "r-o", ms=4, label="mean")
        ax.set_xlabel("relative attention weight r (clipped at 3)")
        ax.set_ylabel("window |error| (y, clipped at 40)")
        ax.set_title(f"{name0}: every test window")
        ax.legend(fontsize=8)
        ax = axes[1][1]
        groups = sorted(coh["group"].unique())
        x = np.arange(len(groups))
        width = 0.8 / max(len(runs), 1)
        for i, n in enumerate(runs):
            v = coh[coh.run == n].set_index("group").reindex(groups)["rho_within"]
            ax.bar(x + i * width, v, width, label=n)
        ax.set_xticks(x + width * (len(runs) - 1) / 2)
        ax.set_xticklabels([g.replace("temple", "TUH").replace("harvard", "HEEDB").replace("_", "\n") for g in groups],
                           fontsize=8)
        ax.set_ylabel("ρ within recording (weight vs. |error|)")
        ax.set_title("per dataset and cohort")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.legend(fontsize=8)
        fig.tight_layout()
        panel_labels([axes[0][0], axes[0][1], axes[1][0], axes[1][1]])
        path = os.path.join(args.fig_dir, f"mil_attention_vs_error_{args.split}.png")
        fig.savefig(path, dpi=120)
        print(path)


if __name__ == "__main__":
    main()
