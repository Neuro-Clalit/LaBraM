#!/usr/bin/env python
"""Window-level confidence against window-level error for the attention ensemble.

For a recording ``r`` with ``n`` scored windows and the fold models ``k`` of an
ensemble (the ``eval_age_by_cohort.py --save-windows`` parquets: window
prediction ``p_ik``, attention weight ``a_ik``), per window ``i``:

* ``pred``     ensemble window prediction, ``p_i = mean_k p_ik``;
* ``err``      window error, ``e_i = |p_i - age_r|`` (the only use of the age);
* ``att``      relative attention weight, ``w_i = n * mean_k a_ik`` (1 = the
               uniform weight; ``a_ik`` re-normalised to sum to 1 over the windows);
* ``dev``      deviation from the recording's answer, ``d_i = |p_i - yhat_r|``,
               ``yhat_r = mean_k sum_i a_ik p_ik`` (the window's share of the
               ``spread`` score);
* ``fold_sd``  disagreement of the fold models on the window, ``sd_k(p_ik)``.

Reported per cohort (``<out>``):

* windows, recordings, window MAE and recording MAE;
* does attention find the better windows: Spearman rho of ``att`` with ``err``
  over all windows and the median of the within-recording rho (recordings with
  at least ``--min-windows`` windows), the share of recordings where it is
  negative, the attention-weighted window MAE ``mean_r sum_i a_i e_i`` against
  the plain one, and the recording MAE with attention pooling against the plain
  mean of the window predictions;
* Spearman rho of ``dev`` and of ``fold_sd`` with ``err``, pooled and within recording.

Figures (binned means only, no per-subject points; ``--fig-dir``):

* ``window_error_vs_attention.png``  (a) window error by decile of the
  relative attention weight, (b) the same within the recording: the window's
  error minus its recording's mean window error, by the window's attention
  rank in the recording;
* ``window_error_vs_spread.png``     window error (a) by decile of ``dev`` and
  (b) by decile of ``fold_sd``;
* ``recording_error_vs_confidence.png``  recording error by decile of the
  recording-level scores (a) ``spread`` and (b) ``total`` (``rejection_scores.py``).

Bins are deciles of the cohort's own windows (recordings: 10 bins from 1,000
recordings, 5 from 200, else 3); a bin is drawn at the
mean of its x values.

    python scripts/window_confidence_figures.py --set ... --model P4ens \\
        --out docs/data/external_eval/tdbrain/window_confidence.csv --fig-dir docs/figures/tdbrain
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from external_cohort_table import load_meta  # noqa: E402

NAMES = {"temple": "Temple", "harvard": "Harvard", "tdbrain": "TDBRAIN", "lemon": "LEMON", "chbp": "CHBP",
         "mipdb": "MIPDB"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # categorical slots 1-6, fixed order
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"


def load_windows(wdir, members, stems):
    """One row per window of the scored recordings, with the columns of the docstring."""
    per = []
    for m in members:
        w = pd.read_parquet(os.path.join(wdir, f"{m}_test.parquet"), columns=["recording", "slot", "alpha", "window_pred", "age"])
        w = w[w["recording"].isin(stems)]
        w["alpha"] = w["alpha"] / w.groupby("recording", sort=False)["alpha"].transform("sum")
        per.append(w.set_index(["recording", "slot"]).sort_index())
    idx = per[0].index
    if any(not p.index.equals(idx) for p in per[1:]):
        raise SystemExit(f"{wdir}: the fold models do not hold the same windows")
    p = np.stack([f["window_pred"].to_numpy() for f in per])
    a = np.stack([f["alpha"].to_numpy() for f in per])
    d = pd.DataFrame({"pred": p.mean(axis=0), "alpha": a.mean(axis=0), "fold_sd": p.std(axis=0),
                      "age": per[0]["age"].to_numpy()}, index=idx).reset_index()
    g = d.groupby("recording", sort=False)
    d["n"] = g["pred"].transform("size")
    yhat_k = pd.DataFrame((a * p).T, index=idx).groupby(level="recording", sort=False).sum()       # [recordings, folds]
    d["yhat"] = d["recording"].map(yhat_k.mean(axis=1))
    d["att"] = d["alpha"] * d["n"]
    d["err"] = (d["pred"] - d["age"]).abs()
    d["dev"] = (d["pred"] - d["yhat"]).abs()
    return d


def within_rho(d, x, min_windows):
    """Spearman rho of ``x`` with the window error inside each recording with enough windows."""
    big = d[d["n"] >= min_windows]
    rx = big.groupby("recording", sort=False)[x].rank()
    re = big.groupby("recording", sort=False)["err"].rank()
    g = pd.DataFrame({"rx": rx, "re": re}).groupby(big["recording"], sort=False)
    rx, re = rx - g["rx"].transform("mean"), re - g["re"].transform("mean")          # Pearson r of the ranks
    s = pd.DataFrame({"xy": rx * re, "xx": rx ** 2, "yy": re ** 2}).groupby(big["recording"], sort=False).sum()
    return (s["xy"] / np.sqrt(s["xx"] * s["yy"]).where(lambda v: v > 0)).dropna()


def summary(name, d, min_windows):
    g = d.groupby("recording", sort=False)
    rec = g.agg(age=("age", "first"), yhat=("yhat", "first"), mean_pred=("pred", "mean"), mean_err=("err", "mean"))
    rec["att_err"] = (d["alpha"] * d["err"]).groupby(d["recording"], sort=False).sum()
    row = {"cohort": name, "windows": len(d), "recordings": len(rec), "windows_per_recording": len(d) / len(rec),
           "window_mae": d["err"].mean(), "window_mae_attention_weighted": rec["att_err"].mean(),
           "window_mae_unweighted_per_recording": rec["mean_err"].mean(),
           "recording_mae_attention": (rec["yhat"] - rec["age"]).abs().mean(),
           "recording_mae_plain_mean": (rec["mean_pred"] - rec["age"]).abs().mean()}
    for x in ("att", "dev", "fold_sd"):
        row[f"rho_{x}_err"] = stats.spearmanr(d[x], d["err"]).statistic
        w = within_rho(d, x, min_windows)
        row[f"rho_{x}_err_within_median"] = w.median()
        row[f"rho_{x}_err_within_share_negative"] = (w < 0).mean()
        row["recordings_within"] = len(w)
    return row


def decile_curve(x, y, bins=10):
    """(mean x, mean y) per quantile bin of x."""
    q = pd.qcut(x, bins, labels=False, duplicates="drop")
    t = pd.DataFrame({"x": x, "y": y}).groupby(q).mean()
    return t["x"].to_numpy(), t["y"].to_numpy()


def new_axes(xlabel, ylabel, titles):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, title, xl, yl in zip(axes, titles, xlabel, ylabel):
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return fig, axes


def finish(fig, axes, path):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=170)
    print(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True, metavar="NAME:COHORT:EVAL.json:WINDOWS_DIR[:FAMILY]")
    ap.add_argument("--model", default="P4ens")
    ap.add_argument("--min-age", type=float, default=18.0)
    ap.add_argument("--min-windows", type=int, default=8, help="recordings with fewer windows are left out of the within-recording statistics")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig-dir", default="")
    args = ap.parse_args()
    import rejection_scores as rs
    rows, win, rec = [], {}, {}
    for spec in args.set:
        name, cdir, ev, wdir, *family = spec.split(":")
        meta = load_meta(cdir, family[0] if family else None, args.min_age)
        members = json.load(open(ev))[args.model].get("members", [args.model])
        d = load_windows(wdir, members, set(meta.index))
        rows.append(summary(name, d, args.min_windows))
        win[name] = d
        rec[name] = rs.cohort_units(meta, wdir, members)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 250):
        print(df.set_index("cohort").T.round(3).to_string())
    if not args.fig_dir:
        return
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(args.fig_dir, exist_ok=True)
    color = {n: COLORS[i] for i, n in enumerate(NAMES)}                 # the colour follows the cohort
    line = dict(lw=2, marker="o", ms=5, zorder=3)

    fig, axes = new_axes(["relative attention weight (1 = uniform)", "attention rank of the window in its recording"],
                         ["window error (years)", "window error minus recording mean (years)"],
                         ["(a) window error by attention weight", "(b) the same within the recording"])
    for name, d in win.items():
        x, y = decile_curve(d["att"], d["err"])
        axes[0].plot(x, y, color=color[name], label=NAMES[name], **line)
        big = d[d["n"] >= args.min_windows]
        g = big.groupby("recording", sort=False)
        rank = (g["att"].rank(method="first") - 0.5) / big["n"]
        excess = big["err"] - g["err"].transform("mean")
        x, y = decile_curve(rank, excess)
        axes[1].plot(x, y, color=color[name], label=NAMES[name], **line)
    axes[0].axvline(1, color=MUTED, lw=1, ls=":", zorder=1)
    axes[1].axhline(0, color=MUTED, lw=1, ls=":", zorder=1)
    axes[1].set_xticks([0, 0.5, 1])
    axes[1].set_xticklabels(["lowest", "median", "highest"])
    finish(fig, axes, os.path.join(args.fig_dir, "window_error_vs_attention.png"))

    fig, axes = new_axes(["deviation of the window from the recording's prediction (years)",
                          "disagreement of the fold models on the window (years)"],
                         ["window error (years)"] * 2,
                         ["(a) window error by deviation from the recording", "(b) window error by fold disagreement"])
    for name, d in win.items():
        for ax, x in zip(axes, ("dev", "fold_sd")):
            ax.plot(*decile_curve(d[x], d["err"]), color=color[name], label=NAMES[name], **line)
    finish(fig, axes, os.path.join(args.fig_dir, "window_error_vs_spread.png"))

    fig, axes = new_axes(["spread of the window predictions (years)", "total uncertainty (years)"],
                         ["recording error (years)"] * 2,
                         ["(a) recording error by spread", "(b) recording error by total uncertainty"])
    for name, d in rec.items():
        bins = 10 if len(d) >= 1000 else 5 if len(d) >= 200 else 3    # at least about 10 recordings per bin
        for ax, x in zip(axes, ("spread", "total")):
            ax.plot(*decile_curve(d[x], d["abs_err"], bins), color=color[name], label=NAMES[name], **line)
    finish(fig, axes, os.path.join(args.fig_dir, "recording_error_vs_confidence.png"))


if __name__ == "__main__":
    main()
