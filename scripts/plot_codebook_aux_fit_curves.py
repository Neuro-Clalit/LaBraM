"""Train / validation / test MSE over epochs for P4, P4-R and P4-Q (fold 0).

Training error: the fixed sample of 500 training recordings per dataset scored
like validation (no mixup), every 4 epochs and at the last; plus the running
training MAE of each epoch (mixup targets; windows in epochs 0-7, bags after).
MSE = RMSE^2 of the recording-level predictions (years^2).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--p4", required=True, help="log.txt of P4_L1 fold 0")
ap.add_argument("--p4r", required=True, help="log.txt of P4R_L1 fold 0")
ap.add_argument("--p4q", required=True, help="log.txt of P4Q_L1 fold 0")
ap.add_argument("--out", default="docs/data/codebook_aux_p4")
args = ap.parse_args()
LOGS = {"P4": args.p4, "P4-R": args.p4r, "P4-Q": args.p4q}
COLOR = {"P4": "#2a78d6", "P4-R": "#eb6834", "P4-Q": "#1baf7a"}        # categorical slots 1-3
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OUT = args.out

rows = {k: [json.loads(line) for line in open(p) if line.startswith("{")] for k, p in LOGS.items()}
EP = list(range(17))


def series(run, key, square=True):
    xs, ys = [], []
    for r in rows[run]:
        v = r.get(key)
        if v is not None:
            xs.append(r["epoch"]); ys.append(v * v if square else v)
    return xs, ys


plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def phases(ax, label=True):
    """Recessive phase separators: window epochs 0-7, attention-only 8, joint 9-16."""
    for x in (7.5, 8.5):
        ax.axvline(x, color=AXIS, linewidth=1, zorder=0)
    if label:
        y = ax.get_ylim()[1]
        for x, t in ((3.5, "window phase"), (8.0, "pool"), (12.5, "joint bag phase")):
            ax.text(x, y, t, ha="center", va="bottom", fontsize=8, color=MUTED)


def line(ax, x, y, run, style="-", marker=None, label=None):
    ax.plot(x, y, style, color=COLOR[run], marker=marker, markersize=6 if marker else 0,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)


def finish(ax, title, ylabel):
    ax.set_title(title, loc="left", color=INK, pad=16)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("epoch")
    ax.set_xticks(range(0, 17, 2))
    ax.set_xlim(-0.4, 16.4)


legend_runs = [Line2D([], [], color=COLOR[r], linewidth=2, label=r) for r in COLOR]

# ---------------------------------------------------------------- Figure 1
fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
(a, b), (c, d) = axes
for run in COLOR:
    line(a, *series(run, "train_eval_rmse"), run, marker="o")
    line(b, *series(run, "val_rmse"), run)
    line(c, *series(run, "test_rmse"), run)
    xs, tr = series(run, "train_eval_rmse")
    va = [rows[run][e]["val_rmse"] ** 2 for e in xs]
    line(d, xs, [v - t for v, t in zip(va, tr)], run, marker="o")
finish(a, "(a) Training recordings, clean score", "MSE (years²)")
finish(b, "(b) Validation (fold 0)", "MSE (years²)")
finish(c, "(c) Test (7,612 recordings)", "MSE (years²)")
finish(d, "(d) Gap: validation − training MSE", "years²")
lo = min(min(series(r, k)[1]) for r in COLOR for k in ("val_rmse", "test_rmse"))
hi = max(max(series(r, k)[1]) for r in COLOR for k in ("val_rmse", "test_rmse"))
for ax in (b, c):
    ax.set_ylim(lo * 0.95, hi * 1.03)
for ax in axes.flat:
    phases(ax)
fig.legend(handles=legend_runs, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.045))
fig.suptitle("Figure 1. MSE by split over epochs: P4 vs P4-R vs P4-Q", x=0.01, ha="left",
             y=1.085, fontsize=12, fontweight="semibold")
fig.savefig(f"{OUT}/fig1_mse_by_split.png", dpi=200, bbox_inches="tight")

# ---------------------------------------------------------------- Figure 2
fig, axes = plt.subplots(3, 1, figsize=(9, 10.5), sharex=True, sharey=True, constrained_layout=True)
for ax, run in zip(axes, COLOR):
    line(ax, *series(run, "val_rmse"), run, "-")
    line(ax, *series(run, "test_rmse"), run, "--")
    line(ax, *series(run, "train_eval_rmse"), run, ":", marker="o")
    finish(ax, run, "MSE (years²)")
    if ax is not axes[-1]:
        ax.set_xlabel("")
    xs, tr = series(run, "train_eval_rmse")
    ax.annotate(f"gap {rows[run][16]['val_rmse'] ** 2 - tr[-1]:.0f}", xy=(16, (rows[run][16]["val_rmse"] ** 2 + tr[-1]) / 2),
                xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.annotate("", xy=(16.15, tr[-1]), xytext=(16.15, rows[run][16]["val_rmse"] ** 2),
                arrowprops=dict(arrowstyle="<->", color=MUTED, linewidth=1))
for ax in axes:
    phases(ax, label=ax is axes[0])
style_key = [Line2D([], [], color=INK2, linestyle=s, marker=m, markersize=5, markeredgecolor=SURFACE, label=l)
             for s, m, l in (("-", None, "validation"), ("--", None, "test"), (":", "o", "training recordings (clean)"))]
fig.legend(handles=style_key, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
fig.suptitle("Figure 2. Training vs validation vs test MSE, one panel per run", x=0.01, ha="left",
             y=1.06, fontsize=12, fontweight="semibold")
fig.savefig(f"{OUT}/fig2_mse_per_run.png", dpi=200, bbox_inches="tight")

# ---------------------------------------------------------------- Figure 3
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
a, b = axes
for run in COLOR:
    line(a, *series(run, "train_mae", square=False), run)
    line(a, *series(run, "train_eval_mae", square=False), run, ":", marker="o")
    line(b, *series(run, "val_mae", square=False), run)
finish(a, "(a) Training MAE: running (solid) and clean (dotted)", "MAE (years)")
finish(b, "(b) Validation MAE", "MAE (years)")
a.set_ylim(b.get_ylim()[0] * 0 + 3, max(a.get_ylim()[1], b.get_ylim()[1]))
b.set_ylim(a.get_ylim())
for ax in axes:
    phases(ax)
a.text(8.6, 9.6, "running MAE: mixup targets;\nwindows in 0-7, bags from 8", fontsize=8, color=MUTED, va="top")
fig.legend(handles=legend_runs, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.1))
fig.suptitle("Figure 3. MAE every epoch, including the running training error", x=0.01, ha="left",
             y=1.2, fontsize=12, fontweight="semibold")
fig.savefig(f"{OUT}/fig3_mae_running.png", dpi=200, bbox_inches="tight")
print("ok")

# per-epoch table next to the figures
import csv  # noqa: E402
with open(f"{OUT}/per_epoch.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["run", "epoch", "train_running_mae", "train_clean_mse", "train_clean_mae", "val_mse", "val_mae",
                "test_mse", "test_mae", "lemon_mae", "tdbrain_mae"])
    for run, rs in rows.items():
        for r in rs:
            sq = lambda k: None if r.get(k) is None else round(r[k] ** 2, 3)
            rd = lambda k: None if r.get(k) is None else round(r[k], 3)
            w.writerow([run, r["epoch"], rd("train_mae"), sq("train_eval_rmse"), rd("train_eval_mae"), sq("val_rmse"),
                        rd("val_mae"), sq("test_rmse"), rd("test_mae"), rd("test_lemon_mae"), rd("test_tdbrain_mae")])
