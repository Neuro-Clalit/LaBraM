#!/usr/bin/env python
"""Explanatory figures of the technical report (``docs/tech_report/``).

* ``fig_pipeline.png``: (a) the LaBraM pre-training that is reused unchanged
  (neural tokenizer, masked EEG modelling) and (b) the recording-level
  brain-age pipeline of this work built on its encoder (Sections 4.1-4.5).
* ``fig_attention_example.png``: attention pooling over the windows of one
  recording of an open cohort (LEMON, CC BY), P4 ensemble: window predictions
  and relative attention weights over time, weight against the deviation from
  the pooled prediction, and the EEG of the highest- and the lowest-weighted
  window. Ensemble window quantities as Eq. (15) of the report: ``p_i`` the
  fold mean of the window predictions, ``w_i = (n/K) sum_k alpha_ik``; the
  pooled prediction is the fold mean of ``sum_i alpha_ik p_ik``.

    python scripts/tech_report_figures.py --out-dir docs/tech_report \\
        --windows /data/runs/heedb/external_eval/windows/lemon_all_models \\
        --cohort /data/datasets/EEG-public/LEMON/cohorts/lemon_v1 --recording sub-032490_lemon-rs --trim-windows 6
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"
SPACING = 80                                                                 # uV between EEG traces
CLIP = SPACING / 2                                                           # uV, display clip: traces never overlap
STYLE = {  # fill, edge
    "data": ("#efeeea", "#6b6a64"),
    "labram": ("#dbe8f7", "#2a78d6"),
    "ours": ("#fbe3d6", "#eb6834"),
    "note": ("#ffffff", "#6b6a64"),
}


def box(ax, x, y, w, h, title, body, kind):
    from matplotlib.patches import FancyBboxPatch
    fill, edge = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fill, ec=edge,
                                lw=1.6, ls="--" if kind == "note" else "-", zorder=2))
    ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=9.4, fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + h - 4.4, body, ha="center", va="top", fontsize=8.4, color=INK, linespacing=1.3, zorder=3)


def arrow(ax, x0, y0, x1, y1, label=None, color=INK, rad=0.0):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, lw=1.4, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        ax.text((x0 + x1) / 2 + 0.8, (y0 + y1) / 2, label, fontsize=8, color=color, va="center", ha="left",
                style="italic")


def pipeline(path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 7.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")
    W, H, X = 16.4, 13.0, [1.5, 21.6, 41.7, 61.8, 81.9]
    ya, yb, yc = 46.5, 24.5, 3.0
    ax.text(0.5, 64.5, "(a) LaBraM pre-training (Jiang et al. 2024): reused, not retrained", fontsize=11,
            fontweight="bold", color=INK, va="top")
    a = [("unlabelled EEG", "2,535 h from 16 sources\n(BCI, emotion, TUH subsets,\nSiena, own recordings)", "data"),
         ("1 s patches", "every channel cut into\n200-sample patches,\nplus channel and time\nembeddings", "labram"),
         ("neural tokenizer", "VQ codebook of 8,192 codes,\nlearned by reconstructing\neach patch's Fourier\namplitude and phase", "labram"),
         ("masked EEG modelling", "50 % of the patches masked;\nthe encoder predicts their\ncodes (as BEiT v2)", "labram"),
         ("pre-trained encoder $f_\\theta$", "12 transformer blocks,\nwidth 200, 10 heads,\n5.8 M parameters", "labram")]
    for x, (t, b, k) in zip(X, a):
        box(ax, x, ya, W, H, t, b, k)
    for x0, x1 in zip(X[:-1], X[1:]):
        arrow(ax, x0 + W + 0.4, ya + H / 2, x1 - 0.4, ya + H / 2)

    ax.text(0.5, 42.0, "(b) This work: recording-level brain age (Sections 4.1–4.5)", fontsize=11,
            fontweight="bold", color=INK, va="top")
    b = [("routine EEG recording", "Temple + Harvard: 103,117\nrecordings of 71,656\npatients (train / val / test\nsplit by patient)", "data"),
         ("one preprocessing", "0.1–75 Hz band-pass,\nline removal at 50 and\n60 Hz, 200 Hz, µV;\nsame for every test cohort", "ours"),
         ("10 s windows", "training: a bag of K = 8\nrandom windows of one\nrecording; test: all\nscored windows", "ours"),
         ("common average ref.", "19 channels of the 10–20\nsystem; missing ones left\nout of the token sequence\n(LaBraM++ input)", "ours"),
         ("fine-tuned encoder $f_\\theta$", "one embedding per window\n$h_i \\in \\mathbb{R}^{200}$ and its\nown prediction $p_i = g(h_i)$", "labram")]
    for x, (t, b_, k) in zip(X, b):
        box(ax, x, yb, W, H, t, b_, k)
    for x0, x1 in zip(X[:-1], X[1:]):
        arrow(ax, x0 + W + 0.4, yb + H / 2, x1 - 0.4, yb + H / 2)
    arrow(ax, X[4] + W / 2, ya - 0.4, X[4] + W / 2, yb + H + 0.4, label="initial weights", color="#2a78d6")

    c = [("training", "17 epochs, one run: window\nphase → attention only →\njoint; Huber loss on the\nz-scored age, Eq. (6)", "note"),
         ("age and confidence", "ensemble age $\\hat y$, gap\n$\\hat y - y$; 20 % least\nconfident set aside by\n$u_r$, Eq. (9)", "ours"),
         ("5-fold ensemble", "mean of 5 fold models,\nEq. (7); their disagreement\n$d_r$ and the window spread\n$s_r$ measure confidence", "ours"),
         ("fold model $k$", "one model per fold of the\npatient-grouped 5-fold CV,\nbest epoch on its own\nvalidation fold", "ours"),
         ("attention pooling", "$\\alpha_i = \\mathrm{softmax}(w^\\top \\tanh(V h_i))$\n$\\hat y_k = g(\\sum_i \\alpha_i h_i)$, Eq. (5)\none prediction per\nrecording", "ours")]
    for x, (t, b_, k) in zip(X, c):
        box(ax, x, yc, W, H, t, b_, k)
    arrow(ax, X[4] + W / 2, yb - 0.4, X[4] + W / 2, yc + H + 0.4)
    for x0, x1 in zip(X[4:1:-1], X[3:0:-1]):
        arrow(ax, x0 - 0.4, yc + H / 2, x1 + W + 0.4, yc + H / 2)
    # legend
    from matplotlib.patches import Patch
    handles = [Patch(fc=STYLE[k][0], ec=STYLE[k][1], label=lab) for k, lab in
               (("data", "data"), ("labram", "LaBraM (architecture / weights)"), ("ours", "this work"),
                ("note", "training schedule"))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.005), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(path)


def attention_example(path, wdir, cohort, stem, trim, model_glob="P4_L1_f*_test.parquet"):
    import matplotlib.pyplot as plt
    from labram.data.tuh_metadata import TUH_CHANNELS
    from labram.eval.age_plots import assert_max_columns, panel_labels
    files = sorted(glob.glob(os.path.join(wdir, model_glob)))
    d = pd.concat([pd.read_parquet(f).assign(fold=k) for k, f in enumerate(files)])
    d = d[d.recording == stem].copy()
    if d.empty:
        raise SystemExit(f"{stem} not in {wdir}")
    d["a"] = d.alpha / d.groupby("fold").alpha.transform("sum")
    pooled = (d.a * d.window_pred).groupby(d.fold).sum().mean()             # Eq. (7) of the fold models' Eq. (5)
    w = d.groupby("slot").agg(p=("window_pred", "mean"), a=("a", "mean"), minute=("minute", "first")).reset_index()
    n, K = len(w), len(files)
    w["wrel"] = w.a * n                                                      # Eq. (15): 1 = uniform weight
    w["dev"] = (w.p - pooled).abs()                                          # Eq. (16)
    age = float(d.age.iloc[0])
    plain = w.p.mean()
    meta = {r["stem"]: r for r in json.load(open(os.path.join(cohort, "manifest.json")))["recordings"]}[stem]
    x = np.load(os.path.join(cohort, "recordings", f"{stem}.npy"), mmap_mode="r")
    mask = np.asarray(meta["channel_mask"], bool)
    hi, lo = int(w.loc[w.wrel.idxmax(), "slot"]), int(w.loc[w.wrel.idxmin(), "slot"])

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "legend.fontsize": 12})
    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.9, 1.25])
    ax_a = fig.add_subplot(gs[0, :])
    sc = ax_a.scatter(w.minute + trim / 6, w.p, c=w.wrel, s=18 + 40 * w.wrel, cmap="viridis", zorder=3,
                      edgecolor="white", linewidth=0.4)
    ax_a.axhline(age, color=INK, ls=":", lw=1.6, label=f"true age (bin midpoint) {age:.1f}")
    ax_a.axhline(pooled, color="#eb6834", lw=2.2, label=f"attention-pooled prediction {pooled:.1f}")
    ax_a.axhline(plain, color=MUTED, lw=1.6, ls="--", label=f"plain mean of the windows {plain:.1f}")
    ax_a.set_xlabel("time in the recording (min)")
    ax_a.set_ylabel("window prediction $p_i$ (y)")
    ax_a.set_title(f"window predictions of one LEMON recording, P4 ensemble ({n} windows of 10 s)", fontsize=14)
    ax_a.set_ylim(w.p.min() - 6, w.p.max() + 1.5)                         # room for the legend below the points
    ax_a.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=12, loc="lower left", ncol=3)
    cb = fig.colorbar(sc, ax=ax_a, pad=0.01)
    cb.set_label("relative attention weight $w_i$")

    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.bar(w.minute + trim / 6, w.wrel, width=1 / 6 * 0.9, color="#2a78d6", zorder=2)
    ax_b.axhline(1.0, color=MUTED, ls="--", lw=1.2, label="uniform weight")
    ax_b.set_xlabel("time in the recording (min)")
    ax_b.set_ylabel("relative attention weight $w_i$")
    ax_b.set_title("attention weight per window", fontsize=14)
    ax_b.legend(frameon=False, fontsize=12)

    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.scatter(w.dev, w.wrel, s=16, color="#1baf7a", zorder=3)
    rho = w[["dev", "wrel"]].corr("spearman").iloc[0, 1]
    ax_c.set_xlabel("deviation from the pooled prediction $|p_i - \\hat y|$ (y)")
    ax_c.set_ylabel("relative attention weight $w_i$")
    ax_c.set_title(f"weight against deviation (Spearman {rho:+.2f})", fontsize=14)

    names = list(TUH_CHANNELS)
    show = [c for c in ("FP1", "F3", "C3", "P3", "O1", "FP2", "F4", "C4", "P4", "O2", "T3", "T4")
            if c in names and mask[names.index(c)]]
    axes_eeg = []
    for j, (slot, label) in enumerate(((hi, "highest"), (lo, "lowest"))):
        ax = fig.add_subplot(gs[2, j])
        k = slot + trim                                                      # scored slot -> window of the file
        seg = np.asarray(x[k * 2000:(k + 1) * 2000], dtype=np.float32)[:, mask]
        seg = seg - seg.mean(axis=1, keepdims=True)                          # common average reference, as the input
        present = [c for c, m in zip(names, mask) if m]
        t = np.arange(len(seg)) / 200.0
        peak = float(np.abs(seg).max())
        for i, ch in enumerate(show):                                        # display clipped at +-CLIP uV
            ax.plot(t, np.clip(seg[:, present.index(ch)], -CLIP, CLIP) - i * SPACING, color=INK, lw=0.6)
        ax.set_yticks([-i * SPACING for i in range(len(show))], [c.capitalize() for c in show], fontsize=11)
        ax.set_xlabel("time in the window (s)")
        r = w[w.slot == slot].iloc[0]
        note = f", peak {peak:,.0f} µV" if peak > 4 * CLIP else ""
        ax.set_title(f"{label} attention weight, minute {r.minute + trim / 6:.1f}\n$w_i$ = {r.wrel:.2f}, $p_i$ = {r.p:.1f} y{note}",
                     fontsize=13)
        ax.plot([9.6, 9.6], [-len(show) * SPACING + 15, -len(show) * SPACING + 65], color="#eb6834", lw=2)
        ax.text(9.5, -len(show) * SPACING + 40, "50 µV", ha="right", va="center", fontsize=11, color="#eb6834")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        axes_eeg.append(ax)
    for ax in (ax_a, ax_b, ax_c):
        ax.grid(color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    panel_labels([ax_a, ax_b, ax_c, *axes_eeg])
    assert_max_columns(fig)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path, f"n={n} pooled={pooled:.2f} plain={plain:.2f} age={age} rho={rho:.2f} hi={hi} lo={lo}")


def attention_heatmaps(path, wdir, cohort, cases, trim, model_glob="P4_L1_f*_test.parquet"):
    """Attention over whole recordings: per case, (top) relative attention weight of every
    window for each fold model and their mean, (middle) the window predictions with the true
    age and the pooled prediction, (bottom) the window's median channel SD after CAR."""
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import assert_max_columns, panel_labels
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
    files = sorted(glob.glob(os.path.join(wdir, model_glob)))
    d = pd.concat([pd.read_parquet(f).assign(fold=k) for k, f in enumerate(files)])
    meta = {r["stem"]: r for r in json.load(open(os.path.join(cohort, "manifest.json")))["recordings"]}
    eeg19 = [i for i in range(23) if i not in (16, 17, 21, 22)]
    fig = plt.figure(figsize=(10.5, 4.0 * len(cases)))
    outer = fig.add_gridspec(len(cases), 1, hspace=0.42)
    tops, im = [], None
    for c, (stem, label) in enumerate(cases):
        r = d[d.recording == stem].copy()
        r["a"] = r.alpha / r.groupby("fold").alpha.transform("sum")
        n = r[r.fold == 0].shape[0]
        r["w"] = r.a * n                                                     # Eq. (15): 1 = uniform weight
        pooled = (r.a * r.window_pred).groupby(r.fold).sum().mean()          # Eq. (7)
        W = r.pivot(index="fold", columns="slot", values="w")
        P = r.pivot(index="fold", columns="slot", values="window_pred").mean(axis=0)
        mat = np.vstack([W.to_numpy(), W.mean(axis=0).to_numpy()])
        minutes = (W.columns.to_numpy() + trim) / 6.0
        age = float(r.age.iloc[0])
        x = np.load(os.path.join(cohort, "recordings", f"{stem}.npy"), mmap_mode="r")
        mask = np.asarray(meta[stem]["channel_mask"], bool)
        keep = [i for i in eeg19 if mask[i]]
        sd = []
        for slot in W.columns:
            k = int(slot) + trim
            seg = np.asarray(x[k * 2000:(k + 1) * 2000], dtype=np.float32)[:, keep]
            seg = seg - seg.mean(axis=1, keepdims=True)
            sd.append(float(np.median(seg.std(axis=0))))
        inner = outer[c].subgridspec(2, 1, height_ratios=[1.2, 1.4], hspace=0.08)
        ax0 = fig.add_subplot(inner[0])
        ax1 = fig.add_subplot(inner[1], sharex=ax0)
        step = 1 / 6.0
        im = ax0.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=2.5, interpolation="nearest",
                        extent=(minutes[0] - step / 2, minutes[-1] + step / 2, len(mat) - 0.5, -0.5))
        ax0.set_yticks(range(len(mat)), [f"fold {k}" for k in range(len(W))] + ["mean"], fontsize=8.5)
        ax0.axhline(len(W) - 0.5, color="white", lw=1.5)
        ax0.set_title(f"{label}: true age {age:.1f}, predicted {pooled:.1f}, error {pooled - age:+.1f} y")
        ax1.scatter(minutes, P, c=mat[-1], cmap="viridis", vmin=0, vmax=2.5, s=26, edgecolor="white", lw=0.3, zorder=3)
        ax1.axhline(age, color=INK, ls=":", lw=1.6, label="true age")
        ax1.axhline(pooled, color="#eb6834", lw=2, label="attention-pooled prediction")
        ax1.axhline(P.mean(), color=MUTED, ls="--", lw=1.4, label="plain mean of the windows")
        ax1.set_ylabel("window prediction (y)")
        ax1.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
        art = np.asarray(sd) > 100                                           # artifact windows: channel SD > 100 uV
        if art.any():
            ax0.scatter(minutes[art], np.full(art.sum(), 0.93), marker="v", s=40, color="#e3242b", zorder=5,
                        edgecolor="white", linewidth=0.6, transform=ax0.get_xaxis_transform())
        ax1.set_xlabel("time in the recording (min)" if c == len(cases) - 1 else "")
        plt.setp(ax0.get_xticklabels(), visible=False)
        for ax in (ax1,):
            ax.grid(color=GRID, lw=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        tops.append(ax0)
    cb = fig.colorbar(im, ax=fig.axes, fraction=0.02, pad=0.01)
    cb.set_label("relative attention weight $w_i$ (1 = uniform)")
    panel_labels(tops)
    assert_max_columns(fig)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(path)


def attention_on_eeg(path, wdir, cohort, cases, trim, model_glob="P4_L1_f*_test.parquet",
                     full_channels=("FP1", "C3", "O1", "FP2", "C4", "O2"),
                     zoom_channels=("FP1", "F3", "C3", "O1", "FP2", "F4", "C4", "O2"), zoom_windows=9):
    """The EEG itself with every 10 s window's background coloured by its relative attention
    weight (mean over the fold models, 1 = uniform): per case, the whole scored recording and a
    zoom of ``zoom_windows`` windows centred on the lowest-weighted one, with each window's
    predicted age and weight written above it. Display only: common average reference over the
    present 10-20 channels and a 0.5 Hz zero-phase high-pass so slow drifts do not hide the traces."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from scipy.signal import butter, sosfiltfilt
    from labram.data.tuh_metadata import TUH_CHANNELS
    from labram.eval.age_plots import assert_max_columns, panel_labels
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5})
    files = sorted(glob.glob(os.path.join(wdir, model_glob)))
    d = pd.concat([pd.read_parquet(f).assign(fold=k) for k, f in enumerate(files)])
    meta = {r["stem"]: r for r in json.load(open(os.path.join(cohort, "manifest.json")))["recordings"]}
    names = list(TUH_CHANNELS)
    cmap, norm = plt.get_cmap("RdYlGn"), TwoSlopeNorm(vmin=0.3, vcenter=1.0, vmax=2.0)
    sos = butter(2, 0.5, btype="highpass", fs=200, output="sos")
    fig = plt.figure(figsize=(11, 5.4 * len(cases)))
    outer = fig.add_gridspec(len(cases), 1, hspace=0.3)
    axes = []
    for c, (stem, label) in enumerate(cases):
        r = d[d.recording == stem].copy()
        r["a"] = r.alpha / r.groupby("fold").alpha.transform("sum")
        n = r[r.fold == 0].shape[0]
        w = (r.a * n).groupby(r.slot).mean()                                  # Eq. (15)
        p = r.groupby("slot").window_pred.mean()
        pooled = (r.a * r.window_pred).groupby(r.fold).sum().mean()          # Eq. (7)
        age = float(r.age.iloc[0])
        mask = np.asarray(meta[stem]["channel_mask"], bool)
        x = np.load(os.path.join(cohort, "recordings", f"{stem}.npy"), mmap_mode="r")
        present = [i for i in range(23) if mask[i] and i not in (16, 17, 21, 22)]
        slots = w.index.to_numpy()
        seg = np.asarray(x[(slots[0] + trim) * 2000:(slots[-1] + trim + 1) * 2000], dtype=np.float64)[:, present]
        seg = seg - seg.mean(axis=1, keepdims=True)                          # common average reference
        seg = sosfiltfilt(sos, seg, axis=0)                                  # display-only high-pass
        t0 = (slots[0] + trim) * 10.0                                         # seconds from the file start
        lo = int(w.idxmin())
        zs = max(slots[0], min(lo - zoom_windows // 2, slots[-1] - zoom_windows + 1))
        inner = outer[c].subgridspec(2, 1, height_ratios=[0.8, 1.2], hspace=0.55)
        for row, (s0, s1, chans, spacing, step) in enumerate(
                ((slots[0], slots[-1] + 1, full_channels, 90.0, 4), (zs, zs + zoom_windows, zoom_channels, 70.0, 1))):
            ax = fig.add_subplot(inner[row])
            scale = (lambda v: v / 60) if row == 0 else (lambda v: v)
            for sl in range(s0, s1):
                ax.axvspan(scale(t0 + (sl - slots[0]) * 10), scale(t0 + (sl - slots[0] + 1) * 10),
                           color=cmap(norm(w[sl])), alpha=0.6, lw=0, zorder=0)
            show = [ch for ch in chans if ch in names and names.index(ch) in present]
            a, b = (s0 - slots[0]) * 2000, (s1 - slots[0]) * 2000
            tt = scale(t0 + np.arange(a, b) / 200.0)
            clip = 0.55 * spacing
            for k, ch in enumerate(show):
                v = seg[a:b:step, present.index(names.index(ch))]
                ax.plot(tt[::step], np.clip(v, -clip, clip) - k * spacing, color="#1f1f1e",
                        lw=0.3 if row == 0 else 0.55, rasterized=True, zorder=2)
            ax.set_yticks([-k * spacing for k in range(len(show))], [ch.capitalize() for ch in show], fontsize=8.5)
            top = 0.7 * spacing if row == 0 else 1.5 * spacing
            ax.set_ylim(-(len(show) - 0.35) * spacing, top)
            ax.set_xlim(tt[0], tt[-1])
            if row == 0:
                ax.set_xlabel("time in the recording (min)")
                ax.axvspan(scale(t0 + (zs - slots[0]) * 10), scale(t0 + (zs - slots[0] + zoom_windows) * 10),
                           fill=False, ec="#1f1f1e", lw=1.6, zorder=3)
                ax.set_title(f"{label}: true age {age:.1f}, predicted {pooled:.1f} (error {pooled - age:+.1f} y)")
            else:
                ax.set_xlabel("time in the recording (s)")
                for sl in range(s0, s1):
                    ax.text(t0 + (sl - slots[0]) * 10 + 5, 0.95 * spacing, f"{p[sl]:.0f} y · w {w[sl]:.2f}",
                            ha="center", va="center", fontsize=8, color="#1f1f1e", zorder=4)
                ax.set_title("zoom on the box: window prediction and attention weight above each 10 s window")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            axes.append(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("relative attention weight $w_i$ (1 = uniform)")
    panel_labels(axes)
    assert_max_columns(fig)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(path)

PHASES = {  # recipe -> [(first epoch, last epoch, phase)], from the runs' mil.window_epochs and schedules
    "P1": [(0, 9, "window"), (10, 11, "attention only"), (12, 15, "joint")],
    "P3": [(0, 9, "window"), (10, 10, "attention only"), (11, 16, "joint")],
    "P4": [(0, 7, "window"), (8, 8, "attention only"), (9, 16, "joint")],
}
PHASE_COLORS = {"window": "#e3edf9", "attention only": "#fbecc8", "joint": "#e9def3"}


RECIPE_COLORS = {"P1": "#6250d6", "P3": "#008080", "P4": "#d9480f"}   # distinct from the split colours
RECIPE_PHASE_STARTS = {r: [e0 for e0, _, ph in v if ph != "window"] for r, v in PHASES.items()}


def welch_p(m1, s1, n1, m2, s2, n2):
    """Two-sided Welch t-test from summary statistics (mean, sample SD, n)."""
    from scipy import stats
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    t = (m1 - m2) / np.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    return 2 * stats.t.sf(abs(t), df)


def recipe_panel(ax, d, split, group, title, sig_pairs=(("P4", "P3"), ("P4", "P1"))):
    """The three recipes on one axis (fold mean +- SD), attention-phase starts as dotted lines
    in the recipe colour, and per-epoch Welch tests of each pair marked under the curves."""
    for recipe, color in RECIPE_COLORS.items():
        g = d[(d.recipe == recipe) & (d.split == split) & (d.group == group)].sort_values("epoch")
        ax.errorbar(g.epoch, g["mean"], yerr=g["sd"], color=color, marker="o", ms=3.8, lw=1.6, capsize=2.5,
                    label=recipe, zorder=3)
        for e in RECIPE_PHASE_STARTS[recipe][:1]:
            ax.axvline(e - 0.5, color=color, ls=":", lw=1.3, zorder=1)
    shown = d[(d.split == split) & (d.group == group)]
    lo, hi = (shown["mean"] - shown["sd"]).min(), (shown["mean"] + shown["sd"]).max()
    ax.set_ylim(lo - 0.25 * (hi - lo), hi + 0.04 * (hi - lo))       # a band below the data for the test marks
    for k, (a, b) in enumerate(sig_pairs):
        ga = d[(d.recipe == a) & (d.split == split) & (d.group == group)].set_index("epoch")
        gb = d[(d.recipe == b) & (d.split == split) & (d.group == group)].set_index("epoch")
        ep = ga.index.intersection(gb.index)
        p = np.array([welch_p(ga.loc[e, "mean"], ga.loc[e, "sd"], ga.loc[e, "n"],
                              gb.loc[e, "mean"], gb.loc[e, "sd"], gb.loc[e, "n"]) for e in ep])
        sig = ep[p < 0.05]
        y = 0.035 + 0.055 * k
        ax.scatter(sig, np.full(len(sig), y), transform=ax.get_xaxis_transform(), marker="*", s=40,
                   color=RECIPE_COLORS[b], edgecolor=RECIPE_COLORS[a], linewidth=0.8, zorder=4)
        ax.text(1.0, y, f"{a} vs {b}", transform=ax.transAxes, fontsize=8, va="center", ha="left", color=INK)
    ax.set_title(title)
    ax.set_xlabel("epoch")


def cv_curves(path, csv):
    """Cross-validation curves per epoch of the one-run recipes, one panel per recipe:
    recording MAE of the train-eval sample, each fold model's own validation fold and the
    held-out test split, mean over the five fold models with the sample SD as error bar;
    background = training phase. The legend takes the fourth grid cell."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator
    from labram.eval.age_plots import assert_max_columns, panel_labels
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    d = pd.read_csv(csv)
    lines = [("train_eval", "train-eval (500 training recordings per dataset)", "#1baf7a", "s"),
             ("val", "validation (own fold)", "#2a78d6", "o"), ("test", "test (held out)", "#eb6834", "^")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    a = list(axes.flat)
    da = d[d.group == "all"]
    for ax, recipe in zip(a[:3], ("P1", "P3", "P4")):
        for e0, e1, ph in PHASES[recipe]:
            ax.axvspan(e0 - 0.5, e1 + 0.5, color=PHASE_COLORS[ph], zorder=0, lw=0)
        for split, label, color, mk in lines:
            g = da[(da.recipe == recipe) & (da.split == split)].sort_values("epoch")
            ax.errorbar(g.epoch, g["mean"], yerr=g["sd"], color=color, marker=mk, ms=4.5, lw=1.6, capsize=3,
                        label=label, zorder=3)
        last = da[da.recipe == recipe].epoch.max()
        gv = da[(da.recipe == recipe) & (da.epoch == last)].set_index("split")["mean"]
        ax.set_title(f"{recipe}: final val {gv['val']:.2f}, train-eval {gv['train_eval']:.2f} "
                     f"(gap {gv['val'] - gv['train_eval']:.2f})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("recording MAE (years)")
        ax.set_ylim(3, 11)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(color=GRID, lw=0.8, zorder=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    a[3].axis("off")
    handles = [plt.Line2D([], [], color=c, marker=m, lw=1.6, label=lbl) for _, lbl, c, m in lines]
    handles += [Patch(facecolor=PHASE_COLORS[p], label=f"{p} phase") for p in PHASE_COLORS]
    a[3].legend(handles=handles, loc="center", frameon=False, fontsize=11)
    fig.tight_layout()
    panel_labels(a[:3])
    assert_max_columns(fig)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)


def cv_compare(path, csv):
    """The three one-run recipes compared over epochs on validation, test and LEMON (fold mean
    +- SD), first attention-only epoch dotted, per-epoch Welch tests marked; legend in the
    fourth grid cell."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from labram.eval.age_plots import assert_max_columns, panel_labels
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    d = pd.read_csv(csv)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    a = list(axes.flat)
    recipe_panel(a[0], d, "val", "all", "validation MAE (own fold)")
    recipe_panel(a[1], d, "test", "all", "test MAE (held out)")
    recipe_panel(a[2], d, "test", "lemon", "LEMON MAE (zero-shot)")
    for ax in a[:3]:
        ax.set_ylabel("recording MAE (years)")
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(color=GRID, lw=0.8, zorder=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    a[3].axis("off")
    handles = [plt.Line2D([], [], color=c, marker="o", lw=1.6, label=r) for r, c in RECIPE_COLORS.items()]
    handles += [plt.Line2D([], [], color=RECIPE_COLORS["P3"], ls=":", lw=1.4, label="first attention-only epoch, P1 and P3 (10)"),
                plt.Line2D([], [], color=RECIPE_COLORS["P4"], ls=":", lw=1.4, label="first attention-only epoch, P4 (8)")]
    handles += [plt.Line2D([], [], color=INK, marker="*", lw=0, ms=9, label="Welch p < 0.05 at that epoch")]
    a[3].legend(handles=handles, loc="center", frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.97, 1))
    panel_labels(a[:3])
    assert_max_columns(fig)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)

def cv_cohorts(path, csv):
    """Held-out test MAE per data cohort over epochs, the three one-run recipes on each axis
    (fold mean +- SD), with per-epoch Welch tests marked."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from labram.eval.age_plots import assert_max_columns, panel_labels
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    d = pd.read_csv(csv)
    panels = [("temple_normal", "Temple, normal report"), ("temple_abnormal", "Temple, abnormal report"),
              ("harvard_normal", "Harvard, normal report"), ("harvard_abnormal", "Harvard, abnormal report"),
              ("harvard_unlabelled", "Harvard, no report"), ("lemon", "LEMON, zero-shot")]
    fig, axes = plt.subplots(3, 2, figsize=(13, 13.5))
    for ax, (group, title) in zip(axes.flat, panels):
        recipe_panel(ax, d, "test", group, title)
        ax.set_ylabel("recording MAE (years)")
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(color=GRID, lw=0.8, zorder=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout(rect=(0, 0, 0.97, 1))
    panel_labels(list(axes.flat))
    assert_max_columns(fig)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default="docs/tech_report")
    ap.add_argument("--windows", required=True, help="saved windows of the cohort (eval_age_by_cohort --save-windows)")
    ap.add_argument("--cohort", required=True, help="cohort folder with manifest.json and recordings/*.npy")
    ap.add_argument("--recording", required=True)
    ap.add_argument("--trim-windows", type=int, default=0, help="windows trimmed at the start when the cohort was scored")
    ap.add_argument("--cv-curves", default=None, help="docs/data/mil_phased_cv/cv_curves.csv -> fig_cv_curves.png")
    ap.add_argument("--eeg-cases", nargs="*", default=[], metavar="STEM=LABEL",
                    help="recordings drawn as EEG with the attention weight as background (fig_attention_on_eeg.png)")
    ap.add_argument("--heatmap-cases", nargs="*", default=[], metavar="STEM=LABEL",
                    help="recordings for the whole-recording attention heatmaps (fig_attention_heatmaps.png)")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(args.out_dir, exist_ok=True)
    pipeline(os.path.join(args.out_dir, "fig_pipeline.png"))
    attention_example(os.path.join(args.out_dir, "fig_attention_example.png"), args.windows, args.cohort,
                      args.recording, args.trim_windows)
    if args.cv_curves:
        cv_curves(os.path.join(args.out_dir, "fig_cv_curves.png"), args.cv_curves)
        cv_compare(os.path.join(args.out_dir, "fig_cv_compare.png"), args.cv_curves)
        cv_cohorts(os.path.join(args.out_dir, "fig_cv_cohorts.png"), args.cv_curves)
    if args.eeg_cases:
        attention_on_eeg(os.path.join(args.out_dir, "fig_attention_on_eeg.png"), args.windows, args.cohort,
                         [tuple(c.split("=", 1)) for c in args.eeg_cases], args.trim_windows)
    if args.heatmap_cases:
        cases = [c.split("=", 1) for c in args.heatmap_cases]
        attention_heatmaps(os.path.join(args.out_dir, "fig_attention_heatmaps.png"), args.windows, args.cohort,
                           [(stem, label) for stem, label in cases], args.trim_windows)


if __name__ == "__main__":
    main()
