#!/usr/bin/env python
"""Tables and figures of MIL Stage 1 (``scripts/mil_stage1.py`` outputs) for
``docs/mil_stage1_results.md``.

    python scripts/mil_stage1_report.py --in /data/runs/heedb/mil_stage1 \\
        --fig-dir docs/figures --csv-dir docs/data/mil_stage1
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ORDER = ["M0-run", "M0", "M1-h", "M1-q", "M1-hq", "M1-hq-time", "M1-hq-noent", "M2-h", "M2-hq",
         "M3-h", "M3-q", "M3-hq", "M4-hq", "M4-hq-sa"]


def load(root, protocol):
    d = os.path.join(root, protocol)
    m = pd.read_csv(os.path.join(d, "metrics.csv"))
    a = pd.read_csv(os.path.join(d, "analytics.csv"))
    with open(os.path.join(d, "excess_by_decile.json")) as fh:
        dec = json.load(fh)
    return m, a, dec


def cell(m, run, variant, split, group, ci=False):
    r = m[(m.run == run) & (m.variant == variant) & (m.split == split) & (m.group == group)]
    if r.empty:
        return "—"
    r = r.iloc[0]
    if ci:
        if pd.isna(r.get("delta_vs_ref")):
            return "—"
        return f"{r.delta_vs_ref:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]"
    return f"{r.mae:.2f}"


def mae_table(m, protocol):
    rows = ["| variant | E2L val | E2L test | E2L test TUH | E2L test HEEDB | E2 val | E2 test | Δ test vs M0, E2L | Δ test vs M0, E2 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for v in ORDER:
        if not ((m.variant == v)).any():
            continue
        rows.append(f"| {v} | {cell(m, 'E2L', v, 'val', 'all')} | {cell(m, 'E2L', v, 'test', 'all')} | "
                    f"{cell(m, 'E2L', v, 'test', 'temple')} | {cell(m, 'E2L', v, 'test', 'harvard')} | "
                    f"{cell(m, 'E2', v, 'val', 'all')} | {cell(m, 'E2', v, 'test', 'all')} | "
                    f"{cell(m, 'E2L', v, 'test', 'all', True)} | {cell(m, 'E2', v, 'test', 'all', True)} |")
    return "\n".join(rows)


def cohort_table(m, run, variants):
    groups = ["temple_normal", "temple_abnormal", "harvard_normal", "harvard_abnormal", "harvard_unlabelled"]
    rows = ["| variant | " + " | ".join(groups) + " |", "|---" * (len(groups) + 1) + "|"]
    for v in variants:
        rows.append(f"| {v} | " + " | ".join(cell(m, run, v, "test", g) for g in groups) + " |")
    return "\n".join(rows)


def analytics_table(a, run):
    cols = [("win_rho_within", "ρ within"), ("win_frac_neg", "share ρ<0"), ("win_err_low_w", "Δ low-w (y)"),
            ("win_err_high_w", "Δ high-w (y)"), ("rho_within_w_minute", "ρ(w, minute)"),
            ("rho_within_w_stft_ch_sd", "ρ(w, ch. spread)"), ("rho_within_w_emg", "ρ(w, EMG)"),
            ("mean_max_w", "mean max w"), ("rec_rho_conc", "ρ(conc, rec err)"),
            ("rec_rho_wsd", "ρ(wSD, rec err)"), ("rec_wsd_mae@50", "MAE@50 by wSD"),
            ("max_abs_weighted_sum_minus_pred", "max |Σw ŷ − ŷ|")]
    rows = ["| variant | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for v in ORDER:
        r = a[(a.run == run) & (a.variant == v)]
        if r.empty or v.startswith("M0"):
            continue
        r = r.iloc[0]
        rows.append(f"| {v} | " + " | ".join("—" if pd.isna(r.get(c)) else f"{r[c]:.2f}" for c, _ in cols) + " |")
    return "\n".join(rows)


def figure(dec, path, protocol, variants=("M1-h", "M1-q", "M3-q", "M2-hq", "M4-hq")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import panel_labels
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, run in zip(axes, ("E2L", "E2")):
        for v in variants:
            key = f"{run}/{protocol}/{v}"
            if key in dec:
                c = dec[key]
                xs = sorted(int(k) for k in c)
                ax.plot(xs, [c[str(x)] for x in xs], "-o", ms=4, label=v)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("within-recording attention-weight decile (1 = lowest weight)")
        ax.set_title(f"{run}, test, {protocol}")
    axes[0].set_ylabel("window |error| minus recording mean (y)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    panel_labels(list(axes))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="root", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--csv-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.csv_dir, exist_ok=True)
    for protocol in ("val_cf", "train"):
        if not os.path.exists(os.path.join(args.root, protocol, "metrics.csv")):
            continue
        m, a, dec = load(args.root, protocol)
        m.to_csv(os.path.join(args.csv_dir, f"{protocol}_metrics.csv"), index=False)
        a.to_csv(os.path.join(args.csv_dir, f"{protocol}_analytics.csv"), index=False)
        print(f"\n## {protocol}: MAE\n" + mae_table(m, protocol))
        print(f"\n## {protocol}: E2L test per cohort\n" + cohort_table(m, "E2L", ["M0-run", "M0", "M1-h", "M1-q", "M3-q", "M2-hq", "M4-hq"]))
        print(f"\n## {protocol}: E2L attention analytics (test)\n" + analytics_table(a, "E2L"))
        print(f"\n## {protocol}: E2 attention analytics (test)\n" + analytics_table(a, "E2"))
        figure(dec, os.path.join(args.fig_dir, f"mil_stage1_weight_rank_{protocol}.png"), protocol)


if __name__ == "__main__":
    main()
