#!/usr/bin/env python
"""CHBP corrections in one figure: what each fix gains on CHBP and what it costs in-domain.

Rows (all the window model E2L or a retrained variant of it, first 5 minutes,
no CHBP labels anywhere; ``docs/chbp_domain_shift.md`` Section 10):

* as recorded            E2L on raw CHBP;
* test-time fixes        one gain for the cohort, and spectrum matching per cap
                         (``chbp_fix_methods.csv``); the model is unchanged, so
                         the in-domain error is E2L's;
* retrained variants     band 1-45 Hz, gain / filter augmentation, band +
                         relative amplitude, band + amplitude normalisation
                         (``cohort_breakdown_all_variants.csv``,
                         ``adults_compare_all_variants.csv``).

Panel (a): MAE on CHBP with the bias in brackets, and the MAE of predicting
CHBP's mean age. Panel (b): MAE on the Temple + Harvard test split.

    python scripts/chbp_variant_bars.py --out docs/tech_report/fig_chbp_fixes.png
"""
import argparse
import os

import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "data", "external_eval", "chbp")
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"
KIND = {"as recorded": ("#6b6a64", "as recorded"), "test": ("#1baf7a", "test-time fix, model unchanged"),
        "train": ("#2a78d6", "retrained on Temple + Harvard")}
ROWS = [("E2L, as recorded", "as recorded", "E2L"),
        ("one gain for the cohort (× 1.615)", "test", "global gain x 1.615"),
        ("spectrum matching per cap", "test", "spectrum matching per cap"),
        ("band 1–45 Hz", "train", "band"),
        ("gain and filter augmentation", "train", "augment"),
        ("band + relative amplitude", "train", "relamp"),
        ("band + amplitude normalisation", "train", "robust")]
PREDICT_MEAN = 7.27          # MAE of predicting CHBP's mean age (docs/chbp_domain_shift.md)


def collect():
    fixes = pd.read_csv(os.path.join(D, "chbp_fix_methods.csv"))
    fixes = fixes[fixes["model"] == "E2L"].set_index("method")
    br = pd.read_csv(os.path.join(D, "cohort_breakdown_all_variants.csv"))
    chbp = br[(br["set"] == "chbp") & (br["ages"] == "adults")].set_index("model")
    indom = pd.read_csv(os.path.join(D, "adults_compare_all_variants.csv"))
    indom = indom[(indom["split"] == "test") & (indom["group"] == "all")].iloc[0]
    rows = []
    for label, kind, key in ROWS:
        if kind == "test":
            rows.append((label, kind, fixes.loc[key, "mae"], fixes.loc[key, "bias"], fixes.loc[key, "r"], indom["E2L"]))
        else:
            rows.append((label, kind, chbp.loc[key, "mae"], chbp.loc[key, "bias"], chbp.loc[key, "r"], indom[key]))
    return pd.DataFrame(rows, columns=["label", "kind", "chbp_mae", "chbp_bias", "chbp_r", "in_domain_mae"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t = collect()
    print(t.round(2).to_string(index=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True)
    y = range(len(t))[::-1]
    colors = [KIND[k][0] for k in t["kind"]]
    for ax, col, title, xmax in zip(axes, ("chbp_mae", "in_domain_mae"),
                                    ("(a) CHBP: MAE in years (bias in brackets)",
                                     "(b) Temple + Harvard test: MAE in years"), (12.6, 9.2)):
        ax.barh(list(y), t[col], color=colors, height=0.62, zorder=3)
        for yi, (_, r) in zip(y, t.iterrows()):
            note = f"{r[col]:.2f}" + (f"  ({r['chbp_bias']:+.1f})".replace("-", "−") if col == "chbp_mae" else "")
            ax.text(r[col] + 0.12, yi, note, va="center", fontsize=9.5, color=INK, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.set_xlim(0, xmax)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(t["label"], fontsize=10)
    axes[0].axvline(PREDICT_MEAN, color=MUTED, lw=1, ls="--", zorder=4)
    axes[1].axvline(t["in_domain_mae"].iloc[0], color=MUTED, lw=1, ls="--", zorder=4)
    for ax in axes:
        ax.set_ylim(-0.95, len(t) - 0.5)
    axes[0].text(PREDICT_MEAN + 0.12, -0.9, "predicting the mean age", fontsize=8.5, color=MUTED, va="bottom")
    axes[1].text(t["in_domain_mae"].iloc[0] + 0.1, -0.9, "E2L", fontsize=8.5, color=MUTED, va="bottom")
    fig.legend(handles=[Patch(color=c, label=n) for c, n in KIND.values()], loc="lower center", ncol=3, frameon=False,
               fontsize=9.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(args.out)


if __name__ == "__main__":
    main()
