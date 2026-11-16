#!/usr/bin/env python
"""Markdown rows of the per-cohort brain-age tables (docs/tdbrain_onboarding.md
Section 17) from ``external_cohort_table.py``'s CSV: one table per cohort, our
models without rejection and with each rejection score, MAE [95 % CI] and R²/r
for all / healthy / pathological (/ unlabelled) recordings.

    python scripts/external_cohort_markdown.py --table docs/data/external_eval/tdbrain/external_cohort_table.csv
"""
import argparse

import pandas as pd

MODELS = [("E2L", "E2L", "LaBraM, window mean"), ("P3ens", "P3 ensemble", "LaBraM + ABMIL, 5-fold ens."),
          ("P4ens", "P4 ensemble", "LaBraM + ABMIL (+ window loss), 5-fold ens.")]
REJ = [("none", "", None), ("reject 20% (spread)", ", reject 20 % (spread)", "+ spread reject"),
       ("reject 20% (total)", ", reject 20 % (total)", "+ total reject"),
       ("reject 20% (learned)", ", reject 20 % (learned)", "+ learned reject")]
GROUPS = {"all": "all", "healthy": "healthy", "pathological": "pathological", "unlabelled": "unlabelled"}


def cell(r, ci):
    if r is None or pd.isna(r.get("mae")):
        return "—", "—"
    mae = f"{r.mae:.2f}" + (f" [{r.mae_lo:.2f}, {r.mae_hi:.2f}]" if ci else "")
    return mae, f"{r.r2:.2f} / {r.r:.2f}".replace("-", "−")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", required=True)
    ap.add_argument("--protocol", nargs="*", default=["temple=in-domain", "harvard=in-domain"])
    args = ap.parse_args()
    t = pd.read_csv(args.table)
    protocol = dict(p.split("=") for p in args.protocol)
    for cohort in dict.fromkeys(t.cohort):
        c = t[t.cohort == cohort]
        groups = [g for g in GROUPS if ((c.group == g) & c.mae.notna()).any()]
        base = c[(c.rejection == "none") & (c.model == MODELS[0][0])].set_index("group")
        print(f"\n### {cohort}: " + "; ".join(f"{g} n = {int(base.loc[g].n):,} ({int(base.loc[g].n_subjects):,} subjects), "
                                             f"mean age {base.loc[g].age_mean:.1f}" for g in groups))
        head = ["model", "method", "protocol"] + [x for g in groups for x in (f"MAE, {g}", f"R²/r, {g}")]
        print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        for m, name, method in MODELS:
            for rej, suffix, extra in REJ:
                rows = c[(c.model == m) & (c.rejection == rej)].set_index("group")
                if rows.empty:
                    continue
                cells = [x for g in groups for x in cell(rows.loc[g] if g in rows.index else None, g == "all")]
                print(f"| ours, {name}{suffix} | {method if extra is None else name.split()[0] + ' ' + extra} | "
                      f"{protocol.get(cohort, 'zero-shot')} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
