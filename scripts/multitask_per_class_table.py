#!/usr/bin/env python
"""Per-class metrics of a multi-task classification run, per split and dataset.

Reads the per-recording prediction table of ``scripts/eval_multitask_classification.py``
(``run, split, recording, target, p, y``; ``y`` NaN = unknown) and writes, per
run x split x dataset (``all`` / ``TUH`` / ``HEEDB``, from the recording id: HEEDB
stems start with ``sub-``) x target: recordings with a known label, positives,
ROC-AUC, PR-AUC and the balanced accuracy at probability 0.5 (as training logs it).
Targets with one class only in a cell are skipped; HEEDB cells with fewer than
10 positives are dropped (DUA small-cell rule).

    python scripts/multitask_per_class_table.py --preds e2p_preds.csv.gz --out docs/data/e2p_per_class.csv
"""
import argparse

import numpy as np
import pandas as pd

SMALL_CELL = 10


def per_class(preds: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
    p = preds.dropna(subset=["y"]).copy()
    p["dataset"] = np.where(p["recording"].astype(str).str.startswith("sub-"), "HEEDB", "TUH")
    rows = []
    for (run, split, target), g in p.groupby(["run", "split", "target"], sort=False):
        for ds, sub in (("all", g), ("TUH", g[g.dataset == "TUH"]), ("HEEDB", g[g.dataset == "HEEDB"])):
            y, s = sub["y"].to_numpy().astype(int), sub["p"].to_numpy()
            pos = int(y.sum())
            if len(np.unique(y)) < 2 or (ds == "HEEDB" and pos < SMALL_CELL):
                continue
            rows.append({"run": run, "split": split, "dataset": ds, "target": target,
                         "known": len(y), "positives": pos, "prevalence": pos / len(y),
                         "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
                         "balanced_accuracy": balanced_accuracy_score(y, (s >= 0.5).astype(int))})
    return pd.DataFrame(rows)


def wide(table: pd.DataFrame, run: str, dataset: str = "all") -> pd.DataFrame:
    """One row per target: val / test / train ROC-AUC, PR-AUC, balanced accuracy."""
    t = table[(table.run == run) & (table.dataset == dataset)]
    out = t.pivot_table(index="target", columns="split",
                        values=["roc_auc", "pr_auc", "balanced_accuracy", "positives"], sort=False)
    cols = [(m, s) for s in ("val", "test", "train") for m in ("roc_auc", "pr_auc", "balanced_accuracy", "positives")
            if (m, s) in out.columns]
    out = out[cols]
    out.columns = [f"{s} {m}" for m, s in cols]
    return out.reindex(list(dict.fromkeys(t.target)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    table = per_class(pd.read_csv(args.preds))
    table.round(4).to_csv(args.out, index=False)
    for run in table.run.unique():
        for ds in ("all", "TUH", "HEEDB"):
            print(f"\n== {run} / {ds}")
            print(wide(table, run, ds).round(3).to_string())


if __name__ == "__main__":
    main()
