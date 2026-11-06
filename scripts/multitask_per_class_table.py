#!/usr/bin/env python
"""Per-class metrics of a multi-task classification run, per split and dataset.

Reads the per-recording prediction table of ``scripts/eval_multitask_classification.py``
(``run, split, recording, target, p, y``; ``y`` NaN = unknown) and writes, per
run x split x dataset (``all`` / ``TUH`` / ``HEEDB``, from the recording id: HEEDB
stems start with ``sub-``) x target: recordings with a known label, positives,
ROC-AUC, PR-AUC; at probability 0.5 (as training logs them) the balanced
accuracy, balanced error and F1; and at the validation best-F1 threshold
``t_val`` of the same run / dataset / target (chosen on val, carried over to
test and train) the F1 and balanced error (``*_tval``); and the best values
over all thresholds of the split itself -- the maximum F1 (``f1_best`` at
``t_f1``) and the minimum balanced error (``balanced_error_best`` at ``t_be``,
the Youden point). The ``*_best`` values are optimistic outside validation
(threshold chosen on the scored labels).
Targets with one class only in a cell are skipped; HEEDB cells with fewer than
10 positives are dropped (DUA small-cell rule).

    python scripts/multitask_per_class_table.py --preds e2p_preds.csv.gz --out docs/data/e2p_per_class.csv
"""
import argparse

import numpy as np
import pandas as pd

SMALL_CELL = 10
METRICS = ["roc_auc", "pr_auc", "balanced_accuracy", "balanced_error", "f1", "f1_tval", "balanced_error_tval",
           "f1_best", "t_f1", "balanced_error_best", "t_be"]


def min_balanced_error(y: np.ndarray, s: np.ndarray):
    """``(threshold, balanced error)`` minimising the balanced error (maximising
    Youden's J = TPR - FPR) over the ROC thresholds."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, s)
    be = 1 - (tpr + 1 - fpr) / 2
    i = int(np.argmin(be))
    return float(min(thr[i], 1.0)), float(be[i])


def _cells(p: pd.DataFrame):
    for (run, split, target), g in p.groupby(["run", "split", "target"], sort=False):
        for ds, sub in (("all", g), ("TUH", g[g.dataset == "TUH"]), ("HEEDB", g[g.dataset == "HEEDB"])):
            y, s = sub["y"].to_numpy().astype(int), sub["p"].to_numpy()
            if len(np.unique(y)) < 2 or (ds == "HEEDB" and int(y.sum()) < SMALL_CELL):
                continue
            yield run, split, target, ds, y, s


def per_class(preds: pd.DataFrame) -> pd.DataFrame:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sklearn.metrics import average_precision_score, roc_auc_score
    from labram.eval.binary_metrics import best_f1_threshold, threshold_metrics
    p = preds.dropna(subset=["y"]).copy()
    p["dataset"] = np.where(p["recording"].astype(str).str.startswith("sub-"), "HEEDB", "TUH")
    t_val = {(run, target, ds): best_f1_threshold(y, s)
             for run, split, target, ds, y, s in _cells(p) if split == "val"}
    rows = []
    for run, split, target, ds, y, s in _cells(p):
        at05 = threshold_metrics(y, s, 0.5)
        row = {"run": run, "split": split, "dataset": ds, "target": target,
               "known": len(y), "positives": int(y.sum()), "prevalence": y.mean(),
               "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
               "balanced_accuracy": 1 - at05["balanced_error"],
               "balanced_error": at05["balanced_error"], "f1": at05["f1"]}
        t_f1 = best_f1_threshold(y, s)
        t_be, be_best = min_balanced_error(y, s)
        row.update({"f1_best": threshold_metrics(y, s, t_f1)["f1"], "t_f1": t_f1,
                    "balanced_error_best": be_best, "t_be": t_be})
        t = t_val.get((run, target, ds))
        if t is not None:
            at_t = threshold_metrics(y, s, t)
            row.update({"t_val": t, "f1_tval": at_t["f1"], "balanced_error_tval": at_t["balanced_error"]})
        rows.append(row)
    return pd.DataFrame(rows)


def wide(table: pd.DataFrame, run: str, dataset: str = "all") -> pd.DataFrame:
    """One row per target: val / test / train :data:`METRICS` and positives."""
    t = table[(table.run == run) & (table.dataset == dataset)]
    out = t.pivot_table(index="target", columns="split",
                        values=METRICS + ["positives"], sort=False)
    cols = [(m, s) for s in ("val", "test", "train") for m in METRICS + ["positives"]
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
