import numpy as np
import pandas as pd
import pytest

from labram.eval.binary_metrics import (best_f1_threshold, bootstrap_ci, classification_table,
                                        score_summary, threshold_metrics)


def test_threshold_metrics_counts():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.2, 0.1])
    m = threshold_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["error_rate"] == pytest.approx(2 / 5)
    assert m["specificity"] == pytest.approx(2 / 3)


def test_best_f1_threshold_is_the_maximum_over_thresholds():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.1).astype(int)
    p = rng.random(300) + 0.5 * y
    t = best_f1_threshold(y, p)
    best = threshold_metrics(y, p, t)["f1"]
    assert all(threshold_metrics(y, p, u)["f1"] <= best + 1e-12 for u in np.unique(p))


def test_perfect_separation():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    s = score_summary(y, p, threshold=0.25)
    assert s["roc_auc"] == 1.0 and s["best_f1"] == 1.0 and s["best_error_rate"] == 0.0
    assert s["fixed_fp"] == 1 and s["fixed_tp"] == 2


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.2).astype(int)
    p = rng.random(400) + 0.4 * y
    ci = bootstrap_ci(y, p, n_boot=200)
    auc = score_summary(y, p)["roc_auc"]
    assert ci["roc_auc_lo"] <= auc <= ci["roc_auc_hi"]


def test_classification_table_carries_val_threshold_to_test():
    rng = np.random.default_rng(2)
    rows = []
    for split in ("val", "test"):
        y = (rng.random(200) < 0.3).astype(float)
        y[:5] = np.nan                                   # unknown labels are dropped
        for r, (yy, pp) in enumerate(zip(y, rng.random(200) + 0.3 * np.nan_to_num(y))):
            rows.append({"run": "A", "split": split, "recording": r, "target": "x", "y": yy, "p": pp})
    t = classification_table(pd.DataFrame(rows), n_boot=0)
    val, test = t[t.split == "val"].iloc[0], t[t.split == "test"].iloc[0]
    assert val.n == 195 and test.fixed_threshold == val.best_threshold
    assert np.isnan(val.get("fixed_threshold", np.nan))
