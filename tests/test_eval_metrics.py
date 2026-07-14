"""Tests for detailed classification metrics and per-case window aggregation
(labram.utils.eval_metrics)."""

import numpy as np
import pytest

from labram.utils.eval_metrics import aggregate_windows, classification_report


def test_binary_report_perfect_separation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    r = classification_report(p, y, is_binary=True, nb_classes=1)
    assert r.scalars["accuracy"] == pytest.approx(1.0)
    assert r.scalars["sensitivity"] == pytest.approx(1.0)
    assert r.scalars["specificity"] == pytest.approx(1.0)
    assert r.scalars["f1"] == pytest.approx(1.0)
    assert r.scalars["roc_auc"] == pytest.approx(1.0)
    # confusion matrix [[tn, fp], [fn, tp]]
    assert r.matrix.tolist() == [[2, 0], [0, 2]]
    assert (r.scalars["cm_tn"], r.scalars["cm_fp"], r.scalars["cm_fn"], r.scalars["cm_tp"]) == (2, 0, 0, 2)
    assert r.roc is not None and r.pr is not None


def test_binary_sensitivity_specificity_values():
    # tp=1, fn=1 -> sens .5 ; tn=2, fp=0 -> spec 1.0
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.2, 0.3])
    r = classification_report(p, y, is_binary=True, nb_classes=1)
    assert r.scalars["sensitivity"] == pytest.approx(0.5)
    assert r.scalars["specificity"] == pytest.approx(1.0)


def test_binary_confusion_matrix_relative_rates():
    # tp=1, fn=1 (2 actual positives); tn=2, fp=0 (2 actual negatives).
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.2, 0.3])
    r = classification_report(p, y, is_binary=True, nb_classes=1)
    # Row-normalized per true class; each row's rates sum to 1.
    assert r.scalars["cm_tp_rate"] == pytest.approx(0.5)   # TPR == sensitivity
    assert r.scalars["cm_fn_rate"] == pytest.approx(0.5)   # FNR
    assert r.scalars["cm_tn_rate"] == pytest.approx(1.0)   # TNR == specificity
    assert r.scalars["cm_fp_rate"] == pytest.approx(0.0)   # FPR
    assert r.scalars["cm_tp_rate"] == pytest.approx(r.scalars["sensitivity"])
    assert r.scalars["cm_tn_rate"] == pytest.approx(r.scalars["specificity"])
    # Absolute counts are still reported alongside the rates.
    assert (r.scalars["cm_tp"], r.scalars["cm_fn"]) == (1, 1)
    assert (r.scalars["cm_tn"], r.scalars["cm_fp"]) == (2, 0)


def test_binary_confusion_matrix_rates_no_positive_samples():
    # No actual positives: positive-row rates fall back to 0.0 (no divide-by-0).
    y = np.array([0, 0, 0, 0])
    p = np.array([0.2, 0.3, 0.1, 0.4])
    r = classification_report(p, y, is_binary=True, nb_classes=1)
    assert r.scalars["cm_tp_rate"] == 0.0
    assert r.scalars["cm_fn_rate"] == 0.0
    assert r.scalars["cm_tn_rate"] == pytest.approx(1.0)
    assert r.scalars["cm_fp_rate"] == pytest.approx(0.0)


def test_binary_single_class_guards_auc():
    y = np.array([1, 1, 1, 1])
    p = np.array([0.9, 0.8, 0.6, 0.7])
    r = classification_report(p, y, is_binary=True, nb_classes=1)
    # AUROC/AUPRC undefined with one class -> reported 0.0, no curve.
    assert r.scalars["roc_auc"] == 0.0
    assert r.scalars["pr_auc"] == 0.0
    assert r.roc is None


def test_multiclass_report_shapes_and_keys():
    logits = np.eye(3)[np.array([0, 1, 2, 0, 1, 2])] * 5.0
    y = np.array([0, 1, 2, 0, 1, 2])
    r = classification_report(logits, y, is_binary=False, nb_classes=3)
    assert r.matrix.shape == (3, 3)
    assert r.scalars["accuracy"] == pytest.approx(1.0)
    assert r.scalars["f1"] == pytest.approx(1.0)
    assert "sensitivity" in r.scalars and "specificity" in r.scalars


@pytest.fixture
def two_cases():
    groups = ["recA", "recA", "recA", "recB", "recB", "recB"]
    probs = np.array([0.9, 0.8, 0.2, 0.1, 0.2, 0.3]).reshape(-1, 1)
    tgt = np.array([1, 1, 1, 0, 0, 0]).reshape(-1, 1)
    return groups, probs, tgt


def test_aggregate_none_is_identity(two_cases):
    groups, probs, tgt = two_cases
    po, to = aggregate_windows(probs, tgt, groups, "none", is_binary=True)
    assert po is probs and to is tgt


def test_aggregate_mean(two_cases):
    groups, probs, tgt = two_cases
    po, to = aggregate_windows(probs, tgt, groups, "mean", is_binary=True)
    assert po.shape == (2, 1)
    assert po.ravel()[0] == pytest.approx((0.9 + 0.8 + 0.2) / 3)
    assert to.ravel().tolist() == [1.0, 0.0]


def test_aggregate_vote(two_cases):
    groups, probs, tgt = two_cases
    po, _ = aggregate_windows(probs, tgt, groups, "vote", is_binary=True)
    # recA: 2/3 windows >=0.5 -> positive; recB: 0/3 -> negative
    assert po.ravel().tolist() == [1.0, 0.0]


def test_aggregate_max(two_cases):
    groups, probs, tgt = two_cases
    po, _ = aggregate_windows(probs, tgt, groups, "max", is_binary=True)
    assert po.ravel().tolist() == [0.9, 0.3]


def test_aggregate_multiclass_mean_recovers_probs():
    groups = ["a", "a", "b"]
    logits = np.log(np.array([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.1, 0.1, 0.8]]))
    tgt = np.array([0, 0, 2]).reshape(-1, 1)
    out, to = aggregate_windows(logits, tgt, groups, "mean", is_binary=False)
    # out are log-probs; softmax recovers the mean of the per-window probs.
    from scipy.special import softmax
    probs = softmax(out, axis=1)
    assert probs[0] == pytest.approx([0.6, 0.25, 0.15])
    assert to.ravel().tolist() == [0.0, 2.0]


def test_aggregate_unknown_mode_raises(two_cases):
    groups, probs, tgt = two_cases
    with pytest.raises(ValueError):
        aggregate_windows(probs, tgt, groups, "bogus", is_binary=True)
