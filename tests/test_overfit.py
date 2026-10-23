"""Learning-curve diagnostics: overfitting, memorization, plateau, headroom."""
import math

from labram.eval.overfit import analyze


def _log(train, val, test=None):
    test = test or val
    return [{"epoch": i, "train_mae": t, "val_mae": v, "test_mae": s}
            for i, (t, v, s) in enumerate(zip(train, val, test))]


def test_overfitting_val_worse_while_train_improves():
    r = analyze(_log([9, 8, 7, 6, 5, 4], [9, 8.5, 8.2, 8.3, 8.4, 8.6]))
    assert r.overfit_streak == 3 and r.best_val_epoch == 2 and "overfitting" in r.verdict


def test_memorization_flagged_even_while_val_improves():
    r = analyze(_log([6, 4, 3, 2], [9, 8.8, 8.6, 8.5]))
    assert r.still_improving and "memorizing" in r.verdict and r.relative_gap > 0.3


def test_geometric_headroom():
    v = [10, 9, 8.5, 8.25, 8.125, 8.0625]          # gains halve: 1, .5, .25, .125, .0625
    r = analyze(_log([10.5, 9.5, 9, 8.75, 8.6, 8.5], v))
    assert abs(r.headroom_ratio - 0.5) < 1e-9 and abs(r.headroom_estimate - 0.0625) < 1e-9
    assert "slowing" in r.verdict


def test_too_few_epochs_and_plateau():
    r = analyze(_log([10, 9, 8.6], [9, 8, 7.5]))
    assert math.isnan(r.headroom_estimate) and "too few epochs" in r.verdict
    r = analyze(_log([10, 9, 8.5, 8.2, 8.0], [9, 8, 7.6, 7.61, 7.62]))
    assert "plateau" in r.verdict


def test_selection_regret_and_val_test_agreement():
    r = analyze(_log([9, 8, 7, 6], [9, 8, 7.5, 7.6], [9, 8.2, 7.9, 7.7]))
    assert r.best_val_epoch == 2 and r.best_test_epoch == 3
    assert abs(r.selection_regret - 0.2) < 1e-9
