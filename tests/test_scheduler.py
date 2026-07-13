"""Tests for the configurable LR scheduler (labram.utils.training.build_lr_schedule)
and its wiring through runs.common.make_lr_schedule."""

import numpy as np
import pytest

from labram.configs.optim_config import OptimizerConfig
from labram.configs.train_config import TrainerConfig
from labram.runs.common import make_lr_schedule
from labram.utils.training import build_lr_schedule, cosine_scheduler

EPOCHS = 10
NITER = 5
TOTAL = EPOCHS * NITER


@pytest.mark.parametrize("sched", ["cosine", "constant", "linear", "step", "multistep"])
def test_schedule_length_and_warmup(sched):
    s = build_lr_schedule(sched, 1e-3, 1e-5, EPOCHS, NITER, warmup_epochs=2,
                          decay_epochs=3, decay_rate=0.1, decay_milestones=[4, 8])
    assert len(s) == TOTAL
    # Linear warmup: starts at start_warmup_value (0) and reaches base at the
    # end of warmup (index warmup_epochs * niter).
    assert s[0] == pytest.approx(0.0)
    assert s[2 * NITER] == pytest.approx(1e-3, rel=1e-6)


def test_cosine_matches_legacy():
    a = build_lr_schedule("cosine", 1e-3, 1e-5, EPOCHS, NITER, warmup_epochs=2)
    b = cosine_scheduler(1e-3, 1e-5, EPOCHS, NITER, warmup_epochs=2)
    assert np.allclose(a, b)


def test_constant_holds_base_after_warmup():
    s = build_lr_schedule("constant", 1e-3, 1e-5, EPOCHS, NITER, warmup_epochs=0)
    assert np.allclose(s, 1e-3)


def test_linear_decreases_to_final():
    s = build_lr_schedule("linear", 1e-3, 1e-5, EPOCHS, NITER, warmup_epochs=0)
    assert s[0] == pytest.approx(1e-3)
    assert s[-1] == pytest.approx(1e-5)
    assert np.all(np.diff(s) <= 1e-12)  # monotonically non-increasing


def test_step_decays_by_rate_every_decay_epochs():
    s = build_lr_schedule("step", 1.0, 0.0, EPOCHS, NITER, warmup_epochs=0,
                          decay_epochs=2, decay_rate=0.5)
    # epoch 0-1 -> 1.0, epoch 2-3 -> 0.5, epoch 4-5 -> 0.25 ...
    assert s[0] == pytest.approx(1.0)
    assert s[2 * NITER] == pytest.approx(0.5)
    assert s[4 * NITER] == pytest.approx(0.25)


def test_multistep_decays_at_milestones():
    s = build_lr_schedule("multistep", 1.0, 0.0, EPOCHS, NITER, warmup_epochs=0,
                          decay_rate=0.1, decay_milestones=[3, 7])
    assert s[0] == pytest.approx(1.0)
    assert s[3 * NITER] == pytest.approx(0.1)
    assert s[7 * NITER] == pytest.approx(0.01)


def test_unknown_scheduler_raises():
    with pytest.raises(ValueError):
        build_lr_schedule("bogus", 1e-3, 1e-5, EPOCHS, NITER)


def test_make_lr_schedule_uses_sched_field():
    optim = OptimizerConfig(lr=1.0, min_lr=0.0, warmup_epochs=0,
                            sched="constant")
    trainer = TrainerConfig(epochs=EPOCHS)
    s = make_lr_schedule(optim, trainer, NITER)
    assert len(s) == TOTAL
    assert np.allclose(s, 1.0)


def test_default_sched_is_cosine():
    assert OptimizerConfig().sched == "cosine"
