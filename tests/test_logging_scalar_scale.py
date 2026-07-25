"""Tests for scalar-scale hygiene in the metric writers: series whose values are
>> 1 (confusion-matrix cell counts, gradient norm, AMP loss scale) are logged on
separate plots so they do not flatten the normalized [0, 1] metrics/lr onto a
single shared axis. Also covers the millisecond timestamp appended to ClearML
task names.
"""

import re

from labram.train.train_finetune import (
    _LOGGED_EVAL_COUNT_KEYS,
    _LOGGED_EVAL_RATE_KEYS,
    _log_eval_stats,
)


class _RecordingWriter:
    def __init__(self):
        self.updates = []  # (head, step, kwargs)

    def update(self, head="scalar", step=None, **kwargs):
        self.updates.append((head, step, kwargs))


def test_eval_stats_split_rates_from_counts():
    w = _RecordingWriter()
    stats = {
        "accuracy": 0.9, "f1": 0.8, "loss": 0.3,   # normalized / O(1)
        "cm_tn": 1500, "cm_fp": 20, "cm_fn": 30, "cm_tp": 1450,  # >> 1 counts
        "n_parameters": 5_000_000,  # not a logged eval key -> ignored
    }
    _log_eval_stats(w, stats, head="val", epoch=4)

    by_head = {}
    for head, step, kwargs in w.updates:
        assert step == 4
        for k in kwargs:
            by_head[k] = head

    # Rates on the normalized "val" plot.
    assert by_head["accuracy"] == "val"
    assert by_head["f1"] == "val"
    assert by_head["loss"] == "val"
    # Counts on the separate "val_cm" plot, never on the normalized one.
    assert by_head["cm_tn"] == "val_cm"
    assert by_head["cm_tp"] == "val_cm"
    # Non-eval keys are not logged.
    assert "n_parameters" not in by_head


def test_rate_and_count_key_sets_are_disjoint():
    assert not (set(_LOGGED_EVAL_RATE_KEYS) & set(_LOGGED_EVAL_COUNT_KEYS))
    # The confusion-matrix cells are exactly the count group.
    assert set(_LOGGED_EVAL_COUNT_KEYS) == {"cm_tn", "cm_fp", "cm_fn", "cm_tp"}


def test_grad_norm_and_loss_scale_not_on_opt_plot():
    # grad_norm (can be >> 1) and loss_scale (AMP, ~65536) must not share the
    # small-valued "opt" (lr/wd) plot.
    import labram.optim_factory as of

    class _Opt:
        param_groups = [{"lr": 5e-4, "weight_decay": 0.05}]

    w = _RecordingWriter()
    of.log_lr_wd_grad_metrics(_MetricLoggerStub(), _Opt(), grad_norm=12.5, log_writer=w)
    head_by_key = {k: head for head, _, kw in w.updates for k in kw}
    assert head_by_key["lr"] == "opt"
    assert head_by_key["weight_decay"] == "opt"
    assert head_by_key["grad_norm"] == "grad"  # moved off "opt"


class _MetricLoggerStub:
    def update(self, **kwargs):
        pass


def test_timestamp_ms_format():
    from labram.runs.common import _timestamp_ms

    ts = _timestamp_ms()
    # YYYYmmdd_HHMMSS_fff (millisecond precision, 3 trailing digits)
    assert re.fullmatch(r"\d{8}_\d{6}_\d{3}", ts), ts


def test_clearml_config_appends_timestamp_by_default():
    from labram.configs.train_config import ClearMLConfig

    assert ClearMLConfig().append_timestamp is True
