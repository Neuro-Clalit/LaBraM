"""Unit coverage for low-overhead training timing helpers."""

import time

import torch

from labram.utils.timing import PhaseTimer, StepTimer, log_timing_stats, timing_stats


def test_cpu_step_timer_reports_non_negative_step_metrics():
    timer = StepTimer(torch.device("cpu"))
    data_time = timer.start_step()
    time.sleep(0.001)
    values = timer.end_step(data_time, batch_size=3)

    assert set(values) == {"data_time_sec", "step_time_sec", "host_compute_time_sec"}
    assert all(value >= 0 for value in values.values())
    assert timer.samples_processed == 3
    assert timer.finish() == {}


def test_phase_timing_includes_throughput_when_samples_are_known():
    timer = PhaseTimer(torch.device("cpu"))
    elapsed = timer.elapsed()
    stats = timing_stats("train", elapsed, samples=4)

    assert stats["train_time_sec"] >= 0
    assert stats["train_samples_per_sec"] > 0


def test_writer_uses_epoch_axis_for_timing_metrics():
    class Writer:
        def __init__(self):
            self.calls = []

        def epoch_step(self, epoch):
            return epoch * 10

        def update(self, **kwargs):
            self.calls.append(kwargs)

    writer = Writer()
    log_timing_stats(writer, {"epoch_time_sec": 1.0}, epoch=2)

    assert writer.calls == [{"epoch_time_sec": 1.0, "head": "timing", "step": 20}]
