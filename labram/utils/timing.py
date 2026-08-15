"""Low-overhead training timing helpers.

Wall-clock timings are always available. Exact GPU kernel timing is optional:
CUDA events are collected opportunistically during an epoch and synchronized
once at its end, rather than forcing a device-wide synchronization per batch.
"""
from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import Deque, Dict, Optional, Tuple

import torch


def _cuda_events_enabled(device: torch.device, precise_cuda: bool) -> bool:
    return bool(precise_cuda and device.type == "cuda" and torch.cuda.is_available())


class StepTimer:
    """Collect per-batch data wait and host timing metrics."""

    def __init__(self, device: torch.device, precise_cuda: bool = False) -> None:
        self.device = device
        self.precise_cuda = _cuda_events_enabled(device, precise_cuda)
        self._last_step_end = perf_counter()
        self._step_start = self._last_step_end
        self._gpu_start: Optional[torch.cuda.Event] = None
        self._pending: Deque[Tuple[torch.cuda.Event, torch.cuda.Event]] = deque()
        self.samples_processed = 0
        self.steps = 0

    def start_step(self) -> float:
        """Mark a batch as available and return the time spent waiting for it."""
        now = perf_counter()
        self._step_start = now
        if self.precise_cuda:
            self._gpu_start = torch.cuda.Event(enable_timing=True)
            self._gpu_start.record()
        return now - self._last_step_end

    def end_step(self, data_time_sec: float, batch_size: int) -> Dict[str, float]:
        """Return host metrics for one batch and queue its optional GPU event."""
        now = perf_counter()
        if self.precise_cuda and self._gpu_start is not None:
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_end.record()
            self._pending.append((self._gpu_start, gpu_end))
            self._gpu_start = None
        step_time = now - self._step_start
        self._last_step_end = now
        self.samples_processed += int(batch_size)
        self.steps += 1
        return {
            "data_time_sec": data_time_sec,
            "step_time_sec": step_time,
            "host_compute_time_sec": max(0.0, step_time - data_time_sec),
        }

    def collect_ready_gpu_times(self) -> Dict[str, float]:
        """Return completed GPU event durations without synchronizing the GPU."""
        values = []
        while self._pending and self._pending[0][1].query():
            start, end = self._pending.popleft()
            values.append(start.elapsed_time(end) / 1000.0)
        return ({"gpu_compute_time_sec": sum(values) / len(values)}
                if values else {})

    def finish(self) -> Dict[str, float]:
        """Flush optional CUDA events at an epoch boundary."""
        if self.precise_cuda and self._pending:
            torch.cuda.synchronize(self.device)
        values = []
        while self._pending:
            start, end = self._pending.popleft()
            values.append(start.elapsed_time(end) / 1000.0)
        return ({"gpu_compute_time_sec": sum(values) / len(values)}
                if values else {})


class PhaseTimer:
    """Measure an epoch-level phase, synchronizing only at phase boundaries."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._started = perf_counter()

    def elapsed(self) -> float:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        return perf_counter() - self._started


def timing_stats(prefix: str, elapsed_sec: float, samples: int = 0) -> Dict[str, float]:
    """Build stable epoch-level timing keys for logs and metric writers."""
    stats = {f"{prefix}_time_sec": elapsed_sec}
    if samples > 0 and elapsed_sec > 0:
        stats[f"{prefix}_samples_per_sec"] = samples / elapsed_sec
    return stats


def log_timing_stats(log_writer, stats: Dict[str, float], epoch: int) -> None:
    """Write epoch timing values without disturbing the per-step writer axis."""
    if log_writer is None:
        return
    step = log_writer.epoch_step(epoch) if hasattr(log_writer, "epoch_step") else epoch
    for key, value in stats.items():
        log_writer.update(**{key: value}, head="timing", step=step)
