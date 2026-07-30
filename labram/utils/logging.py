# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# SmoothedValue, MetricLogger, TensorboardLogger, ClearMLLogger, MultiWriter,
# and the shared Python-logging setup.
# ---------------------------------------------------------

import datetime
import logging
import math
import os
import sys
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from labram.utils.distributed import is_dist_avail_and_initialized


# ============================================================
# Python logging setup
# ============================================================

LOGGER_NAME = "labram"

# Handler attached by ``configure_logging`` is tagged so re-configuration is
# idempotent (we replace our own handlers without touching foreign ones).
_HANDLER_TAG = "_labram_handler"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return the shared ``labram`` logger (or a dotted child of it).

    Modules should call ``get_logger(__name__)`` at import time and log through
    the result. Output only appears once ``configure_logging`` has attached a
    handler (done by the runners in :func:`setup_environment`); until then the
    records are silently dropped, which keeps library/test imports quiet.
    """
    root = logging.getLogger(LOGGER_NAME)
    if name is None or name == LOGGER_NAME or not name:
        return root
    # Collapse the package prefix so ``labram.train.foo`` -> ``labram.train.foo``
    # stays a child of ``labram`` rather than creating a detached tree.
    if name.startswith(LOGGER_NAME + "."):
        return logging.getLogger(name)
    return root.getChild(name)


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    rank: int = 0,
    force: bool = True,
) -> logging.Logger:
    """Configure the shared ``labram`` logger with a rank-aware console handler.

    Only rank 0 logs at ``level``; other ranks are raised to WARNING so
    per-process chatter does not multiply across a distributed job. A file
    handler is added on rank 0 when ``log_file`` is given. Safe to call more
    than once — our own handlers are cleared and rebuilt, foreign handlers are
    left untouched.
    """
    logger = logging.getLogger(LOGGER_NAME)

    if force:
        for handler in [h for h in logger.handlers if getattr(h, _HANDLER_TAG, False)]:
            logger.removeHandler(handler)
    elif logger.handlers:
        return logger

    effective_level = level if rank == 0 else max(level, logging.WARNING)
    logger.setLevel(effective_level)
    # Own the output; don't double-log through the root logger's handlers.
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="[%(asctime)s][rank%(rank)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class _RankFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.rank = rank
            return True

    rank_filter = _RankFilter()

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(fmt)
    stream_handler.addFilter(rank_filter)
    setattr(stream_handler, _HANDLER_TAG, True)
    logger.addHandler(stream_handler)

    if log_file and rank == 0:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        file_handler.addFilter(rank_filter)
        setattr(file_handler, _HANDLER_TAG, True)
        logger.addHandler(file_handler)

    return logger


_logger = get_logger()


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """Warning: does not synchronize the deque."""
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    _logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    _logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        _logger.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


# ============================================================
# Relative (scale-free) metric logging
# ============================================================

RELATIVE_SUFFIX = "_rel"


def relative_components(values: Mapping[str, Any],
                        suffix: str = RELATIVE_SUFFIX) -> Dict[str, float]:
    """Turn absolute per-component metrics into their share of the total.

    Used for the per-loss-component and per-component-gradient-norm series: each
    value is divided by the sum of the absolute component values, so the result
    is scale-free (the shares sum to 1, and lie in ``[0, 1]`` for the
    non-negative losses and grad norms this is applied to) and shows the
    *balance* between the terms rather than their raw magnitudes — which is what
    makes the plots comparable across runs, loss weights and datasets.

    ``values`` must contain only components (no total). Keys gain ``suffix``.
    A zero (or non-finite) total yields all-zero shares rather than NaN/inf.
    """
    numeric: Dict[str, float] = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().float().mean().item()
        numeric[key] = float(value)

    total = sum(abs(v) for v in numeric.values())
    if not total or not math.isfinite(total):
        return {f"{key}{suffix}": 0.0 for key in numeric}
    return {f"{key}{suffix}": v / total for key, v in numeric.items()}


def relative_components_if_enabled(values: Optional[Mapping[str, Any]],
                                   logging_cfg: Any = None) -> Optional[Dict[str, Any]]:
    """:func:`relative_components` gated on ``logging_cfg.relative_loss_components``.

    Returns the values unchanged (as a plain dict) when the option is off, and
    passes ``None``/empty through so call sites can forward an unavailable loss
    breakdown untouched.
    """
    if not values:
        return values
    if logging_cfg is None or not getattr(logging_cfg, 'relative_loss_components', False):
        return dict(values)
    return relative_components(values)


class _RelativeStepMixin(object):
    """Maps absolute training steps onto a normalized progress x-axis.

    Once :meth:`configure_relative_steps` is called with the run's total number
    of logging steps, ``set_step`` reports progress in ``[0, scale]`` (per-mille
    of the run by default) instead of the raw global iteration, so runs with
    different dataset sizes / batch sizes / epoch counts overlay on a shared
    axis. Until it is called (or when it is called with a non-positive total)
    the writers behave exactly as before: absolute steps, passed through.

    Call sites that log per-epoch metrics with an explicit ``step=`` pass the
    epoch through :meth:`epoch_step` so those series land on the same axis.
    """

    def _init_relative_steps(self) -> None:
        self.step = 0
        self._abs_step = 0
        self._rel_enabled = False
        self._rel_total_steps: Optional[int] = None
        self._rel_total_epochs: Optional[int] = None
        self._rel_scale = 1000

    def configure_relative_steps(self, total_steps: Optional[int] = None,
                                 total_epochs: Optional[int] = None,
                                 scale: int = 1000) -> bool:
        """Enable the relative axis for a run of ``total_steps`` logging steps.

        Returns whether the relative axis is active (a non-positive
        ``total_steps``/``scale`` leaves the writer on the absolute axis).
        """
        if not total_steps or total_steps <= 0 or not scale or scale <= 0:
            self._rel_enabled = False
            return False
        self._rel_total_steps = int(total_steps)
        self._rel_total_epochs = int(total_epochs) if total_epochs else None
        self._rel_scale = int(scale)
        self._rel_enabled = True
        self.step = self._to_relative(self._abs_step)
        return True

    def _to_relative(self, step: int) -> int:
        """Map an absolute iteration onto the configured progress axis."""
        if not self._rel_enabled:
            return step
        progress = min(max(step / self._rel_total_steps, 0.0), 1.0)
        return int(round(progress * self._rel_scale))

    def epoch_step(self, epoch: int) -> int:
        """Map an epoch index onto the same axis as the iteration-based steps.

        Epoch-level metrics describe the state *after* the epoch, so epoch ``e``
        of ``E`` maps to progress ``(e + 1) / E``. Returns ``epoch`` unchanged
        when the relative axis is off.
        """
        if not self._rel_enabled or not self._rel_total_epochs:
            return epoch
        progress = min(max((epoch + 1) / self._rel_total_epochs, 0.0), 1.0)
        return int(round(progress * self._rel_scale))

    def set_step(self, step: Optional[int] = None) -> None:
        if step is not None:
            self._abs_step = step
        else:
            self._abs_step += 1
        self.step = self._to_relative(self._abs_step)


def _confusion_matrix_markdown(matrix: Any, labels: Optional[List[Any]] = None) -> str:
    """Render a confusion matrix as a GitHub-flavoured markdown table.

    Used by :class:`TensorboardLogger` (which has no native confusion-matrix
    widget) so the matrix is still viewable in the TensorBoard *Text* tab.
    """
    rows = [list(r) for r in matrix]
    n = len(rows)
    ticks = list(labels) if labels is not None else list(range(n))
    header = "| true\\pred | " + " | ".join(str(t) for t in ticks) + " |"
    sep = "| --- " * (n + 1) + "|"
    body = []
    for i, row in enumerate(rows):
        body.append("| " + str(ticks[i]) + " | " + " | ".join(str(int(v)) for v in row) + " |")
    return "\n".join([header, sep] + body)


class TensorboardLogger(_RelativeStepMixin):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self._init_relative_steps()

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def update_image(self, head='images', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            self.writer.add_image(head + "/" + k, v, self.step if step is None else step)

    def report_confusion_matrix(self, title, matrix, step=None, labels=None, series='confusion_matrix'):
        """TensorBoard has no native CM widget — log a markdown table as text."""
        if matrix is None:
            return
        self.writer.add_text(
            f"{title}/{series}", _confusion_matrix_markdown(matrix, labels),
            self.step if step is None else step)

    def report_figure(self, title, figure, step=None, series='figure'):
        if figure is None:
            return
        self.writer.add_figure(
            f"{title}/{series}", figure, self.step if step is None else step)

    def report_single_value(self, name, value):
        """TensorBoard has no single-value concept — record it as a scalar under
        the ``summary/`` namespace so the final value is still captured."""
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.writer.add_scalar(f"summary/{name}", float(value), 0)

    def flush(self):
        self.writer.flush()


class ClearMLLogger(_RelativeStepMixin):
    """Scalar/image logger that mirrors :class:`TensorboardLogger` but forwards
    to a ClearML ``Task`` logger.

    The writer surface (``set_step`` / ``update`` / ``update_image`` / ``flush``)
    matches ``TensorboardLogger`` so it can be dropped into the same
    ``log_writer`` slot or combined with it via :class:`MultiWriter`. When the
    task's logger is unavailable (e.g. ClearML not installed) every method is a
    no-op, so training is never blocked by the optional dependency.

    ClearML groups scalars as (title, series); we map ``head`` -> title and each
    keyword name -> series, matching Tensorboard's ``head/name`` convention.
    """

    def __init__(self, task: Any = None, clearml_logger: Any = None):
        self.task = task
        if clearml_logger is not None:
            self._logger = clearml_logger
        elif task is not None and hasattr(task, "get_logger"):
            self._logger = task.get_logger()
        else:
            self._logger = None
        self._init_relative_steps()

    def update(self, head='scalar', step=None, **kwargs):
        if self._logger is None:
            return
        iteration = self.step if step is None else step
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            self._logger.report_scalar(
                title=head, series=k, value=float(v), iteration=iteration)

    def update_image(self, head='images', step=None, **kwargs):
        if self._logger is None:
            return
        iteration = self.step if step is None else step
        for k, v in kwargs.items():
            if v is None:
                continue
            self._logger.report_image(
                title=head, series=k, iteration=iteration, image=v)

    def report_confusion_matrix(self, title, matrix, step=None, labels=None, series='confusion_matrix'):
        """Log a native (interactive) ClearML confusion matrix."""
        if self._logger is None or matrix is None:
            return
        import numpy as np
        kwargs = {}
        if labels is not None:
            ticks = [str(t) for t in labels]
            kwargs['xlabels'] = ticks
            kwargs['ylabels'] = ticks
        self._logger.report_confusion_matrix(
            title=title, series=series, matrix=np.asarray(matrix),
            iteration=self.step if step is None else step,
            xaxis='Predicted', yaxis='True', **kwargs)

    def report_figure(self, title, figure, step=None, series='figure'):
        if self._logger is None or figure is None:
            return
        self._logger.report_matplotlib_figure(
            title=title, series=series,
            iteration=self.step if step is None else step,
            figure=figure, report_image=False)

    def report_media(self, title, series, local_path, step=None):
        """Upload a media file (e.g. an SVG model graph) to ClearML as-is,
        preserving vector graphics."""
        if self._logger is None or not local_path:
            return
        self._logger.report_media(
            title=title, series=series,
            iteration=self.step if step is None else step,
            local_path=local_path)

    def report_single_value(self, name, value):
        """Log a single (iteration-independent) value. ClearML collects these in
        the experiment's SCALARS "Summary" and, crucially, renders them as a
        side-by-side **table** in COMPARE mode — the right home for a run's final
        metrics so they can be compared across experiments/folds."""
        if self._logger is None or value is None:
            return
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._logger.report_single_value(name=name, value=float(value))

    def flush(self):
        if self._logger is not None:
            self._logger.flush()


class MultiWriter(_RelativeStepMixin):
    """Fan-out writer that forwards every call to a list of underlying writers.

    Lets the training loops keep a single ``log_writer`` object while metrics go
    to both TensorBoard and ClearML. ``None`` writers are dropped, and optional
    methods (``update_image`` / ``flush``) are only called on writers that
    provide them.
    """

    def __init__(self, writers: List[Any]):
        self.writers = [w for w in writers if w is not None]
        self._init_relative_steps()

    def configure_relative_steps(self, total_steps=None, total_epochs=None, scale=1000):
        """Configure this writer and every underlying writer identically."""
        enabled = super().configure_relative_steps(
            total_steps=total_steps, total_epochs=total_epochs, scale=scale)
        for w in self.writers:
            if hasattr(w, 'configure_relative_steps'):
                w.configure_relative_steps(
                    total_steps=total_steps, total_epochs=total_epochs, scale=scale)
        return enabled

    def set_step(self, step=None):
        # Children receive the *absolute* step and map it themselves, so the
        # mapping is applied exactly once per writer.
        super().set_step(step)
        for w in self.writers:
            w.set_step(step)

    def update(self, head='scalar', step=None, **kwargs):
        for w in self.writers:
            w.update(head=head, step=step, **kwargs)

    def update_image(self, head='images', step=None, **kwargs):
        for w in self.writers:
            if hasattr(w, 'update_image'):
                w.update_image(head=head, step=step, **kwargs)

    def report_confusion_matrix(self, title, matrix, step=None, labels=None, series='confusion_matrix'):
        for w in self.writers:
            if hasattr(w, 'report_confusion_matrix'):
                w.report_confusion_matrix(title, matrix, step=step, labels=labels, series=series)

    def report_figure(self, title, figure, step=None, series='figure'):
        for w in self.writers:
            if hasattr(w, 'report_figure'):
                w.report_figure(title, figure, step=step, series=series)

    def report_media(self, title, series, local_path, step=None):
        for w in self.writers:
            if hasattr(w, 'report_media'):
                w.report_media(title, series, local_path, step=step)

    def report_single_value(self, name, value):
        for w in self.writers:
            if hasattr(w, 'report_single_value'):
                w.report_single_value(name, value)

    def flush(self):
        for w in self.writers:
            if hasattr(w, 'flush'):
                w.flush()
