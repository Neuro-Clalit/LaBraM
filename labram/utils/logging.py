# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# SmoothedValue, MetricLogger, TensorboardLogger, ClearMLLogger, MultiWriter,
# and the shared Python-logging setup.
# ---------------------------------------------------------

import datetime
import logging
import os
import sys
import time
from collections import defaultdict, deque
from typing import Any, List, Optional

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


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

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

    def flush(self):
        self.writer.flush()


class ClearMLLogger(object):
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
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

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

    def flush(self):
        if self._logger is not None:
            self._logger.flush()


class MultiWriter(object):
    """Fan-out writer that forwards every call to a list of underlying writers.

    Lets the training loops keep a single ``log_writer`` object while metrics go
    to both TensorBoard and ClearML. ``None`` writers are dropped, and optional
    methods (``update_image`` / ``flush``) are only called on writers that
    provide them.
    """

    def __init__(self, writers: List[Any]):
        self.writers = [w for w in writers if w is not None]
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1
        for w in self.writers:
            w.set_step(step)

    def update(self, head='scalar', step=None, **kwargs):
        for w in self.writers:
            w.update(head=head, step=step, **kwargs)

    def update_image(self, head='images', step=None, **kwargs):
        for w in self.writers:
            if hasattr(w, 'update_image'):
                w.update_image(head=head, step=step, **kwargs)

    def flush(self):
        for w in self.writers:
            if hasattr(w, 'flush'):
                w.flush()
