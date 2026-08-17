# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Helpers shared across the three runner scripts
# (run_class_finetuning, run_labram_pretraining, run_vqnsp_training).
#
# Each runner still owns its own argparse, model factory call, dataset
# preparation, and per-epoch logging shape. What lives here is the
# bookkeeping every runner duplicated: distributed init, device/seed,
# tensorboard wiring, list-of-dataloader construction, DDP wrap,
# auto-resume hook, train-log line, and the cosine schedules.
# --------------------------------------------------------

import datetime
import json
import os
import time
from argparse import Namespace
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

import labram.utils as utils
from labram.configs.optim_config import OptimizerConfig
from labram.configs.train_config import ClearMLConfig, DistributedConfig, OutputConfig, TrainerConfig

logger = utils.get_logger(__name__)

# ClearML tag applied to runs whose config has SageMaker submission enabled.
SAGEMAKER_TAG = 'sagemaker'


def setup_environment(config, init_cudnn_benchmark: bool = True) -> Tuple[torch.device, int, int]:
    """Initialize distributed, resolve device, seed, and cudnn flags.

    Accepts either a full RunConfig or any object with a ``.distributed``
    attribute of type DistributedConfig.  The runtime-computed ``distributed``
    and ``gpu`` fields are written back onto ``config.distributed`` so callers
    can access them without a separate return value.

    Returns (device, num_tasks, global_rank).
    """
    dist_cfg = config.distributed
    # Bridge to the legacy init_distributed_mode which mutates a Namespace.
    _ns = Namespace(
        dist_on_itp=dist_cfg.dist_on_itp,
        dist_url=dist_cfg.dist_url,
        world_size=dist_cfg.world_size,
        local_rank=dist_cfg.local_rank,
    )
    utils.init_distributed_mode(_ns)
    dist_cfg.distributed = getattr(_ns, 'distributed', False)
    dist_cfg.gpu = getattr(_ns, 'gpu', 0)

    device_str = getattr(_ns, 'device', dist_cfg.device)
    if device_str == 'auto':
        if torch.cuda.is_available():
            device_str = 'cuda'
        elif torch.backends.mps.is_available():
            device_str = 'mps'
        else:
            device_str = 'cpu'
    device = torch.device(device_str)
    dist_cfg.device = str(device)

    seed = dist_cfg.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if init_cudnn_benchmark and torch.cuda.is_available():
        cudnn.benchmark = True

    _configure_run_logging(config)

    return device, utils.get_world_size(), utils.get_rank()


def _configure_run_logging(config) -> None:
    """Attach the rank-aware ``labram`` logger. On rank 0 also tee to a file
    (``run.log``) under the run's log_dir/output_dir when either is set."""
    rank = utils.get_rank()
    log_file = None
    output_cfg = getattr(config, 'output', None)
    if rank == 0 and output_cfg is not None:
        base_dir = output_cfg.log_dir or output_cfg.output_dir
        if base_dir:
            log_file = os.path.join(base_dir, 'run.log')
    utils.configure_logging(rank=rank, log_file=log_file)


def flatten_config(d: Any, prefix: str = '') -> dict:
    """Flatten a nested config dict into dotted-key -> scalar pairs for ClearML
    hyperparameter (tabular) display. Lists/tuples are rendered as strings."""
    out = {}
    if not isinstance(d, dict):
        return {prefix.rstrip('/'): d}
    for key, value in d.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten_config(value, full + '/'))
        elif isinstance(value, (list, tuple)):
            out[full] = str(list(value))
        else:
            out[full] = value
    return out


def _clearml_task_url(task: Any) -> str:
    """Extract the web UI URL from a ClearML ``Task``, or return ``''``."""
    if task is None:
        return ''
    try:
        url = task.get_output_log_web_page()
        if url:
            return str(url)
    except Exception:
        pass
    try:
        task_id = task.id
        project_id = task.project
        if task_id:
            web_host = os.environ.get('CLEARML_WEB_HOST', 'https://app.clear.ml')
            return f"{web_host}/projects/{project_id or '*'}/experiments/{task_id}/output/log"
    except Exception:
        pass
    return ''


def _derive_clearml_task_name(run_config: Any) -> str:
    """Pick a stable, human-readable ClearML task name from the run config."""
    output_cfg = getattr(run_config, 'output', None)
    if output_cfg is not None and output_cfg.output_dir:
        name = os.path.basename(os.path.normpath(output_cfg.output_dir))
        if name:
            return name
    model_cfg = getattr(run_config, 'model', None)
    if model_cfg is not None and getattr(model_cfg, 'model', None):
        return str(model_cfg.model)
    return 'labram-run'


def _timestamp_ms() -> str:
    """Current local time as ``YYYYmmdd_HHMMSS_fff`` (millisecond precision)."""
    now = datetime.datetime.now()
    return now.strftime('%Y%m%d_%H%M%S') + f'_{now.microsecond // 1000:03d}'


def _clearml_default_output_uri() -> Optional[str]:
    """The ``sdk.development.default_output_uri`` from the ClearML config, if set."""
    try:
        from clearml.backend_config.config import Config
        cfg = Config()
        cfg.reload()
        return cfg.get('sdk.development.default_output_uri', None) or None
    except Exception:  # pragma: no cover - clearml absent/misconfigured -> no base
        return None


def _debug_output_uri(clearml_cfg: ClearMLConfig, project_name: str) -> Optional[str]:
    """Storage URI for debug runs: ``<base>/<project>/debug``.

    ``<base>`` is the run's ``clearml.output_uri`` when set, otherwise ClearML's
    configured ``default_output_uri`` (e.g. the S3 bucket in ``clearml.conf``).
    Returns ``None`` when no base is known, in which case ClearML falls back to
    its own default and only the ``debug`` tag distinguishes the run.
    """
    base = clearml_cfg.output_uri or _clearml_default_output_uri()
    if not base:
        return None
    return base.rstrip('/') + f'/{project_name}/debug'


def _sagemaker_enabled(run_config: Any) -> bool:
    """True when the run's config opts into SageMaker (``sagemaker.enabled``).

    Tolerates configs/namespaces without a ``sagemaker`` section (tests, legacy
    callers), which read as not enabled.
    """
    return bool(getattr(getattr(run_config, 'sagemaker', None), 'enabled', False))


def init_clearml_task(
    clearml_cfg: ClearMLConfig,
    run_config: Any,
    global_rank: int,
) -> Optional[Any]:
    """Initialize a ClearML ``Task`` on rank 0 when tracking is enabled.

    Returns the ``Task`` (or ``None`` when disabled, off-rank, or ClearML is not
    installed). ClearML is an optional dependency: a missing package downgrades
    to a warning and TensorBoard-only logging rather than failing the run.
    """
    if not clearml_cfg.enabled or global_rank != 0:
        return None
    try:
        from clearml import Task
    except ImportError:
        logger.warning(
            "clearml.enabled is set but the `clearml` package is not installed; "
            "continuing with TensorBoard-only logging. Run `pip install clearml` "
            "to enable ClearML experiment tracking.")
        return None

    if clearml_cfg.offline:
        Task.set_offline(offline_mode=True)

    project_name = clearml_cfg.project_name or 'LaBraM'
    task_name = clearml_cfg.task_name or _derive_clearml_task_name(run_config)
    # Uniquely identify each experiment: append a millisecond-precision timestamp
    # to the task name (e.g. 'finetune_tuab_base_20260718_143025_123').
    if getattr(clearml_cfg, 'append_timestamp', True):
        task_name = f"{task_name}_{_timestamp_ms()}"

    output_uri = clearml_cfg.output_uri or None
    tags = list(clearml_cfg.tags)
    is_debug = bool(getattr(run_config, 'debug', False))
    if is_debug:
        # Debug runs are tagged 'debug' and their artifacts are isolated in a
        # '<project>/debug' subfolder of the configured output storage, so they
        # never mix with real experiment outputs.
        if 'debug' not in tags:
            tags.append('debug')
        output_uri = _debug_output_uri(clearml_cfg, project_name)
    # Runs whose config has SageMaker submission enabled are tagged 'sagemaker'
    # so managed-training runs are filterable apart from local ones in ClearML.
    if _sagemaker_enabled(run_config) and SAGEMAKER_TAG not in tags:
        tags.append(SAGEMAKER_TAG)

    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        output_uri=output_uri,
        continue_last_task=clearml_cfg.continue_last_task,
        auto_connect_frameworks=clearml_cfg.auto_connect_frameworks,
    )
    if tags:
        task.add_tags(tags)
    # A SageMaker job runs from a source tarball with no .git, so ClearML cannot
    # auto-detect the code state. When the submitter shipped its git metadata,
    # replay it so branch/commit/uncommitted-diff still reach the experiment.
    utils.apply_git_info_to_task(task)
    if run_config is not None and hasattr(run_config, 'as_dict'):
        try:
            task.connect_configuration(run_config.as_dict(), name='run_config')
        except Exception as exc:  # pragma: no cover - defensive: never fail a run on tracking
            logger.warning("ClearML connect_configuration failed: %s", exc)
        # Also connect the config as flat dotted-key hyperparameters so it shows
        # as a searchable/sortable table in the ClearML experiment (not just a
        # JSON blob under Configuration).
        try:
            task.connect(flatten_config(run_config.as_dict()), name='config')
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ClearML connect (hyperparameters) failed: %s", exc)
    task_url = _clearml_task_url(task)
    logger.info(
        "ClearML tracking enabled: project=%r task=%r debug=%s output_uri=%r",
        project_name, task_name, is_debug, output_uri)
    if task_url:
        logger.info("ClearML task URL: %s", task_url)
    return task


def create_log_writer(
    output_cfg: OutputConfig,
    global_rank: int,
    clearml_cfg: Optional[ClearMLConfig] = None,
    run_config: Any = None,
) -> Optional[Any]:
    """Build the rank-0 metric writer: TensorBoard, ClearML, or both.

    Returns ``None`` on non-rank-0 processes or when no sink is configured. A
    single sink is returned bare; multiple sinks are combined in a
    :class:`~labram.utils.MultiWriter` so callers keep one ``log_writer``.
    """
    if global_rank != 0:
        return None

    writers: List[Any] = []
    if output_cfg.log_dir:
        os.makedirs(output_cfg.log_dir, exist_ok=True)
        writers.append(utils.TensorboardLogger(log_dir=output_cfg.log_dir))

    if clearml_cfg is not None and clearml_cfg.enabled:
        task = init_clearml_task(clearml_cfg, run_config, global_rank)
        if task is not None:
            writers.append(utils.ClearMLLogger(task=task))

    if not writers:
        return None
    if len(writers) == 1:
        return writers[0]
    return utils.MultiWriter(writers)


def configure_relative_step_axis(
    log_writer: Any,
    config: Any,
    num_training_steps_per_epoch: int,
    steps_per_logged_step: int = 1,
) -> bool:
    """Put ``log_writer`` on the normalized-progress x-axis for this run.

    Metrics are then plotted against training progress (0 ->
    ``logging.relative_step_scale`` over the whole run) instead of the raw
    global iteration, so runs with different dataset sizes, batch sizes or
    epoch counts overlay directly. ``steps_per_logged_step`` accounts for
    trainers that advance the writer once per micro-batch while
    ``num_training_steps_per_epoch`` counts optimizer steps (fine-tuning's
    ``update_freq``).

    No-op (absolute axis kept) when there is no writer, when the run has no
    logging config, or when ``logging.relative_step_axis`` is off. Returns
    whether the relative axis is active.
    """
    logging_cfg = getattr(config, 'logging', None)
    if log_writer is None or logging_cfg is None:
        return False
    if not getattr(logging_cfg, 'relative_step_axis', False):
        return False
    if not hasattr(log_writer, 'configure_relative_steps'):
        return False

    epochs = config.trainer.epochs
    total_steps = int(num_training_steps_per_epoch) * int(epochs) * int(steps_per_logged_step)
    enabled = log_writer.configure_relative_steps(
        total_steps=total_steps,
        total_epochs=epochs,
        scale=logging_cfg.relative_step_scale,
    )
    if enabled:
        logger.info(
            "Metrics use the relative x-axis: %d step(s) mapped onto 0..%d "
            "(training progress).", total_steps, logging_cfg.relative_step_scale)
    return enabled


def build_distributed_train_sampler_list(
    datasets: Sequence[torch.utils.data.Dataset],
    num_tasks: int,
    rank: int,
) -> List[torch.utils.data.DistributedSampler]:
    """One shuffled DistributedSampler per training dataset."""
    return [
        torch.utils.data.DistributedSampler(
            d, num_replicas=num_tasks, rank=rank, shuffle=True,
        )
        for d in datasets
    ]


def build_distributed_eval_sampler_list(
    datasets: Sequence[torch.utils.data.Dataset],
    num_tasks: int,
    rank: int,
    dist_eval: bool,
) -> List[torch.utils.data.Sampler]:
    """DistributedSampler (shuffle=False) when dist_eval else SequentialSampler."""
    if dist_eval:
        return [
            torch.utils.data.DistributedSampler(
                d, num_replicas=num_tasks, rank=rank, shuffle=False,
            )
            for d in datasets
        ]
    return [torch.utils.data.SequentialSampler(d) for d in datasets]


def build_dataloader_list(
    datasets: Sequence[torch.utils.data.Dataset],
    samplers: Sequence[torch.utils.data.Sampler],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> List[torch.utils.data.DataLoader]:
    """Pair each (dataset, sampler) into a DataLoader with shared per-loader settings."""
    return [
        torch.utils.data.DataLoader(
            d,
            sampler=s,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        for d, s in zip(datasets, samplers)
    ]


def wrap_distributed(dist_cfg: DistributedConfig, model: torch.nn.Module) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """DDP-wrap model when dist_cfg.distributed is true. Returns (model, model_without_ddp)."""
    if dist_cfg.distributed:
        wrapped = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist_cfg.gpu], find_unused_parameters=True,
        )
        return wrapped, wrapped.module
    return model, model


def make_lr_schedule(optim_cfg: OptimizerConfig, trainer_cfg: TrainerConfig, num_training_steps_per_epoch: int) -> np.ndarray:
    """LR schedule (cosine/step/multistep/linear/constant) with optional warmup.

    The policy is selected by ``optim_cfg.sched`` (default 'cosine', which
    reproduces the historical behaviour). Common to all runs.
    """
    return utils.build_lr_schedule(
        optim_cfg.sched, optim_cfg.lr, optim_cfg.min_lr,
        trainer_cfg.epochs, num_training_steps_per_epoch,
        warmup_epochs=optim_cfg.warmup_epochs, warmup_steps=optim_cfg.warmup_steps,
        decay_epochs=optim_cfg.decay_epochs, decay_rate=optim_cfg.decay_rate,
        decay_milestones=optim_cfg.decay_milestones,
    )


def make_wd_schedule(optim_cfg: OptimizerConfig, trainer_cfg: TrainerConfig, num_training_steps_per_epoch: int) -> np.ndarray:
    """Cosine WD schedule. If weight_decay_end is None, decay stays flat at weight_decay."""
    wd_end = optim_cfg.weight_decay_end if optim_cfg.weight_decay_end is not None else optim_cfg.weight_decay
    return utils.cosine_scheduler(
        optim_cfg.weight_decay, wd_end, trainer_cfg.epochs, num_training_steps_per_epoch,
    )


def append_log_line(output_cfg: OutputConfig, log_stats: dict) -> None:
    """Append a single JSON line to output_cfg.output_dir/log.txt (main process only)."""
    if output_cfg.output_dir and utils.is_main_process():
        with open(os.path.join(output_cfg.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")


def print_training_time(start_time: float) -> None:
    """Log elapsed wall time as HH:MM:SS."""
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str}')


def log_model_visualization(logging_cfg: Any, output_cfg: OutputConfig,
                            model: torch.nn.Module, log_writer: Any = None) -> None:
    """Render the model architecture coloured by frozen/trainable layers and log
    it: a matplotlib figure to TensorBoard, the vector (SVG) file to ClearML as
    media, and the graph files (SVG/PNG + DOT source) as ClearML artifacts."""
    if logging_cfg is None or not getattr(logging_cfg, 'log_model_graph', False):
        return
    if not utils.is_main_process():
        return
    from labram.utils.model_graph import save_model_graph
    from labram.utils.clearml_artifacts import get_clearml_task, upload_file_artifact

    out_dir = output_cfg.output_dir or output_cfg.log_dir
    try:
        graph = save_model_graph(model, out_dir, fmt=getattr(logging_cfg, 'model_graph_format', 'svg'))
    except Exception as exc:  # pragma: no cover - never fail a run on a diagram
        logger.warning("Model graph rendering failed: %s", exc)
        return

    if log_writer is not None and graph.get('figure') is not None:
        log_writer.report_figure('model', graph['figure'], series='architecture')
    if log_writer is not None and graph.get('image_path') and hasattr(log_writer, 'report_media'):
        log_writer.report_media('model', 'architecture', graph['image_path'])

    task = get_clearml_task(log_writer)
    if task is not None:
        for key, name in (('image_path', 'model_graph'), ('dot_path', 'model_graph_dot')):
            if graph.get(key):
                upload_file_artifact(task, name, graph[key])


def log_data_split_artifact(logging_cfg: Any, output_cfg: OutputConfig,
                            dataset_train, dataset_val, dataset_test,
                            log_writer: Any = None, dataset_name: Optional[str] = None) -> Optional[str]:
    """Record the train/val/test case assignment to JSON and upload it as a
    ClearML artifact. Returns the JSON path (rank 0), else None."""
    if logging_cfg is None or not getattr(logging_cfg, 'log_data_split', False):
        return None
    if not utils.is_main_process():
        return None
    from labram.utils.data_split import build_data_split, save_data_split
    from labram.utils.clearml_artifacts import get_clearml_task, upload_file_artifact

    out_dir = output_cfg.output_dir or output_cfg.log_dir
    try:
        split = build_data_split(dataset_train, dataset_val, dataset_test, dataset_name)
        path = save_data_split(split, out_dir)
    except Exception as exc:  # pragma: no cover
        logger.warning("Data-split recording failed: %s", exc)
        return None
    if path is not None:
        task = get_clearml_task(log_writer)
        upload_file_artifact(task, 'data_split', path)
    return path


# Top-level metric keys an eval-only regression run reports.
REGRESSION_SUMMARY_KEYS = ('mae', 'rmse', 'r2', 'pearson_r', 'mae_corrected',
                           'age_bias_slope')


def flatten_summary_metrics(summary: Any) -> dict:
    """Flatten a training ``summary`` (from ``train_loop`` / an eval-only run)
    into a flat ``{name: float}`` of the final/best metrics, e.g.
    ``{'best_epoch': 12, 'val_accuracy': 81.3, 'test_accuracy': 79.0,
    'test_balanced_accuracy': 0.78, ...}``. Non-numeric fields are dropped."""
    flat: dict = {}
    if not isinstance(summary, dict):
        return flat

    def _is_num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    if _is_num(summary.get('best_epoch')) and summary['best_epoch'] >= 0:
        flat['best_epoch'] = summary['best_epoch']
    for key, prefix in (('best_val_stats', 'val'), ('best_test_stats', 'test')):
        for k, v in (summary.get(key) or {}).items():
            if _is_num(v):
                flat[f'{prefix}_{k}'] = v
    # Eval-only runs return their metrics at the top level: accuracy/balanced
    # accuracy for classification, the regression metrics for a scalar target.
    for k in ('accuracy', 'balanced_accuracy') + REGRESSION_SUMMARY_KEYS:
        if _is_num(summary.get(k)):
            flat.setdefault(f'test_{k}', summary[k])
    return flat


def log_summary_metrics(log_writer: Any, summary: Any, config: Any = None,
                        section: str = 'final_metrics') -> dict:
    """Record a run's final/best eval metrics so they are comparable across
    experiments in ClearML **compare** mode.

    Reports each metric as a ClearML *single value* (which renders as a
    side-by-side table when experiments are compared) and also connects the flat
    metric dict to the task as a ``final_metrics`` hyperparameter section, so the
    values additionally show up as sortable columns in the experiments table and
    the hyperparameter comparison. Rank-0 only; returns the flat metric dict."""
    if not utils.is_main_process() or log_writer is None:
        return {}
    from labram.utils.clearml_artifacts import get_clearml_task

    flat = flatten_summary_metrics(summary)
    if not flat:
        return {}
    for name, value in flat.items():
        if hasattr(log_writer, 'report_single_value'):
            try:
                log_writer.report_single_value(name, value)
            except Exception as exc:  # pragma: no cover - never fail a run on logging
                logger.warning("report_single_value(%s) failed: %s", name, exc)
    task = get_clearml_task(log_writer)
    if task is not None:
        try:
            # str-cast values so ClearML records them as a stable config section.
            task.connect({k: v for k, v in flat.items()}, name=section)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ClearML connect(%s) failed: %s", section, exc)
    logger.info("Logged %d final metric(s) for comparison: %s",
                len(flat), sorted(flat))
    return flat


def log_cv_split_artifact(output_cfg: OutputConfig, log_writer: Any = None,
                          filename: str = 'cv_split.json') -> Optional[str]:
    """Upload the cross-validation fold partition (``cv_split.json``, written to
    the fold's output dir by the CV runner) as a ClearML artifact so every fold
    experiment records the exact partition it belongs to. Rank-0 only; returns
    the path if found, else None."""
    if not utils.is_main_process():
        return None
    out_dir = output_cfg.output_dir or output_cfg.log_dir
    if not out_dir:
        return None
    path = os.path.join(out_dir, filename)
    if not os.path.exists(path):
        return None
    from labram.utils.clearml_artifacts import get_clearml_task, upload_file_artifact
    task = get_clearml_task(log_writer)
    upload_file_artifact(task, 'cv_split', path)
    return path


def finalize_run(config: Any, log_writer: Any = None) -> None:
    """Post-training hook (rank 0): publish the model + optionally stop the box.

    1. When ClearML tracking is on and ``clearml.upload_model_artifact`` is set,
       explicitly register the final/best checkpoint as a ClearML ``OutputModel``
       so it is uploaded to ``clearml.output_uri`` (e.g. S3) rather than relying
       only on framework auto-capture.
    2. When ``shutdown.stop_instance_on_finish`` is set, stop the machine after
       ``shutdown.stop_delay_minutes``.

    Both steps are best-effort and never raise, so finishing touches can't turn a
    successful run into a failure.
    """
    if not utils.is_main_process():
        return

    from labram.utils.clearml_artifacts import (
        finalize_clearml_task, get_clearml_task, resolve_final_checkpoint,
        upload_model_artifact,
    )
    from labram.utils.shutdown import maybe_stop_instance

    clearml_cfg = getattr(config, 'clearml', None)
    output_cfg = getattr(config, 'output', None)
    clearml_on = clearml_cfg is not None and clearml_cfg.enabled
    if (clearml_on and getattr(clearml_cfg, 'upload_model_artifact', False)
            and output_cfg is not None):
        task = get_clearml_task(log_writer)
        model_path = resolve_final_checkpoint(output_cfg.output_dir)
        if task is not None and model_path is not None:
            name = clearml_cfg.artifact_name or None
            upload_model_artifact(task, model_path, name=name)
        elif task is not None:
            logger.warning("ClearML upload requested but no checkpoint found in %r",
                           getattr(output_cfg, 'output_dir', None))

    if clearml_on:
        task = get_clearml_task(log_writer)
        task_url = _clearml_task_url(task)
        if task_url:
            logger.info("ClearML results: %s", task_url)

    # If we're about to stop the machine, finish the ClearML task first so its
    # final state and pending uploads are persisted before the box is killed —
    # otherwise the abrupt stop can leave the task stuck "running" or lose the
    # last metrics/artifacts.
    shutdown_cfg = getattr(config, 'shutdown', None)
    will_stop = shutdown_cfg is not None and getattr(shutdown_cfg, 'stop_instance_on_finish', False)
    if will_stop and clearml_on:
        finalize_clearml_task(get_clearml_task(log_writer))

    try:
        maybe_stop_instance(shutdown_cfg)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("maybe_stop_instance failed: %s", exc)
