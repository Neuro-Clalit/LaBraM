from dataclasses import dataclass, field
from typing import List

from labram.configs.base_configs import ConfigBase
from labram.configs.defaults import (
    DEFAULT_AUTO_RESUME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLEARML_ARTIFACT_NAME,
    DEFAULT_CLEARML_AUTO_CONNECT_FRAMEWORKS,
    DEFAULT_CLEARML_CONTINUE_LAST_TASK,
    DEFAULT_CLEARML_ENABLED,
    DEFAULT_CLEARML_OFFLINE,
    DEFAULT_CLEARML_OUTPUT_URI,
    DEFAULT_CLEARML_PROJECT_NAME,
    DEFAULT_CLEARML_TASK_NAME,
    DEFAULT_CLEARML_UPLOAD_MODEL_ARTIFACT,
    DEFAULT_DEVICE,
    DEFAULT_DIST_EVAL,
    DEFAULT_DIST_ON_ITP,
    DEFAULT_DIST_URL,
    DEFAULT_EVAL_AGG_CASE_BY,
    DEFAULT_EVAL_AGG_WINDOWS,
    DEFAULT_EVAL_DETAILED_METRICS,
    DEFAULT_EVAL_LOG_CONFUSION_MATRIX,
    DEFAULT_EVAL_LOG_CURVES,
    DEFAULT_EVAL_LOG_GRAD_COMPONENTS,
    DEFAULT_EVAL_LOG_GRAD_FREQ,
    DEFAULT_EVAL_PLOT_PER_EPOCH,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LOCAL_RANK,
    DEFAULT_LOG_DATA_SPLIT,
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_MODEL_GRAPH,
    DEFAULT_MODEL_GRAPH_FORMAT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRETRAIN_EPOCHS,
    DEFAULT_PRETRAIN_SAVE_CKPT_FREQ,
    DEFAULT_RESUME,
    DEFAULT_SAVE_CKPT,
    DEFAULT_SAVE_ONLY_FINAL_MODEL,
    DEFAULT_SEED,
    DEFAULT_START_EPOCH,
    DEFAULT_STOP_DELAY_MINUTES,
    DEFAULT_STOP_INSTANCE_ON_FINISH,
    DEFAULT_STOP_METHOD,
    DEFAULT_UPDATE_FREQ,
    DEFAULT_WORLD_SIZE,
)


@dataclass
class TrainerConfig(ConfigBase):
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_PRETRAIN_EPOCHS
    start_epoch: int = DEFAULT_START_EPOCH
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    update_freq: int = DEFAULT_UPDATE_FREQ
    disable_eval_during_finetuning: bool = False
    debug: bool = False
    disable_weight_decay_on_rel_pos_bias: bool = False
    eval: bool = False
    enable_deepspeed: bool = False
    debug_samples: int = 16


@dataclass
class OutputConfig(ConfigBase):
    output_dir: str = DEFAULT_OUTPUT_DIR
    log_dir: str = DEFAULT_LOG_DIR
    resume: str = DEFAULT_RESUME
    auto_resume: bool = DEFAULT_AUTO_RESUME
    save_ckpt: bool = DEFAULT_SAVE_CKPT
    save_ckpt_freq: int = DEFAULT_PRETRAIN_SAVE_CKPT_FREQ
    # Skip periodic/rolling per-epoch checkpoints; save only the final model.
    save_only_final_model: bool = DEFAULT_SAVE_ONLY_FINAL_MODEL


@dataclass
class ClearMLConfig(ConfigBase):
    """ClearML experiment-tracking settings (opt-in via ``enabled``).

    When enabled, rank 0 initializes a ClearML ``Task`` and metrics are mirrored
    there alongside TensorBoard. ClearML is an optional dependency; if it is not
    installed the run continues with a warning and TensorBoard-only logging.
    """
    enabled: bool = DEFAULT_CLEARML_ENABLED
    project_name: str = DEFAULT_CLEARML_PROJECT_NAME
    task_name: str = DEFAULT_CLEARML_TASK_NAME
    tags: List[str] = field(default_factory=list)
    output_uri: str = DEFAULT_CLEARML_OUTPUT_URI
    offline: bool = DEFAULT_CLEARML_OFFLINE
    continue_last_task: bool = DEFAULT_CLEARML_CONTINUE_LAST_TASK
    auto_connect_frameworks: bool = DEFAULT_CLEARML_AUTO_CONNECT_FRAMEWORKS
    # When set, the final/best checkpoint is explicitly registered as a ClearML
    # OutputModel at the end of training so it is uploaded to ``output_uri``
    # (e.g. an S3 bucket) rather than relying solely on framework auto-capture.
    upload_model_artifact: bool = DEFAULT_CLEARML_UPLOAD_MODEL_ARTIFACT
    artifact_name: str = DEFAULT_CLEARML_ARTIFACT_NAME


@dataclass
class LoggingConfig(ConfigBase):
    """Logging / experiment-artifact options that are independent of the metric
    backend.

    Groups the "what to log" toggles for the run's diagnostic artifacts:
    the model graph visualization (colored by frozen/trainable layers) and the
    train/val/test data-split record. The metric backend itself (TensorBoard /
    ClearML) is configured separately in :class:`ClearMLConfig` / ``output.log_dir``.
    """
    log_model_graph: bool = DEFAULT_LOG_MODEL_GRAPH
    model_graph_format: str = DEFAULT_MODEL_GRAPH_FORMAT  # svg (vector) | png
    log_data_split: bool = DEFAULT_LOG_DATA_SPLIT


@dataclass
class ShutdownConfig(ConfigBase):
    """Optionally stop the (EC2) machine after training finishes.

    When ``stop_instance_on_finish`` is set, the rank-0 process waits
    ``stop_delay_minutes`` (leaving a window to cancel / inspect) and then stops
    the instance. ``stop_method`` selects how: ``ec2`` issues a boto3
    ``stop_instances`` call for the instance's own id (read from EC2 IMDS);
    ``os`` runs ``shutdown -h`` locally. See ``labram/utils/shutdown.py``.
    """
    stop_instance_on_finish: bool = DEFAULT_STOP_INSTANCE_ON_FINISH
    stop_delay_minutes: int = DEFAULT_STOP_DELAY_MINUTES
    stop_method: str = DEFAULT_STOP_METHOD


@dataclass
class EvaluationConfig(ConfigBase):
    """Detailed evaluation metrics + inference-time window aggregation.

    ``detailed_metrics`` turns on the richer classification report (confusion
    matrix, F1, sensitivity/specificity, ROC/PR AUC) for train/val/test;
    ``log_confusion_matrix`` / ``log_curves`` additionally push the matrix and
    ROC/PR curves to TensorBoard / ClearML. ``plot_per_epoch`` gives each
    epoch's figure its own plot series so all epochs are retained (otherwise
    only the latest epoch is shown). ``log_grad_components`` logs the
    per-loss-component gradient norms every ``log_grad_freq`` steps.
    ``agg_windows`` selects how per-window predictions are pooled into a single
    per-case prediction during eval-only (inference) runs, and ``agg_case_by``
    whether a "case" is a recording/session or a subject.
    """
    detailed_metrics: bool = DEFAULT_EVAL_DETAILED_METRICS
    log_confusion_matrix: bool = DEFAULT_EVAL_LOG_CONFUSION_MATRIX
    log_curves: bool = DEFAULT_EVAL_LOG_CURVES
    plot_per_epoch: bool = DEFAULT_EVAL_PLOT_PER_EPOCH
    log_grad_components: bool = DEFAULT_EVAL_LOG_GRAD_COMPONENTS
    log_grad_freq: int = DEFAULT_EVAL_LOG_GRAD_FREQ
    agg_windows: str = DEFAULT_EVAL_AGG_WINDOWS
    agg_case_by: str = DEFAULT_EVAL_AGG_CASE_BY


@dataclass
class DistributedConfig(ConfigBase):
    world_size: int = DEFAULT_WORLD_SIZE
    local_rank: int = DEFAULT_LOCAL_RANK
    dist_on_itp: bool = DEFAULT_DIST_ON_ITP
    dist_url: str = DEFAULT_DIST_URL
    device: str = DEFAULT_DEVICE
    seed: int = DEFAULT_SEED
    dist_eval: bool = DEFAULT_DIST_EVAL
    # Runtime fields — set by setup_environment after init_distributed_mode runs.
    distributed: bool = False
    gpu: int = 0
