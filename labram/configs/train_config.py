from dataclasses import dataclass, field
from typing import List

from labram.configs.base_configs import ConfigBase
from labram.configs.defaults import (
    DEFAULT_AUTO_RESUME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLEARML_AUTO_CONNECT_FRAMEWORKS,
    DEFAULT_CLEARML_CONTINUE_LAST_TASK,
    DEFAULT_CLEARML_ENABLED,
    DEFAULT_CLEARML_OFFLINE,
    DEFAULT_CLEARML_OUTPUT_URI,
    DEFAULT_CLEARML_PROJECT_NAME,
    DEFAULT_CLEARML_TASK_NAME,
    DEFAULT_DEVICE,
    DEFAULT_DIST_EVAL,
    DEFAULT_DIST_ON_ITP,
    DEFAULT_DIST_URL,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LOCAL_RANK,
    DEFAULT_LOG_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRETRAIN_EPOCHS,
    DEFAULT_PRETRAIN_SAVE_CKPT_FREQ,
    DEFAULT_RESUME,
    DEFAULT_SAVE_CKPT,
    DEFAULT_SEED,
    DEFAULT_START_EPOCH,
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


@dataclass
class OutputConfig(ConfigBase):
    output_dir: str = DEFAULT_OUTPUT_DIR
    log_dir: str = DEFAULT_LOG_DIR
    resume: str = DEFAULT_RESUME
    auto_resume: bool = DEFAULT_AUTO_RESUME
    save_ckpt: bool = DEFAULT_SAVE_CKPT
    save_ckpt_freq: int = DEFAULT_PRETRAIN_SAVE_CKPT_FREQ


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
