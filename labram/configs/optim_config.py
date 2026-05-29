from dataclasses import dataclass
from typing import List, Optional

from labram.configs.base_configs import ConfigBase
from labram.configs.defaults import (
    DEFAULT_MIN_LR,
    DEFAULT_MOMENTUM,
    DEFAULT_OPT_EPS,
    DEFAULT_OPTIMIZER,
    DEFAULT_PRETRAIN_LR,
    DEFAULT_WARMUP_EPOCHS,
    DEFAULT_WARMUP_LR,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_WEIGHT_DECAY,
)


@dataclass
class OptimizerConfig(ConfigBase):
    opt: str = DEFAULT_OPTIMIZER
    opt_eps: float = DEFAULT_OPT_EPS
    opt_betas: Optional[List[float]] = None
    momentum: float = DEFAULT_MOMENTUM
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    weight_decay_end: Optional[float] = None
    lr: float = DEFAULT_PRETRAIN_LR
    warmup_lr: float = DEFAULT_WARMUP_LR
    min_lr: float = DEFAULT_MIN_LR
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    clip_grad: Optional[float] = None
