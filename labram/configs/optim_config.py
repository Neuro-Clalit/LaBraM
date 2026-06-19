from dataclasses import dataclass
from typing import List, Optional

from labram.configs.base_configs import ConfigBase
from labram.configs import defaults as conf_consts


@dataclass
class OptimizerConfig(ConfigBase):
    opt: str = conf_consts.DEFAULT_OPTIMIZER
    opt_eps: float = conf_consts.DEFAULT_OPT_EPS
    opt_betas: Optional[List[float]] = None
    momentum: float = conf_consts.DEFAULT_MOMENTUM
    weight_decay: float = conf_consts.DEFAULT_WEIGHT_DECAY
    weight_decay_end: Optional[float] = None
    lr: float = conf_consts.DEFAULT_PRETRAIN_LR
    warmup_lr: float = conf_consts.DEFAULT_WARMUP_LR
    min_lr: float = conf_consts.DEFAULT_MIN_LR
    warmup_epochs: int = conf_consts.DEFAULT_WARMUP_EPOCHS
    warmup_steps: int = conf_consts.DEFAULT_WARMUP_STEPS
    clip_grad: Optional[float] = None
