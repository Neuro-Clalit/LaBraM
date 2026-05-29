"""Top-level run configurations for LaBraM training scripts.

Each `*RunConfig` composes the shared sub-configs with phase-specific
model/tokenizer settings. Sub-config types live in their own modules:

    labram.configs.data_config    — DataConfig
    labram.configs.model_config   — PretrainModelConfig, VQNSPModelConfig,
                                    FinetuneModelConfig, TokenizerConfig,
                                    FinetuneCheckpointConfig
    labram.configs.optim_config   — OptimizerConfig
    labram.configs.train_config   — TrainerConfig, OutputConfig, DistributedConfig

``RunConfig.to_namespace()`` flattens every leaf field into an
``argparse.Namespace`` for legacy consumers (``init_distributed_mode``,
``create_optimizer``, ``auto_load_model``, ``save_model``) that read a
flat ``args.X`` surface. Mutations to the namespace are isolated from the
typed config.
"""

from argparse import Namespace
from dataclasses import dataclass, field
from typing import List

from labram.configs.base_configs import ConfigBase
from labram.configs.data_config import DataConfig
from labram.configs.defaults import (
    DEFAULT_FINETUNE_DATA_PATH,
    DEFAULT_FINETUNE_DATASET,
    DEFAULT_FINETUNE_DEBUG,
    DEFAULT_FINETUNE_DEBUG_SAMPLES,
    DEFAULT_FINETUNE_DEVICE,
    DEFAULT_FINETUNE_DISABLE_EVAL,
    DEFAULT_FINETUNE_DISABLE_WD_ON_REL_POS_BIAS,
    DEFAULT_FINETUNE_ENABLE_DEEPSPEED,
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_FINETUNE_LABEL_SMOOTHING,
    DEFAULT_FINETUNE_LAYER_DECAY,
    DEFAULT_FINETUNE_LR,
    DEFAULT_FINETUNE_MIN_LR,
    DEFAULT_FINETUNE_MODEL_EMA,
    DEFAULT_FINETUNE_MODEL_EMA_DECAY,
    DEFAULT_FINETUNE_MODEL_EMA_FORCE_CPU,
    DEFAULT_FINETUNE_ROBUST_TEST,
    DEFAULT_FINETUNE_SAVE_CKPT_FREQ,
    DEFAULT_VQNSP_DIST_EVAL,
    DEFAULT_VQNSP_EPOCHS,
    DEFAULT_VQNSP_LR,
    DEFAULT_VQNSP_SAVE_CKPT_FREQ,
    DEFAULT_VQNSP_STRIDE,
    DEFAULT_VQNSP_WEIGHT_DECAY,
)
from labram.configs.model_config import (
    FinetuneCheckpointConfig,
    FinetuneModelConfig,
    PretrainModelConfig,
    TokenizerConfig,
    VQNSPModelConfig,
)
from labram.configs.optim_config import OptimizerConfig
from labram.configs.train_config import DistributedConfig, OutputConfig, TrainerConfig


# ============================================================
# Base run config
# ============================================================


@dataclass
class RunConfig(ConfigBase):
    """Base for any single-phase training run. Subclasses inject the
    phase-specific ``model`` (and any extras like ``tokenizer``)."""

    output: OutputConfig = field(default_factory=OutputConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_namespace(self) -> Namespace:
        """Flatten every leaf field into a single ``argparse.Namespace``.

        Legacy consumers (``init_distributed_mode``, ``create_optimizer``,
        ``auto_load_model``, ``save_model``) read a flat ``args.X`` surface
        and sometimes mutate it. We give them a separate Namespace so those
        mutations (rank, gpu, distributed, resume) don't leak back into the
        typed config.
        """
        ns = Namespace()
        for sub_name, sub in self.__dict__.items():
            if isinstance(sub, ConfigBase):
                for k, v in sub.__dict__.items():
                    setattr(ns, k, v)
            else:
                setattr(ns, sub_name, sub)
        return ns


# ============================================================
# Phase-specific run configs
# ============================================================


@dataclass
class PretrainRunConfig(RunConfig):
    model: PretrainModelConfig = field(default_factory=PretrainModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)


@dataclass
class VQNSPRunConfig(RunConfig):
    model: VQNSPModelConfig = field(default_factory=VQNSPModelConfig)
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            lr=DEFAULT_VQNSP_LR,
            weight_decay=DEFAULT_VQNSP_WEIGHT_DECAY,
        ),
    )
    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(epochs=DEFAULT_VQNSP_EPOCHS),
    )
    distributed: DistributedConfig = field(
        default_factory=lambda: DistributedConfig(dist_eval=DEFAULT_VQNSP_DIST_EVAL),
    )
    data: DataConfig = field(
        default_factory=lambda: DataConfig(stride=DEFAULT_VQNSP_STRIDE),
    )
    output: OutputConfig = field(
        default_factory=lambda: OutputConfig(save_ckpt_freq=DEFAULT_VQNSP_SAVE_CKPT_FREQ),
    )
    disable_eval: bool = False
    eval: bool = False
    calculate_codebook_usage: bool = False


@dataclass
class FinetuneRunConfig(RunConfig):
    model: FinetuneModelConfig = field(default_factory=FinetuneModelConfig)
    finetune_checkpoint: FinetuneCheckpointConfig = field(default_factory=FinetuneCheckpointConfig)
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            lr=DEFAULT_FINETUNE_LR,
            min_lr=DEFAULT_FINETUNE_MIN_LR,
        ),
    )
    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(epochs=DEFAULT_FINETUNE_EPOCHS),
    )
    distributed: DistributedConfig = field(
        default_factory=lambda: DistributedConfig(device=DEFAULT_FINETUNE_DEVICE),
    )
    output: OutputConfig = field(
        default_factory=lambda: OutputConfig(save_ckpt_freq=DEFAULT_FINETUNE_SAVE_CKPT_FREQ),
    )
    layer_decay: float = DEFAULT_FINETUNE_LAYER_DECAY
    smoothing: float = DEFAULT_FINETUNE_LABEL_SMOOTHING
    model_ema: bool = DEFAULT_FINETUNE_MODEL_EMA
    model_ema_decay: float = DEFAULT_FINETUNE_MODEL_EMA_DECAY
    model_ema_force_cpu: bool = DEFAULT_FINETUNE_MODEL_EMA_FORCE_CPU
    disable_eval_during_finetuning: bool = DEFAULT_FINETUNE_DISABLE_EVAL
    disable_weight_decay_on_rel_pos_bias: bool = DEFAULT_FINETUNE_DISABLE_WD_ON_REL_POS_BIAS
    eval: bool = False
    enable_deepspeed: bool = DEFAULT_FINETUNE_ENABLE_DEEPSPEED
    dataset: str = DEFAULT_FINETUNE_DATASET
    data_path: str = DEFAULT_FINETUNE_DATA_PATH
    debug: bool = DEFAULT_FINETUNE_DEBUG
    debug_samples: int = DEFAULT_FINETUNE_DEBUG_SAMPLES
    robust_test: str = DEFAULT_FINETUNE_ROBUST_TEST


# ============================================================
# CLI override parsing (shared by all run scripts)
# ============================================================


def _coerce_override(s: str):
    """Map a CLI override string to a Python literal.

    Order: None → bool → int → float → str.
    """
    if s.lower() == 'none':
        return None
    if s.lower() == 'true':
        return True
    if s.lower() == 'false':
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def parse_overrides(items: List[str]) -> dict:
    """Parse ``--set key.sub=value`` strings into a dict for
    :meth:`ConfigBase.update`."""
    out: dict = {}
    for raw in items:
        if '=' not in raw:
            raise ValueError(f'Override must be key=value: {raw!r}')
        key, value = raw.split('=', 1)
        out[key.strip()] = _coerce_override(value.strip())
    return out
