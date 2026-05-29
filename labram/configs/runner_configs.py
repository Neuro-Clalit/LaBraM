"""Typed, nested run configurations for LaBraM training scripts.

Top-level `*RunConfig` subclasses contain `model`, `optimizer`, `trainer`,
`data`, `output`, and `distributed` member configs. Field defaults are
sourced from :mod:`labram.configs.defaults` so every literal that used to
live in a `parser.add_argument(default=...)` call has exactly one home.

A run config loaded from JSON/YAML can be flattened to an
``argparse.Namespace`` via :meth:`RunConfig.to_namespace`. The flat
namespace is what legacy consumers (``init_distributed_mode``,
``create_optimizer``, ``auto_load_model``, ``save_model``) read from —
the typed config remains the user-facing surface.
"""

from argparse import Namespace
from dataclasses import dataclass, field
from typing import List, Optional

from labram.configs.base_configs import ConfigBase
from labram.configs.defaults import (
    DEFAULT_ATTN_DROP_RATE,
    DEFAULT_AUTO_RESUME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CODEBOOK_SIZE,
    DEFAULT_DATASET_END_PERCENTAGE,
    DEFAULT_DATASET_START_PERCENTAGE,
    DEFAULT_DEVICE,
    DEFAULT_DIST_EVAL,
    DEFAULT_DIST_ON_ITP,
    DEFAULT_DIST_URL,
    DEFAULT_DROP,
    DEFAULT_DROP_PATH,
    DEFAULT_FINETUNE_ABS_POS_EMB,
    DEFAULT_FINETUNE_CHECKPOINT,
    DEFAULT_FINETUNE_DATASET,
    DEFAULT_FINETUNE_DATA_PATH,
    DEFAULT_FINETUNE_DEBUG,
    DEFAULT_FINETUNE_DEBUG_SAMPLES,
    DEFAULT_FINETUNE_DEVICE,
    DEFAULT_FINETUNE_DISABLE_EVAL,
    DEFAULT_FINETUNE_DISABLE_WD_ON_REL_POS_BIAS,
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_FINETUNE_ENABLE_DEEPSPEED,
    DEFAULT_FINETUNE_INIT_SCALE,
    DEFAULT_FINETUNE_INPUT_SIZE,
    DEFAULT_FINETUNE_LABEL_SMOOTHING,
    DEFAULT_FINETUNE_LAYER_DECAY,
    DEFAULT_FINETUNE_LR,
    DEFAULT_FINETUNE_MIN_LR,
    DEFAULT_FINETUNE_MODEL,
    DEFAULT_FINETUNE_MODEL_EMA,
    DEFAULT_FINETUNE_MODEL_EMA_DECAY,
    DEFAULT_FINETUNE_MODEL_EMA_FORCE_CPU,
    DEFAULT_FINETUNE_MODEL_FILTER_NAME,
    DEFAULT_FINETUNE_MODEL_KEY,
    DEFAULT_FINETUNE_MODEL_PREFIX,
    DEFAULT_FINETUNE_NB_CLASSES,
    DEFAULT_FINETUNE_QKV_BIAS,
    DEFAULT_FINETUNE_REL_POS_BIAS,
    DEFAULT_FINETUNE_ROBUST_TEST,
    DEFAULT_FINETUNE_SAVE_CKPT_FREQ,
    DEFAULT_FINETUNE_USE_MEAN_POOLING,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LAYER_SCALE_INIT_VALUE,
    DEFAULT_LOCAL_RANK,
    DEFAULT_LOG_DIR,
    DEFAULT_MIN_LR,
    DEFAULT_MOMENTUM,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OPT_EPS,
    DEFAULT_OPTIMIZER,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PIN_MEM,
    DEFAULT_PRETRAIN_ABS_POS_EMB,
    DEFAULT_PRETRAIN_EPOCHS,
    DEFAULT_PRETRAIN_INPUT_SIZE,
    DEFAULT_PRETRAIN_LR,
    DEFAULT_PRETRAIN_MODEL,
    DEFAULT_PRETRAIN_REL_POS_BIAS,
    DEFAULT_PRETRAIN_SAVE_CKPT_FREQ,
    DEFAULT_PRETRAIN_STRIDE,
    DEFAULT_QUANTIZER_DIM,
    DEFAULT_RESUME,
    DEFAULT_SAVE_CKPT,
    DEFAULT_SEED,
    DEFAULT_START_EPOCH,
    DEFAULT_TIME_WINDOWS,
    DEFAULT_TOKENIZER_MODEL,
    DEFAULT_TOKENIZER_WEIGHT,
    DEFAULT_UPDATE_FREQ,
    DEFAULT_VQNSP_DIST_EVAL,
    DEFAULT_VQNSP_EMA_DECAY,
    DEFAULT_VQNSP_EPOCHS,
    DEFAULT_VQNSP_INPUT_SIZE,
    DEFAULT_VQNSP_LR,
    DEFAULT_VQNSP_MODEL,
    DEFAULT_VQNSP_QUANTIZE_KMEANS_INIT,
    DEFAULT_VQNSP_SAVE_CKPT_FREQ,
    DEFAULT_VQNSP_STRIDE,
    DEFAULT_VQNSP_WEIGHT_DECAY,
    DEFAULT_WARMUP_EPOCHS,
    DEFAULT_WARMUP_LR,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WORLD_SIZE,
)


# ============================================================
# Shared sub-configs
# ============================================================


@dataclass
class OutputConfig(ConfigBase):
    output_dir: str = DEFAULT_OUTPUT_DIR
    log_dir: str = DEFAULT_LOG_DIR
    resume: str = DEFAULT_RESUME
    auto_resume: bool = DEFAULT_AUTO_RESUME
    save_ckpt: bool = DEFAULT_SAVE_CKPT
    save_ckpt_freq: int = DEFAULT_PRETRAIN_SAVE_CKPT_FREQ


@dataclass
class DistributedConfig(ConfigBase):
    world_size: int = DEFAULT_WORLD_SIZE
    local_rank: int = DEFAULT_LOCAL_RANK
    dist_on_itp: bool = DEFAULT_DIST_ON_ITP
    dist_url: str = DEFAULT_DIST_URL
    device: str = DEFAULT_DEVICE
    seed: int = DEFAULT_SEED
    dist_eval: bool = DEFAULT_DIST_EVAL


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


@dataclass
class TrainerConfig(ConfigBase):
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_PRETRAIN_EPOCHS
    start_epoch: int = DEFAULT_START_EPOCH
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    update_freq: int = DEFAULT_UPDATE_FREQ


@dataclass
class DataConfig(ConfigBase):
    """Pre-training and VQNSP dataset spec. ``datasets_train`` mirrors the
    nested-list layout build_pretraining_dataset expects: outer list groups
    files that share a channel montage."""

    num_workers: int = DEFAULT_NUM_WORKERS
    pin_mem: bool = DEFAULT_PIN_MEM
    datasets_train: List[List[str]] = field(default_factory=list)
    datasets_val: List[List[str]] = field(default_factory=list)
    time_window: List[int] = field(default_factory=lambda: list(DEFAULT_TIME_WINDOWS))
    val_time_window: List[int] = field(default_factory=list)
    stride: int = DEFAULT_PRETRAIN_STRIDE
    start_percentage: float = DEFAULT_DATASET_START_PERCENTAGE
    end_percentage: float = DEFAULT_DATASET_END_PERCENTAGE


@dataclass
class TokenizerConfig(ConfigBase):
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL
    tokenizer_weight: str = DEFAULT_TOKENIZER_WEIGHT
    codebook_size: int = DEFAULT_CODEBOOK_SIZE
    quantizer_dim: int = DEFAULT_QUANTIZER_DIM


# ============================================================
# Phase-specific model configs
# ============================================================


@dataclass
class PretrainModelConfig(ConfigBase):
    model: str = DEFAULT_PRETRAIN_MODEL
    input_size: int = DEFAULT_PRETRAIN_INPUT_SIZE
    rel_pos_bias: bool = DEFAULT_PRETRAIN_REL_POS_BIAS
    abs_pos_emb: bool = DEFAULT_PRETRAIN_ABS_POS_EMB
    layer_scale_init_value: float = DEFAULT_LAYER_SCALE_INIT_VALUE
    drop_path: float = DEFAULT_DROP_PATH


@dataclass
class VQNSPModelConfig(ConfigBase):
    model: str = DEFAULT_VQNSP_MODEL
    input_size: int = DEFAULT_VQNSP_INPUT_SIZE
    ema_decay: float = DEFAULT_VQNSP_EMA_DECAY
    quantize_kmeans_init: bool = DEFAULT_VQNSP_QUANTIZE_KMEANS_INIT


@dataclass
class FinetuneModelConfig(ConfigBase):
    model: str = DEFAULT_FINETUNE_MODEL
    input_size: int = DEFAULT_FINETUNE_INPUT_SIZE
    qkv_bias: bool = DEFAULT_FINETUNE_QKV_BIAS
    rel_pos_bias: bool = DEFAULT_FINETUNE_REL_POS_BIAS
    abs_pos_emb: bool = DEFAULT_FINETUNE_ABS_POS_EMB
    layer_scale_init_value: float = DEFAULT_LAYER_SCALE_INIT_VALUE
    drop: float = DEFAULT_DROP
    attn_drop_rate: float = DEFAULT_ATTN_DROP_RATE
    drop_path: float = DEFAULT_DROP_PATH
    use_mean_pooling: bool = DEFAULT_FINETUNE_USE_MEAN_POOLING
    init_scale: float = DEFAULT_FINETUNE_INIT_SCALE
    nb_classes: int = DEFAULT_FINETUNE_NB_CLASSES


@dataclass
class FinetuneCheckpointConfig(ConfigBase):
    """Checkpoint-loading knobs that only apply during fine-tuning."""

    finetune: str = DEFAULT_FINETUNE_CHECKPOINT
    model_key: str = DEFAULT_FINETUNE_MODEL_KEY
    model_prefix: str = DEFAULT_FINETUNE_MODEL_PREFIX
    model_filter_name: str = DEFAULT_FINETUNE_MODEL_FILTER_NAME


# ============================================================
# Top-level run configs
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
        ``auto_load_model``, ``save_model``, ``cosine_scheduler``) read a flat
        ``args.X`` surface and sometimes mutate it. We give them a separate
        Namespace so those mutations don't leak back into the typed config.
        """
        ns = Namespace()
        for sub_name, sub in self.__dict__.items():
            if isinstance(sub, ConfigBase):
                for k, v in sub.__dict__.items():
                    setattr(ns, k, v)
            else:
                setattr(ns, sub_name, sub)
        return ns


@dataclass
class PretrainRunConfig(RunConfig):
    model: PretrainModelConfig = field(default_factory=PretrainModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)


@dataclass
class VQNSPRunConfig(RunConfig):
    model: VQNSPModelConfig = field(default_factory=VQNSPModelConfig)
    # vqnsp.py defaults differ from the shared OptimizerConfig defaults
    # (lower lr, lower wd); we still expose the same OptimizerConfig type
    # but adjust at construction time below.
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
    # VQNSP-only flags
    disable_eval: bool = False
    eval: bool = False
    calculate_codebook_usage: bool = False


@dataclass
class FinetuneRunConfig(RunConfig):
    model: FinetuneModelConfig = field(default_factory=FinetuneModelConfig)
    finetune_checkpoint: FinetuneCheckpointConfig = field(default_factory=FinetuneCheckpointConfig)
    # finetune defaults: different lr/min_lr/layer_decay defaults
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
# CLI override parsing
# ============================================================


def _coerce_override(s: str):
    """Map a CLI override value (string) to a Python literal.

    Order matters: None, then bool, then int, then float, then string.
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
    """Parse ``--set key.sub=value`` arguments into a dict suitable for
    :meth:`ConfigBase.update`."""
    out: dict = {}
    for raw in items:
        if '=' not in raw:
            raise ValueError(f'Override must be key=value: {raw!r}')
        key, value = raw.split('=', 1)
        out[key.strip()] = _coerce_override(value.strip())
    return out
