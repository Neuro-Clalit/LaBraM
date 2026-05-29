from dataclasses import dataclass

from labram.configs.base_configs import ConfigBase
from labram.configs.defaults import (
    DEFAULT_ATTN_DROP_RATE,
    DEFAULT_CODEBOOK_SIZE,
    DEFAULT_DROP,
    DEFAULT_DROP_PATH,
    DEFAULT_FINETUNE_ABS_POS_EMB,
    DEFAULT_FINETUNE_CHECKPOINT,
    DEFAULT_FINETUNE_INIT_SCALE,
    DEFAULT_FINETUNE_INPUT_SIZE,
    DEFAULT_FINETUNE_MODEL,
    DEFAULT_FINETUNE_MODEL_FILTER_NAME,
    DEFAULT_FINETUNE_MODEL_KEY,
    DEFAULT_FINETUNE_MODEL_PREFIX,
    DEFAULT_FINETUNE_NB_CLASSES,
    DEFAULT_FINETUNE_QKV_BIAS,
    DEFAULT_FINETUNE_REL_POS_BIAS,
    DEFAULT_FINETUNE_USE_MEAN_POOLING,
    DEFAULT_LAYER_SCALE_INIT_VALUE,
    DEFAULT_PRETRAIN_ABS_POS_EMB,
    DEFAULT_PRETRAIN_INPUT_SIZE,
    DEFAULT_PRETRAIN_MODEL,
    DEFAULT_PRETRAIN_REL_POS_BIAS,
    DEFAULT_QUANTIZER_DIM,
    DEFAULT_TOKENIZER_MODEL,
    DEFAULT_TOKENIZER_WEIGHT,
    DEFAULT_VQNSP_EMA_DECAY,
    DEFAULT_VQNSP_INPUT_SIZE,
    DEFAULT_VQNSP_MODEL,
    DEFAULT_VQNSP_QUANTIZE_KMEANS_INIT,
)


@dataclass
class TokenizerConfig(ConfigBase):
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL
    tokenizer_weight: str = DEFAULT_TOKENIZER_WEIGHT
    codebook_size: int = DEFAULT_CODEBOOK_SIZE
    quantizer_dim: int = DEFAULT_QUANTIZER_DIM


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
    codebook_size: int = DEFAULT_CODEBOOK_SIZE
    quantizer_dim: int = DEFAULT_QUANTIZER_DIM


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
