from dataclasses import dataclass, field
from typing import Optional

from labram.configs.base_configs import ConfigBase
from labram.configs.defaults import (
    DEFAULT_ARCH_ATTN_DROP_RATE,
    DEFAULT_ARCH_DECODER_OUT_DIM,
    DEFAULT_ARCH_DEPTH,
    DEFAULT_ARCH_DROP_PATH_RATE,
    DEFAULT_ARCH_DROP_RATE,
    DEFAULT_ARCH_EEG_WINDOW_SIZE,
    DEFAULT_ARCH_EMBED_DIM,
    DEFAULT_ARCH_IN_CHANS,
    DEFAULT_ARCH_INIT_SCALE,
    DEFAULT_ARCH_INIT_STD,
    DEFAULT_ARCH_MLP_RATIO,
    DEFAULT_ARCH_NUM_CLASSES,
    DEFAULT_ARCH_NUM_HEADS,
    DEFAULT_ARCH_OUT_CHANS,
    DEFAULT_ARCH_PATCH_SIZE,
    DEFAULT_ARCH_QKV_BIAS,
    DEFAULT_ARCH_SMOOTH_L1_LOSS,
    DEFAULT_ARCH_USE_ABS_POS_EMB,
    DEFAULT_ARCH_USE_MEAN_POOLING,
    DEFAULT_ARCH_USE_NORM,
    DEFAULT_ARCH_USE_REL_POS_BIAS,
    DEFAULT_ARCH_USE_SHARED_REL_POS_BIAS,
    DEFAULT_ARCH_VOCAB_SIZE,
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

DEFAULT_QUANTIZER_BETA: float = 1.0


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


# ============================================================
# Architecture-level configs (transformer backbone hyperparams)
# ============================================================


@dataclass
class TransformerArchConfig(ConfigBase):
    """All serialisable hyperparams of NeuralTransformerBase.

    Non-serialisable params (``norm_layer``, ``qk_norm``) stay as
    keyword arguments on the model constructors; they are set by the
    timm registry factory and are not user-configurable from YAML/JSON.
    """
    eeg_window_size: int = DEFAULT_ARCH_EEG_WINDOW_SIZE
    patch_size: int = DEFAULT_ARCH_PATCH_SIZE
    in_chans: int = DEFAULT_ARCH_IN_CHANS
    out_chans: int = DEFAULT_ARCH_OUT_CHANS
    num_classes: int = DEFAULT_ARCH_NUM_CLASSES
    embed_dim: int = DEFAULT_ARCH_EMBED_DIM
    depth: int = DEFAULT_ARCH_DEPTH
    num_heads: int = DEFAULT_ARCH_NUM_HEADS
    mlp_ratio: float = DEFAULT_ARCH_MLP_RATIO
    qkv_bias: bool = DEFAULT_ARCH_QKV_BIAS
    drop_rate: float = DEFAULT_ARCH_DROP_RATE
    attn_drop_rate: float = DEFAULT_ARCH_ATTN_DROP_RATE
    drop_path_rate: float = DEFAULT_ARCH_DROP_PATH_RATE
    init_values: Optional[float] = None
    use_abs_pos_emb: bool = DEFAULT_ARCH_USE_ABS_POS_EMB
    use_rel_pos_bias: bool = DEFAULT_ARCH_USE_REL_POS_BIAS
    use_shared_rel_pos_bias: bool = DEFAULT_ARCH_USE_SHARED_REL_POS_BIAS
    use_mean_pooling: bool = DEFAULT_ARCH_USE_MEAN_POOLING
    init_scale: float = DEFAULT_ARCH_INIT_SCALE
    init_std: float = DEFAULT_ARCH_INIT_STD
    use_norm: bool = DEFAULT_ARCH_USE_NORM


@dataclass
class NeuralTransformerForMEMConfig(TransformerArchConfig):
    """TransformerArchConfig extended with the masked-EEG head vocab size."""
    vocab_size: int = DEFAULT_ARCH_VOCAB_SIZE


@dataclass
class QuantizerConfig(ConfigBase):
    """Config for NormEMAVectorQuantizer."""
    num_codebook_tokens: int = DEFAULT_CODEBOOK_SIZE
    quantizer_dim: int = DEFAULT_QUANTIZER_DIM
    decay: float = DEFAULT_VQNSP_EMA_DECAY
    kmeans_init: bool = DEFAULT_VQNSP_QUANTIZE_KMEANS_INIT
    beta: float = DEFAULT_QUANTIZER_BETA


@dataclass
class VQNSPArchConfig(ConfigBase):
    """Architecture config for a VQNSP tokenizer (encoder + decoder + quantizer)."""
    encoder: TransformerArchConfig = field(default_factory=TransformerArchConfig)
    decoder: TransformerArchConfig = field(default_factory=TransformerArchConfig)
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)
    decoder_out_dim: int = DEFAULT_ARCH_DECODER_OUT_DIM
    smooth_l1_loss: bool = DEFAULT_ARCH_SMOOTH_L1_LOSS
