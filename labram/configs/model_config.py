from dataclasses import dataclass, field
from typing import Optional

from labram.configs.base_configs import ConfigBase
from labram.configs import defaults as conf_consts



@dataclass
class TokenizerConfig(ConfigBase):
    tokenizer_model: str = conf_consts.DEFAULT_TOKENIZER_MODEL
    tokenizer_weight: str = conf_consts.DEFAULT_TOKENIZER_WEIGHT
    codebook_size: int = conf_consts.DEFAULT_CODEBOOK_SIZE
    quantizer_dim: int = conf_consts.DEFAULT_QUANTIZER_DIM


@dataclass
class PretrainModelConfig(ConfigBase):
    model: str = conf_consts.DEFAULT_PRETRAIN_MODEL
    input_size: int = conf_consts.DEFAULT_PRETRAIN_INPUT_SIZE
    rel_pos_bias: bool = conf_consts.DEFAULT_PRETRAIN_REL_POS_BIAS
    abs_pos_emb: bool = conf_consts.DEFAULT_PRETRAIN_ABS_POS_EMB
    layer_scale_init_value: float = conf_consts.DEFAULT_LAYER_SCALE_INIT_VALUE
    drop_path: float = conf_consts.DEFAULT_DROP_PATH


@dataclass
class VQNSPModelConfig(ConfigBase):
    model: str = conf_consts.DEFAULT_VQNSP_MODEL
    input_size: int = conf_consts.DEFAULT_VQNSP_INPUT_SIZE
    ema_decay: float = conf_consts.DEFAULT_VQNSP_EMA_DECAY
    quantize_kmeans_init: bool = conf_consts.DEFAULT_VQNSP_QUANTIZE_KMEANS_INIT
    codebook_size: int = conf_consts.DEFAULT_CODEBOOK_SIZE
    quantizer_dim: int = conf_consts.DEFAULT_QUANTIZER_DIM


@dataclass
class FinetuneModelConfig(ConfigBase):
    model: str = conf_consts.DEFAULT_FINETUNE_MODEL
    input_size: int = conf_consts.DEFAULT_FINETUNE_INPUT_SIZE
    qkv_bias: bool = conf_consts.DEFAULT_FINETUNE_QKV_BIAS
    rel_pos_bias: bool = conf_consts.DEFAULT_FINETUNE_REL_POS_BIAS
    abs_pos_emb: bool = conf_consts.DEFAULT_FINETUNE_ABS_POS_EMB
    layer_scale_init_value: float = conf_consts.DEFAULT_LAYER_SCALE_INIT_VALUE
    drop: float = conf_consts.DEFAULT_DROP
    attn_drop_rate: float = conf_consts.DEFAULT_ATTN_DROP_RATE
    drop_path: float = conf_consts.DEFAULT_DROP_PATH
    use_mean_pooling: bool = conf_consts.DEFAULT_FINETUNE_USE_MEAN_POOLING
    init_scale: float = conf_consts.DEFAULT_FINETUNE_INIT_SCALE
    nb_classes: int = conf_consts.DEFAULT_FINETUNE_NB_CLASSES


@dataclass
class FinetuneCheckpointConfig(ConfigBase):
    """Checkpoint-loading knobs that only apply during fine-tuning."""
    finetune: str = conf_consts.DEFAULT_FINETUNE_CHECKPOINT
    model_key: str = conf_consts.DEFAULT_FINETUNE_MODEL_KEY
    model_prefix: str = conf_consts.DEFAULT_FINETUNE_MODEL_PREFIX
    model_filter_name: str = conf_consts.DEFAULT_FINETUNE_MODEL_FILTER_NAME


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
    eeg_window_size: int = conf_consts.DEFAULT_ARCH_EEG_WINDOW_SIZE
    patch_size: int = conf_consts.DEFAULT_ARCH_PATCH_SIZE
    in_chans: int = conf_consts.DEFAULT_ARCH_IN_CHANS
    out_chans: int = conf_consts.DEFAULT_ARCH_OUT_CHANS
    num_classes: int = conf_consts.DEFAULT_ARCH_NUM_CLASSES
    embed_dim: int = conf_consts.DEFAULT_ARCH_EMBED_DIM
    depth: int = conf_consts.DEFAULT_ARCH_DEPTH
    num_heads: int = conf_consts.DEFAULT_ARCH_NUM_HEADS
    mlp_ratio: float = conf_consts.DEFAULT_ARCH_MLP_RATIO
    qkv_bias: bool = conf_consts.DEFAULT_ARCH_QKV_BIAS
    drop_rate: float = conf_consts.DEFAULT_ARCH_DROP_RATE
    attn_drop_rate: float = conf_consts.DEFAULT_ARCH_ATTN_DROP_RATE
    drop_path_rate: float = conf_consts.DEFAULT_ARCH_DROP_PATH_RATE
    init_values: Optional[float] = None
    use_abs_pos_emb: bool = conf_consts.DEFAULT_ARCH_USE_ABS_POS_EMB
    use_rel_pos_bias: bool = conf_consts.DEFAULT_ARCH_USE_REL_POS_BIAS
    use_shared_rel_pos_bias: bool = conf_consts.DEFAULT_ARCH_USE_SHARED_REL_POS_BIAS
    use_mean_pooling: bool = conf_consts.DEFAULT_ARCH_USE_MEAN_POOLING
    init_scale: float = conf_consts.DEFAULT_ARCH_INIT_SCALE
    init_std: float = conf_consts.DEFAULT_ARCH_INIT_STD
    use_norm: bool = conf_consts.DEFAULT_ARCH_USE_NORM


@dataclass
class NeuralTransformerForMEMConfig(TransformerArchConfig):
    """TransformerArchConfig extended with the masked-EEG head vocab size."""
    vocab_size: int = conf_consts.DEFAULT_ARCH_VOCAB_SIZE


@dataclass
class QuantizerConfig(ConfigBase):
    """Config for NormEMAVectorQuantizer."""
    num_codebook_tokens: int = conf_consts.DEFAULT_CODEBOOK_SIZE
    quantizer_dim: int = conf_consts.DEFAULT_QUANTIZER_DIM
    decay: float = conf_consts.DEFAULT_VQNSP_EMA_DECAY
    kmeans_init: bool = conf_consts.DEFAULT_VQNSP_QUANTIZE_KMEANS_INIT
    beta: float = conf_consts.DEFAULT_QUANTIZER_BETA


@dataclass
class VQNSPArchConfig(ConfigBase):
    """Architecture config for a VQNSP tokenizer (encoder + decoder + quantizer)."""
    encoder: TransformerArchConfig = field(default_factory=TransformerArchConfig)
    decoder: TransformerArchConfig = field(default_factory=TransformerArchConfig)
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)
    decoder_out_dim: int = conf_consts.DEFAULT_ARCH_DECODER_OUT_DIM
    smooth_l1_loss: bool = conf_consts.DEFAULT_ARCH_SMOOTH_L1_LOSS
