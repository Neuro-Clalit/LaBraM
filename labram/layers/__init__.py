# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Neural-network layer primitives, one responsibility per module.
# ---------------------------------------------------------

from labram.layers.attention import Attention
from labram.layers.drop_path import DropPath
from labram.layers.feature_embedders import CodeBookBagEmbedder, FeatureEmbedder
from labram.layers.lora import LoRALinear, inject_lora, mark_only_lora_trainable
from labram.layers.mlp import Mlp
from labram.layers.patch_embed import PatchEmbed, TemporalConv
from labram.layers.transformer_block import Block


__all__ = [
    'Attention',
    'Block',
    'CodeBookBagEmbedder',
    'DropPath',
    'FeatureEmbedder',
    'LoRALinear',
    'Mlp',
    'inject_lora',
    'mark_only_lora_trainable',
    'PatchEmbed',
    'TemporalConv',
]
