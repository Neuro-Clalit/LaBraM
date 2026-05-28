# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# ---------------------------------------------------------
#
# All timm @register_model factory functions live here. Importing this module
# triggers registration of every labram model name with timm, so runners only
# need ``import labram.models.registry``.

from functools import partial

import torch
import torch.nn as nn
from timm.models import register_model

from labram.models.masked_eeg import NeuralTransformerForMEM
from labram.models.neural_transformer import NeuralTransformer, _cfg
from labram.models.vqnsp import VQNSP, load_vqnsp_weights


def _strip_timm_kwargs(kwargs: dict) -> dict:
    """Drop kwargs injected by timm.create_model that the labram models do not accept."""
    kwargs.pop("pretrained_cfg", None)
    kwargs.pop("pretrained_cfg_overlay", None)
    return kwargs


# ---------------------------------------------------------------------------
# Fine-tuning / backbone factories
# ---------------------------------------------------------------------------

@register_model
def labram_base_patch200_200(pretrained=False, **kwargs):
    model = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **_strip_timm_kwargs(kwargs))
    model.default_cfg = _cfg()
    return model


@register_model
def labram_large_patch200_200(pretrained=False, **kwargs):
    model = NeuralTransformer(
        patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, out_chans=16, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **_strip_timm_kwargs(kwargs))
    model.default_cfg = _cfg()
    return model


@register_model
def labram_huge_patch200_200(pretrained=False, **kwargs):
    model = NeuralTransformer(
        patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, out_chans=32, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **_strip_timm_kwargs(kwargs))
    model.default_cfg = _cfg()
    return model


# ---------------------------------------------------------------------------
# Pre-training factories
# ---------------------------------------------------------------------------

def _build_pretrain_model(arch_kwargs: dict, pretrained: bool, kwargs: dict) -> NeuralTransformerForMEM:
    """Shared factory boilerplate: pop timm-injected keys, build model, optionally load weights."""
    kw = dict(kwargs)
    kw.pop("num_classes", None)
    _strip_timm_kwargs(kw)
    vocab_size = kw.pop("vocab_size", 8192)
    init_ckpt = kw.pop("init_ckpt", None)
    model = NeuralTransformerForMEM(**arch_kwargs, vocab_size=vocab_size, **kw)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def labram_base_patch200_1600_8k_vocab(pretrained=False, **kwargs):  # 5 M params
    return _build_pretrain_model(
        dict(patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4,
             qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6),
             norm_layer=partial(nn.LayerNorm, eps=1e-6)),
        pretrained, kwargs,
    )


@register_model
def labram_large_patch200_1600_8k_vocab(pretrained=False, **kwargs):  # 50 M params
    return _build_pretrain_model(
        dict(patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4,
             qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=16,
             norm_layer=partial(nn.LayerNorm, eps=1e-6)),
        pretrained, kwargs,
    )


@register_model
def labram_huge_patch200_1600_8k_vocab(pretrained=False, **kwargs):  # 380 M params
    return _build_pretrain_model(
        dict(patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4,
             qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=32,
             norm_layer=partial(nn.LayerNorm, eps=1e-6)),
        pretrained, kwargs,
    )


# ---------------------------------------------------------------------------
# VQNSP tokenizer factories
# ---------------------------------------------------------------------------

def _vqnsp_default_params():
    return dict(eeg_window_size=1600, patch_size=200, in_chans=1, num_classes=1000, embed_dim=200, depth=12, num_heads=10,
                mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., use_abs_pos_emb=True,
                use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001)


@register_model
def vqnsp_encoder_base_decoder_3x200x12(pretrained=False, pretrained_weight=None, as_tokenzer=False, eeg_window_size=1600,
                                        num_codebook_tokens=8192, quantizer_dim=32, **kwargs):
    encoder_config, decoder_config = _vqnsp_default_params(), _vqnsp_default_params()

    # encoder settings
    encoder_config['eeg_window_size'] = eeg_window_size
    encoder_config['num_classes'] = 0
    # decoder settings
    decoder_config['eeg_window_size'] = eeg_window_size // decoder_config['patch_size']
    decoder_config['patch_size'] = 1
    decoder_config['in_chans'] = quantizer_dim
    decoder_config['num_classes'] = 0
    decoder_config['depth'] = 3
    decoder_out_dim = 200

    model = VQNSP(encoder_config, decoder_config, num_codebook_tokens, quantizer_dim,
                  decoder_out_dim=decoder_out_dim, **_strip_timm_kwargs(kwargs))

    if as_tokenzer:
        assert pretrained
        assert pretrained_weight is not None
        load_vqnsp_weights(model, pretrained_weight)
    return model


@register_model
def vqnsp_encoder_large_decoder_3x200x24(pretrained=False, pretrained_weight=None, as_tokenzer=False, eeg_window_size=1600,
                                         num_codebook_tokens=8192, quantizer_dim=32, **kwargs):
    encoder_config, decoder_config = _vqnsp_default_params(), _vqnsp_default_params()

    # encoder settings
    encoder_config['eeg_window_size'] = eeg_window_size
    encoder_config['num_classes'] = 0
    encoder_config['depth'] = 24
    # decoder settings
    decoder_config['eeg_window_size'] = eeg_window_size // decoder_config['patch_size']
    decoder_config['patch_size'] = 1
    decoder_config['in_chans'] = quantizer_dim
    decoder_config['num_classes'] = 0
    decoder_config['depth'] = 3
    decoder_out_dim = 200

    model = VQNSP(encoder_config, decoder_config, num_codebook_tokens, quantizer_dim,
                  decoder_out_dim=decoder_out_dim, **_strip_timm_kwargs(kwargs))

    if as_tokenzer:
        assert pretrained
        assert pretrained_weight is not None
        load_vqnsp_weights(model, pretrained_weight)
    return model
