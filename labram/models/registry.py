# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# ---------------------------------------------------------
#
# All timm @register_model factory functions live here. Importing this module
# triggers registration of every labram model name with timm, so runs only
# need ``import labram.models.registry``.

from functools import partial

import torch
import torch.nn as nn
from timm.models import register_model

from labram.configs import defaults as conf_consts
from labram.configs.model_config import (
    NeuralTransformerForMEMConfig,
    QuantizerConfig,
    TransformerArchConfig,
    VQNSPArchConfig,
)
from labram.models.masked_eeg import NeuralTransformerForMEM
from labram.models.neural_transformer import NeuralTransformer, _cfg
from labram.models.vqnsp import VQNSP, load_vqnsp_weights


def _strip_timm_kwargs(kwargs: dict) -> dict:
    """Drop kwargs injected by timm.create_model that labram models do not accept."""
    kwargs.pop("pretrained_cfg", None)
    kwargs.pop("pretrained_cfg_overlay", None)
    return kwargs


def _coerce_labram_plus(labram_plus):
    """Normalise a factory ``labram_plus`` kwarg into a LaBraMPlusConfig.

    Accepts ``None`` (-> disabled default), an existing config, or a plain dict
    (as it would arrive from a serialized run config)."""
    from labram.configs.labram_plus_config import LaBraMPlusConfig
    if labram_plus is None:
        return LaBraMPlusConfig()
    if isinstance(labram_plus, LaBraMPlusConfig):
        return labram_plus
    if isinstance(labram_plus, dict):
        return LaBraMPlusConfig.load_config(labram_plus)
    raise TypeError(f"labram_plus must be None, LaBraMPlusConfig, or dict; got {type(labram_plus)}")


def _cfg_arch_from_kwargs(arch_fields: dict, caller_kwargs: dict) -> TransformerArchConfig:
    """Build a TransformerArchConfig from declared arch fields + any matching
    caller kwargs (e.g. num_classes, drop_path_rate passed via create_model)."""
    merged = dict(arch_fields)
    for k, v in caller_kwargs.items():
        if TransformerArchConfig.exist(k):
            merged[k] = v
    return TransformerArchConfig(**merged)


# ---------------------------------------------------------------------------
# Fine-tuning / backbone factories
# ---------------------------------------------------------------------------

@register_model
def labram_base_patch200_200(pretrained=False, **kwargs):
    kw = _strip_timm_kwargs(kwargs)
    cfg = _cfg_arch_from_kwargs(
        dict(patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4,
             use_norm=True), kw)
    model = NeuralTransformer(
        cfg, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        **{k: v for k, v in kw.items() if not TransformerArchConfig.exist(k)})
    model.default_cfg = _cfg()
    return model


@register_model
def labram_large_patch200_200(pretrained=False, **kwargs):
    kw = _strip_timm_kwargs(kwargs)
    cfg = _cfg_arch_from_kwargs(
        dict(patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4,
             out_chans=16, use_norm=True), kw)
    model = NeuralTransformer(
        cfg, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        **{k: v for k, v in kw.items() if not TransformerArchConfig.exist(k)})
    model.default_cfg = _cfg()
    return model


@register_model
def labram_huge_patch200_200(pretrained=False, **kwargs):
    kw = _strip_timm_kwargs(kwargs)
    cfg = _cfg_arch_from_kwargs(
        dict(patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4,
             out_chans=32, use_norm=True), kw)
    model = NeuralTransformer(
        cfg, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        **{k: v for k, v in kw.items() if not TransformerArchConfig.exist(k)})
    model.default_cfg = _cfg()
    return model


# ---------------------------------------------------------------------------
# Pre-training factories
# ---------------------------------------------------------------------------

def _build_pretrain_model(arch_fields: dict, pretrained: bool, kwargs: dict) -> NeuralTransformerForMEM:
    """Build NeuralTransformerForMEM from arch fields + caller kwargs."""
    kw = dict(kwargs)
    kw.pop("num_classes", None)
    _strip_timm_kwargs(kw)
    vocab_size = kw.pop("vocab_size", conf_consts.DEFAULT_ARCH_VOCAB_SIZE)
    init_ckpt = kw.pop("init_ckpt", None)

    mem_cfg = NeuralTransformerForMEMConfig(
        vocab_size=vocab_size,
        **{k: v for k, v in arch_fields.items() if NeuralTransformerForMEMConfig.exist(k)},
        **{k: v for k, v in kw.items() if NeuralTransformerForMEMConfig.exist(k)},
    )
    # Pass remaining non-config kwargs (norm_layer, qk_norm) as keyword args
    extra_kw = {k: v for k, v in kw.items() if not NeuralTransformerForMEMConfig.exist(k)}
    extra_kw.update({k: v for k, v in arch_fields.items()
                     if not NeuralTransformerForMEMConfig.exist(k)})
    model = NeuralTransformerForMEM(mem_cfg, **extra_kw)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
    return model


def _labram_pretrain_arch(**overrides) -> dict:
    """Predefined NeuralTransformerForMEM arch fields (base variant); pass
    per-size overrides (embed_dim/depth/num_heads/out_chans) for large/huge."""
    fields = dict(
        patch_size=conf_consts.DEFAULT_ARCH_PATCH_SIZE,
        embed_dim=conf_consts.DEFAULT_ARCH_EMBED_DIM,
        depth=conf_consts.DEFAULT_ARCH_DEPTH,
        num_heads=conf_consts.DEFAULT_ARCH_NUM_HEADS,
        mlp_ratio=conf_consts.DEFAULT_ARCH_MLP_RATIO,
        qkv_bias=False,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    fields.update(overrides)
    return fields


@register_model
def labram_base_patch200_1600_8k_vocab(pretrained=False, **kwargs):  # 5 M params
    return _build_pretrain_model(_labram_pretrain_arch(), pretrained, kwargs)


@register_model
def labram_large_patch200_1600_8k_vocab(pretrained=False, **kwargs):  # 50 M params
    return _build_pretrain_model(
        _labram_pretrain_arch(embed_dim=400, depth=24, num_heads=16, out_chans=16),
        pretrained, kwargs,
    )


@register_model
def labram_huge_patch200_1600_8k_vocab(pretrained=False, **kwargs):  # 380 M params
    return _build_pretrain_model(
        _labram_pretrain_arch(embed_dim=800, depth=48, num_heads=16, out_chans=32),
        pretrained, kwargs,
    )


# ---------------------------------------------------------------------------
# VQNSP tokenizer factories
# ---------------------------------------------------------------------------

def _vqnsp_base_arch(eeg_window_size: int = conf_consts.DEFAULT_ARCH_EEG_WINDOW_SIZE,
                     **overrides) -> TransformerArchConfig:
    """Shared VQNSP encoder/decoder base architecture (base variant).

    ``overrides`` may replace any field (e.g. the decoder passes
    ``patch_size=1``), so merge them over the defaults rather than passing both.
    """
    fields = dict(
        eeg_window_size=eeg_window_size, patch_size=conf_consts.DEFAULT_ARCH_PATCH_SIZE, in_chans=1,
        num_classes=0, embed_dim=conf_consts.DEFAULT_ARCH_EMBED_DIM,
        depth=conf_consts.DEFAULT_ARCH_DEPTH, num_heads=conf_consts.DEFAULT_ARCH_NUM_HEADS,
        mlp_ratio=conf_consts.DEFAULT_ARCH_MLP_RATIO, qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0., init_values=0., use_abs_pos_emb=True,
        use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
        use_mean_pooling=True, init_scale=conf_consts.DEFAULT_ARCH_INIT_SCALE,
    )
    fields.update(overrides)
    return TransformerArchConfig(**fields)


@register_model
def vqnsp_encoder_base_decoder_3x200x12(pretrained=False, pretrained_weight=None,
                                         as_tokenzer=False,
                                         eeg_window_size=conf_consts.DEFAULT_ARCH_EEG_WINDOW_SIZE,
                                         num_codebook_tokens=conf_consts.DEFAULT_CODEBOOK_SIZE,
                                         quantizer_dim=conf_consts.DEFAULT_QUANTIZER_DIM,
                                         decay=conf_consts.DEFAULT_VQNSP_EMA_DECAY,
                                         quantize_kmeans_init=True, labram_plus=None, **kwargs):
    _strip_timm_kwargs(kwargs)
    enc_cfg = _vqnsp_base_arch(eeg_window_size=eeg_window_size)
    dec_cfg = _vqnsp_base_arch(
        eeg_window_size=eeg_window_size // conf_consts.DEFAULT_ARCH_PATCH_SIZE,
        patch_size=1, in_chans=quantizer_dim, depth=3,
    )
    arch_cfg = VQNSPArchConfig(
        encoder=enc_cfg, decoder=dec_cfg,
        quantizer=QuantizerConfig(
            num_codebook_tokens=num_codebook_tokens,
            quantizer_dim=quantizer_dim,
            decay=decay,
            kmeans_init=quantize_kmeans_init,
        ),
        decoder_out_dim=conf_consts.DEFAULT_ARCH_DECODER_OUT_DIM,
        labram_plus=_coerce_labram_plus(labram_plus),
    )
    model = VQNSP(arch_cfg)
    if as_tokenzer:
        assert pretrained and pretrained_weight is not None
        load_vqnsp_weights(model, pretrained_weight)
    return model


@register_model
def vqnsp_encoder_large_decoder_3x200x24(pretrained=False, pretrained_weight=None,
                                          as_tokenzer=False,
                                          eeg_window_size=conf_consts.DEFAULT_ARCH_EEG_WINDOW_SIZE,
                                          num_codebook_tokens=conf_consts.DEFAULT_CODEBOOK_SIZE,
                                          quantizer_dim=conf_consts.DEFAULT_QUANTIZER_DIM,
                                          decay=conf_consts.DEFAULT_VQNSP_EMA_DECAY,
                                          quantize_kmeans_init=True, labram_plus=None, **kwargs):
    _strip_timm_kwargs(kwargs)
    enc_cfg = _vqnsp_base_arch(eeg_window_size=eeg_window_size, depth=24)
    dec_cfg = _vqnsp_base_arch(
        eeg_window_size=eeg_window_size // conf_consts.DEFAULT_ARCH_PATCH_SIZE,
        patch_size=1, in_chans=quantizer_dim, depth=3,
    )
    arch_cfg = VQNSPArchConfig(
        encoder=enc_cfg, decoder=dec_cfg,
        quantizer=QuantizerConfig(
            num_codebook_tokens=num_codebook_tokens,
            quantizer_dim=quantizer_dim,
            decay=decay,
            kmeans_init=quantize_kmeans_init,
        ),
        decoder_out_dim=conf_consts.DEFAULT_ARCH_DECODER_OUT_DIM,
        labram_plus=_coerce_labram_plus(labram_plus),
    )
    model = VQNSP(arch_cfg)
    if as_tokenzer:
        assert pretrained and pretrained_weight is not None
        load_vqnsp_weights(model, pretrained_weight)
    return model
