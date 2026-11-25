# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# The one place that builds a fine-tune model from a run config.
# ---------------------------------------------------------
"""``build_finetune_model(config)``: the model of a ``FinetuneRunConfig``.

Training (``run_finetune.get_model``) and every evaluation path
(``labram/eval/loading.py``, the scoring scripts) build the model here, so a
run is always rebuilt the way it was trained. ``model.model`` selects the
encoder: a LaBraM ``NeuralTransformer`` (``labram_*``), or a third-party
pretrained encoder wrapped in a ``FoundationRegressor`` (``cbramod_*``,
``reve_*``; ``labram/models/fm_regressor.py``).

The factory never opens ``finetune_checkpoint.finetune``: a saved run config
holds the path the weights had in the training container. Loading them is
``finetune_setup.load_finetune_checkpoint``.
"""
from timm.models import create_model

import labram.models.registry  # noqa: F401  (registers the timm factories)
import labram.utils as utils
from labram.layers.lora import inject_lora, mark_only_lora_trainable
from labram.models.registry import is_foreign_encoder
from labram.runs.codebook_setup import build_codebook_classifier
from labram.runs.finetune_setup import required_time_patches

logger = utils.get_logger(__name__)

# LaBraM-only model options a foreign encoder does not read (name, value that means "unused").
_LABRAM_ONLY = (("use_mean_pooling", None), ("rel_pos_bias", None), ("abs_pos_emb", None),
                ("qkv_bias", None), ("layer_scale_init_value", None), ("drop_path", None),
                ("attn_drop_rate", None), ("drop", None), ("fused_attention", None))


def check_codebook_aux_supported(config, time_patches: int) -> None:
    """``codebook_aux`` reads the patch tokens of the plain LaBraM encoder."""
    config.codebook_aux.validate()
    m = config.model
    if m.codebook_reg.enabled:
        raise ValueError("codebook_aux and model.codebook_reg are two ways to add the same losses: enable one")
    if m.session_features != "off" or m.age_input != "off":
        raise ValueError("codebook_aux is not supported with session features or age_input")
    if config.trainer.enable_deepspeed:
        raise ValueError("codebook_aux is not supported with DeepSpeed")
    if time_patches > 16:
        raise ValueError("codebook_aux needs windows of at most 16 patches: the VQNSP decoder "
                         "has a fixed 16-patch time embedding")


def _build_foreign(config):
    """``FoundationRegressor`` around CBraMod / REVE (``model.model``)."""
    m = config.model
    m.validate()
    if getattr(config, 'codebook_aux', None) is not None and config.codebook_aux.enabled:
        raise ValueError(f"codebook_aux is not supported with model.model={m.model!r}: the "
                         "auxiliary reads LaBraM patch tokens")
    if config.trainer.enable_deepspeed:
        raise ValueError(f"DeepSpeed is not supported with model.model={m.model!r}")
    model = create_model(
        m.model, pretrained=False,
        num_classes=m.nb_classes, head_type=m.head_type, mlp_head=m.mlp_head,
        multitask_head=m.multitask_head, init_scale=m.init_scale,
        encoder=m.encoder, labram_plus=config.labram_plus,
        input_clip=m.input_clip_uv / 100,
    )
    logger.info("Foreign encoder %s: %d layers, width %d; readout=%s input_norm=%s drop_path=%g. "
                "LaBraM-only model options are not used: %s", m.model, model.get_num_layers(),
                model.embed_dim, model.readout, model.input_norm, m.encoder.drop_path,
                ", ".join(name for name, _ in _LABRAM_ONLY))
    return model


def build_finetune_model(config):
    """The fine-tune model of ``config`` (a ``FinetuneRunConfig``), with its
    bag pool, auxiliary loss and LoRA factors when the config enables them."""
    time_patches = required_time_patches(config.data)
    if config.model.codebook_reg.enabled:
        if time_patches > 16:
            raise ValueError(
                "data.window_sec > 16 is not supported with model.codebook_reg: the "
                "grafted VQNSP decoder has a fixed 16-patch time embedding")
        return build_codebook_classifier(config)
    m = config.model
    foreign = is_foreign_encoder(m.model)
    if foreign:
        model = _build_foreign(config)
    else:
        model = create_model(
            m.model, pretrained=False,
            num_classes=m.nb_classes, drop_rate=m.drop,
            drop_path_rate=m.drop_path, attn_drop_rate=m.attn_drop_rate,
            use_mean_pooling=m.use_mean_pooling, init_scale=m.init_scale,
            concat_cls_token=m.concat_cls_token,
            head_type=m.head_type, attention_head=m.attention_head, mlp_head=m.mlp_head,
            multitask_head=m.multitask_head,
            condition_on_abnormal=m.condition_on_abnormal,
            abnormal_embed_dim=m.abnormal_embed_dim,
            # model.age_input=tokens reuses the session-token path: one [1, 1] group.
            session_features="tokens" if m.age_input == "tokens" else m.session_features,
            session_feature_groups=[[1, 1]] if m.age_input == "tokens" else m.session_feature_groups,
            session_feature_drop=m.session_feature_drop,
            session_feature_norm=m.session_feature_norm,
            session_token_gain=m.session_token_gain,
            session_head_dim=m.session_head_dim,
            use_rel_pos_bias=m.rel_pos_bias, use_abs_pos_emb=m.abs_pos_emb,
            init_values=m.layer_scale_init_value, qkv_bias=m.qkv_bias,
            labram_plus=config.labram_plus, max_time_patches=time_patches,
            **({'depth': m.depth} if m.depth is not None else {}),
            input_clip=m.input_clip_uv / 100,
        )
        if m.fused_attention:
            from labram.layers.attention import set_fused_attention
            set_fused_attention(model, True)
    if getattr(config, 'mil', None) is not None and config.mil.enabled:
        from labram.train.train_mil import attach_mil
        attach_mil(model, config.mil)
    if foreign:
        return model
    if getattr(config, 'codebook_aux', None) is not None and config.codebook_aux.enabled:
        check_codebook_aux_supported(config, time_patches)
        from labram.models.codebook_aux import attach_codebook_aux
        attach_codebook_aux(model, config.codebook_aux)
    if m.lora.enabled:
        # Injected before the pretrained checkpoint loads: the base weights keep
        # their keys, the LoRA factors are new (B = 0, so the model is unchanged
        # until training), and only they + the head remain trainable.
        n_layers = inject_lora(model, m.lora.targets, m.lora.rank, m.lora.alpha, m.lora.dropout)
        n_trainable = mark_only_lora_trainable(model, m.lora.train_prefixes)
        logger.info("LoRA: rank %d (alpha %.1f) on %d linears; %d trainable parameters",
                    m.lora.rank, m.lora.alpha, n_layers, n_trainable)
    return model
