# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Assembly + optimizer-grouping for codebook-regularized fine-tuning: build the
# classifier from a fine-tune encoder + a pre-trained VQNSP tokenizer, apply the
# configured per-component trainability, and assign per-component LR scales.
# ---------------------------------------------------------

import warnings

from timm.models import create_model

from labram.configs.loss_config import LossConfig
from labram.configs.run_configs import FinetuneRunConfig
from labram.models.codebook_classifier import CodebookRegularizedClassifier
from labram.models.vqnsp import load_vqnsp_weights
from labram.optim_factory import get_num_layer_for_vit


def loss_config_from_codebook_reg(cr, label_smoothing: float) -> LossConfig:
    """Map a CodebookRegConfig (+ label smoothing) into a LossConfig the
    criterion consumes."""
    return LossConfig(
        classification_label_smoothing=label_smoothing,
        classifier_weight=cr.classifier_weight,
        amplitude_weight=cr.amplitude_weight,
        phase_weight=cr.phase_weight,
        embedding_weight=cr.embedding_weight,
    )


def validate_lr_scales(cr) -> None:
    """Encoder/decoder must learn slower than the classification head."""
    for name in ('encoder', 'decoder'):
        scale = getattr(cr, name).lr_scale
        if scale >= 1.0:
            raise ValueError(
                f"codebook_reg.{name}.lr_scale must be < 1.0 (smaller than the "
                f"classification head LR); got {scale}")


def _freeze_encoder_except_last_n(encoder, n: int) -> None:
    num = encoder.get_num_layers()
    n = max(0, min(n, num))
    trainable_blocks = set(range(num - n, num))
    for name, param in encoder.named_parameters():
        if name.startswith('blocks.'):
            block_id = int(name.split('.')[1])
            param.requires_grad_(block_id in trainable_blocks)
        elif name.startswith(('norm.', 'fc_norm.')):
            param.requires_grad_(True)
        else:  # patch_embed / cls_token / pos_embed / time_embed -> early, frozen
            param.requires_grad_(False)


def apply_component_trainability(model: CodebookRegularizedClassifier, cr) -> None:
    """Freeze/unfreeze components per config.

    Encoder: all layers (or only the last N) trainable, or fully frozen.
    Decoder + projection task layers: trainable together (reconstruction head).
    Codebook: EMA updates toggled (it is never gradient-trained).
    """
    if not cr.encoder.trainable:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    elif cr.encoder.n_last_trainable_layers is not None:
        _freeze_encoder_except_last_n(model.encoder, cr.encoder.n_last_trainable_layers)
    else:
        for p in model.encoder.parameters():
            p.requires_grad_(True)

    recon_modules = (model.decoder, model.encode_task_layer,
                     model.decode_task_layer_magnitude, model.decode_task_layer_phase)
    for mod in recon_modules:
        for p in mod.parameters():
            p.requires_grad_(cr.decoder.trainable)

    model.quantizer.embedding.update = cr.codebook.trainable


def build_codebook_classifier(config: FinetuneRunConfig) -> CodebookRegularizedClassifier:
    """Build the encoder + graft a pre-trained VQNSP quantizer/decoder, then
    apply trainability. The encoder's pre-trained weights are loaded separately
    (into ``model.encoder``) by the caller."""
    m = config.model
    cr = m.codebook_reg
    validate_lr_scales(cr)

    encoder = create_model(
        m.model, pretrained=False, num_classes=0,
        drop_rate=m.drop, drop_path_rate=m.drop_path, attn_drop_rate=m.attn_drop_rate,
        use_mean_pooling=m.use_mean_pooling, init_scale=m.init_scale,
        use_rel_pos_bias=m.rel_pos_bias, use_abs_pos_emb=m.abs_pos_emb,
        init_values=m.layer_scale_init_value, qkv_bias=m.qkv_bias,
    )

    tokenizer = create_model(cr.tokenizer_model, pretrained=False)
    if cr.tokenizer_weight:
        load_vqnsp_weights(tokenizer, cr.tokenizer_weight)
    else:
        warnings.warn(
            "codebook_reg.enabled but tokenizer_weight is empty: the quantizer/"
            "decoder start from random init, so the reconstruction/quantization "
            "losses are not meaningful regularizers until trained.")

    if encoder.embed_dim != tokenizer.encoder.embed_dim:
        raise ValueError(
            f"encoder embed_dim ({encoder.embed_dim}) must match the VQNSP encoder "
            f"embed_dim ({tokenizer.encoder.embed_dim}) for the projection layers to fit")

    model = CodebookRegularizedClassifier(
        encoder=encoder,
        quantizer=tokenizer.quantizer,
        decoder=tokenizer.decoder,
        encode_task_layer=tokenizer.encode_task_layer,
        decode_task_layer_magnitude=tokenizer.decode_task_layer_magnitude,
        decode_task_layer_phase=tokenizer.decode_task_layer_phase,
        num_classes=m.nb_classes,
        feature_sources=cr.feature_sources,
        features_emb_dim=cr.features_emb_dim,
        linear_embedding=cr.linear_embedding,
        norm_embedding=cr.norm_embedding,
        classifier_type=cr.classifier_type,
    )
    apply_component_trainability(model, cr)
    return model


class CodebookRegLayerAssigner:
    """Per-component LR-scale assigner for the wrapped classifier.

    Bucket 0 (scale 1.0) is the classification head / feature embedders. The
    encoder gets ``encoder.lr_scale`` (optionally with per-block layer decay on
    top); the decoder + projection task layers get ``decoder.lr_scale``. The
    codebook has no gradient params, so it needs no bucket. Plugs into
    ``create_optimizer`` via ``get_layer_id`` / ``get_scale``.
    """

    _RECON_PREFIXES = ('decoder.', 'encode_task_layer.',
                       'decode_task_layer_magnitude.', 'decode_task_layer_phase.')

    def __init__(self, cr, num_encoder_layers: int, layer_decay: float):
        self.cr = cr
        self.layer_decay = layer_decay
        self.num_encoder_layers = num_encoder_layers
        self.scales = [1.0]  # id 0: head + embedders
        self.enc_base = len(self.scales)
        if layer_decay < 1.0:
            n = num_encoder_layers + 2
            self.scales += [cr.encoder.lr_scale * (layer_decay ** (n - 1 - i)) for i in range(n)]
        else:
            self.scales.append(cr.encoder.lr_scale)
        self.dec_id = len(self.scales)
        self.scales.append(cr.decoder.lr_scale)

    def get_scale(self, layer_id: int) -> float:
        return self.scales[layer_id]

    def get_layer_id(self, name: str) -> int:
        if name.startswith('encoder.'):
            sub = name[len('encoder.'):]
            if self.layer_decay < 1.0:
                return self.enc_base + get_num_layer_for_vit(sub, self.num_encoder_layers + 2)
            return self.enc_base
        if name.startswith(self._RECON_PREFIXES):
            return self.dec_id
        return 0
