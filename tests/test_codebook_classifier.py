"""Integration tests for codebook-regularized fine-tuning (PR2).

Covers the assembled model + run/training-loop wiring:
- CodebookRegularizedClassifier forward (full + classify-only), feature sources,
- per-component trainability (encoder last-N, frozen decoder/codebook),
- a real optimizer step advances the encoder while the codebook stays frozen,
- setup helpers (lr-scale validation, LossConfig mapping, LR-scale assigner),
- FinetuneRunConfig wiring + default JSONs load,
- train_one_epoch runs end to end and logs component losses.
"""
import math

import pytest
import torch
import torch.nn as nn

from labram.configs.loss_config import LossConfig
from labram.configs.model_config import (
    CodebookRegConfig,
    ComponentTrainConfig,
    QuantizerConfig,
    TransformerArchConfig,
    VQNSPArchConfig,
)
from labram.configs.optim_config import OptimizerConfig
from labram.configs.train_config import TrainerConfig
from labram.data import get_channel_indices
from labram.losses import CodebookRegularizedCriterion
from labram.models.codebook_classifier import CodebookRegularizedClassifier
from labram.models.neural_transformer import NeuralTransformer
from labram.models.outputs import PredictorOutput
from labram.models.vqnsp import VQNSP
from labram.runs.codebook_setup import (
    CodebookRegLayerAssigner,
    apply_component_trainability,
    loss_config_from_codebook_reg,
    validate_lr_scales,
)
from labram.optim_factory import optimizer_update
from labram.train.train_finetune import train_one_epoch
from labram.utils import NativeScalerWithGradNormCount


# ---------------------------------------------------------------------------
# Tiny builders (CPU, fast)
# ---------------------------------------------------------------------------

def _arch_cfg(**overrides) -> TransformerArchConfig:
    defaults = dict(
        eeg_window_size=400, patch_size=200, in_chans=1, out_chans=8, num_classes=0,
        embed_dim=200, depth=2, num_heads=10, mlp_ratio=4.0, qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, init_values=0.1,
        use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
        use_mean_pooling=True, init_scale=0.001,
    )
    return TransformerArchConfig(**{**defaults, **overrides})


def _make_tiny_vqnsp() -> VQNSP:
    enc = _arch_cfg()
    dec = _arch_cfg(eeg_window_size=400 // 200, patch_size=1, in_chans=8, depth=1)
    cfg = VQNSPArchConfig(
        encoder=enc, decoder=dec,
        quantizer=QuantizerConfig(num_codebook_tokens=32, quantizer_dim=8, kmeans_init=False),
        decoder_out_dim=200,
    )
    return VQNSP(cfg)


def _tiny_classifier(num_classes=2, feature_sources=('encoder_mean',), **kwargs):
    encoder = NeuralTransformer(_arch_cfg(num_classes=0))
    vq = _make_tiny_vqnsp()
    return CodebookRegularizedClassifier(
        encoder=encoder, quantizer=vq.quantizer, decoder=vq.decoder,
        encode_task_layer=vq.encode_task_layer,
        decode_task_layer_magnitude=vq.decode_task_layer_magnitude,
        decode_task_layer_phase=vq.decode_task_layer_phase,
        num_classes=num_classes, feature_sources=feature_sources,
        features_emb_dim=16, **kwargs)


def _ch():
    return get_channel_indices(['FP1', 'FP2', 'F3', 'F4'])


def _x():
    return (torch.randn(2, 4, 400) * 0.1).reshape(2, 4, 2, 200)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

class TestForward:
    def test_full_forward(self):
        model = _tiny_classifier(num_classes=3)
        model.train()
        out = model(_x(), channel_indices=_ch())
        assert isinstance(out, PredictorOutput)
        assert out.logits.shape == (2, 3)
        assert out.recon_magnitude.shape == out.recon_phase.shape == (2, 8, 200)
        assert torch.isfinite(out.recon_magnitude).all()
        assert out.quantize_loss is not None

    def test_classify_only_skips_recon(self):
        model = _tiny_classifier()
        out = model(_x(), channel_indices=_ch(), classify_only=True)
        assert out.logits.shape == (2, 2)
        assert out.recon_magnitude is None and out.quantize_loss is None

    @pytest.mark.parametrize('sources', [
        ['encoder_mean'], ['quantize_mean'], ['bag_of_codes'],
        ['encoder_mean', 'quantize_mean', 'bag_of_codes'],
    ])
    def test_feature_sources(self, sources):
        model = _tiny_classifier(num_classes=2, feature_sources=sources)
        out = model(_x(), channel_indices=_ch())
        assert out.logits.shape == (2, 2)

    def test_unknown_feature_source_raises(self):
        with pytest.raises(ValueError):
            _tiny_classifier(feature_sources=['bogus'])

    def test_helper_methods(self):
        model = _tiny_classifier()
        assert model.get_num_layers() == 2
        nwd = model.no_weight_decay()
        assert 'encoder.pos_embed' in nwd and 'quantizer.embedding.weight' in nwd


# ---------------------------------------------------------------------------
# Trainability
# ---------------------------------------------------------------------------

class TestTrainability:
    def test_defaults_freeze_decoder_and_codebook(self):
        model = _tiny_classifier()
        apply_component_trainability(model, CodebookRegConfig(enabled=True))
        assert all(not p.requires_grad for p in model.decoder.parameters())
        assert all(not p.requires_grad for p in model.encode_task_layer.parameters())
        assert all(p.requires_grad for p in model.encoder.parameters())
        assert model.quantizer.embedding.update is False

    def test_decoder_trainable_when_enabled(self):
        model = _tiny_classifier()
        cr = CodebookRegConfig(enabled=True,
                               decoder=ComponentTrainConfig(trainable=True, lr_scale=0.1))
        apply_component_trainability(model, cr)
        assert all(p.requires_grad for p in model.decoder.parameters())
        assert all(p.requires_grad for p in model.decode_task_layer_magnitude.parameters())

    def test_encoder_last_n_layers(self):
        model = _tiny_classifier()
        cr = CodebookRegConfig(enabled=True,
                               encoder=ComponentTrainConfig(trainable=True, lr_scale=0.1,
                                                            n_last_trainable_layers=1))
        apply_component_trainability(model, cr)
        b0 = [p.requires_grad for n, p in model.encoder.named_parameters() if n.startswith('blocks.0.')]
        b1 = [p.requires_grad for n, p in model.encoder.named_parameters() if n.startswith('blocks.1.')]
        assert b0 and not any(b0)
        assert b1 and all(b1)

    def test_encoder_fully_frozen(self):
        model = _tiny_classifier()
        cr = CodebookRegConfig(enabled=True,
                               encoder=ComponentTrainConfig(trainable=False, lr_scale=0.1))
        apply_component_trainability(model, cr)
        assert all(not p.requires_grad for p in model.encoder.parameters())

    def test_step_advances_encoder_and_freezes_codebook(self):
        torch.manual_seed(0)
        model = _tiny_classifier(num_classes=2)
        apply_component_trainability(model, CodebookRegConfig(enabled=True))
        model.train()
        crit = CodebookRegularizedCriterion(nn.CrossEntropyLoss(), LossConfig())
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-2)
        scaler = NativeScalerWithGradNormCount()
        enc_before = model.encoder.cls_token.detach().clone()
        codebook_before = model.quantizer.embedding.weight.detach().clone()
        out = model(_x(), channel_indices=_ch())
        breakdown = crit(out, torch.tensor([0, 1]))
        optimizer_update(breakdown.total, optimizer=opt, loss_scaler=scaler,
                         parameters=params, clip_grad=None, update_grad=True)
        assert not torch.equal(enc_before, model.encoder.cls_token.detach())
        assert torch.equal(codebook_before, model.quantizer.embedding.weight.detach())


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

class TestSetupHelpers:
    def test_validate_lr_scales_rejects_ge_one(self):
        with pytest.raises(ValueError):
            validate_lr_scales(CodebookRegConfig(encoder=ComponentTrainConfig(lr_scale=1.0)))

    def test_validate_lr_scales_accepts_defaults(self):
        validate_lr_scales(CodebookRegConfig())  # 0.1 everywhere

    def test_loss_config_mapping(self):
        cr = CodebookRegConfig(classifier_weight=2.0, amplitude_weight=0.5,
                               phase_weight=0.25, embedding_weight=3.0)
        lc = loss_config_from_codebook_reg(cr, label_smoothing=0.1)
        assert lc.classifier_weight == 2.0 and lc.amplitude_weight == 0.5
        assert lc.phase_weight == 0.25 and lc.embedding_weight == 3.0
        assert lc.classification_label_smoothing == 0.1

    def test_assigner_component_scales(self):
        cr = CodebookRegConfig(encoder=ComponentTrainConfig(lr_scale=0.1),
                               decoder=ComponentTrainConfig(trainable=True, lr_scale=0.2))
        a = CodebookRegLayerAssigner(cr, num_encoder_layers=2, layer_decay=1.0)
        assert a.get_scale(a.get_layer_id('classifier_head.weight')) == 1.0
        assert a.get_scale(a.get_layer_id('encoder.blocks.0.attn.qkv.weight')) == 0.1
        assert a.get_scale(a.get_layer_id('decoder.blocks.0.attn.qkv.weight')) == 0.2
        assert a.get_scale(a.get_layer_id('encode_task_layer.0.weight')) == 0.2

    def test_assigner_layer_decay_increases_with_depth(self):
        cr = CodebookRegConfig(encoder=ComponentTrainConfig(lr_scale=0.5))
        a = CodebookRegLayerAssigner(cr, num_encoder_layers=2, layer_decay=0.5)
        s0 = a.get_scale(a.get_layer_id('encoder.blocks.0.attn.qkv.weight'))
        s1 = a.get_scale(a.get_layer_id('encoder.blocks.1.attn.qkv.weight'))
        assert s1 > s0
        assert a.get_scale(a.get_layer_id('classifier_head.weight')) == 1.0


# ---------------------------------------------------------------------------
# FinetuneRunConfig wiring
# ---------------------------------------------------------------------------

class TestFinetuneRunConfigWiring:
    def test_default_disabled(self):
        from labram.configs.run_configs import FinetuneRunConfig
        assert FinetuneRunConfig().codebook_reg.enabled is False

    def test_default_jsons_load(self):
        from labram.configs.run_configs import FinetuneRunConfig
        for name in ('finetune_tuab', 'finetune_tuev', 'finetune_tuab_codebook'):
            cfg = FinetuneRunConfig.load_from(f'labram/configs/defaults/{name}.json')
            assert isinstance(cfg.codebook_reg, CodebookRegConfig)
        # the example config has the feature enabled
        ex = FinetuneRunConfig.load_from('labram/configs/defaults/finetune_tuab_codebook.json')
        assert ex.codebook_reg.enabled is True


# ---------------------------------------------------------------------------
# Training-loop integration
# ---------------------------------------------------------------------------

class _DS(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class TestTrainOneEpoch:
    def test_runs_and_logs_component_losses(self):
        torch.manual_seed(0)
        model = _tiny_classifier(num_classes=2)
        apply_component_trainability(model, CodebookRegConfig(enabled=True))
        crit = CodebookRegularizedCriterion(nn.CrossEntropyLoss(), LossConfig())
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-3)
        scaler = NativeScalerWithGradNormCount()

        x = torch.randn(4, 4, 400) * 0.1
        y = torch.randint(0, 2, (4,))
        loader = torch.utils.data.DataLoader(_DS(x, y), batch_size=2, drop_last=True)

        stats = train_one_epoch(
            model=model, criterion=crit, data_loader=loader, optimizer=opt,
            device=torch.device('cpu'), epoch=0, loss_scaler=scaler,
            trainer_cfg=TrainerConfig(update_freq=1),
            optim_cfg=OptimizerConfig(clip_grad=None),
            num_training_steps_per_epoch=2, start_steps=0,
            ch_names=['FP1', 'FP2', 'F3', 'F4'], is_binary=False,
        )
        assert math.isfinite(stats['loss'])
        for key in ('classifier_loss', 'magnitude_loss', 'phase_loss', 'quantize_loss'):
            assert key in stats, f"missing {key} in {sorted(stats)}"
