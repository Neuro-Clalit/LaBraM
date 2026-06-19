"""Unit tests for the codebook-regularized fine-tuning building blocks (PR1).

Covers the standalone pieces the CodebookRegularizedClassifier (PR2) composes:
- ``LossConfig.classifier_weight`` default,
- ``CodebookRegularizedCriterion`` combines/weights losses into a LossBreakdown,
- ``FeatureEmbedder`` / ``CodeBookBagEmbedder`` output shapes,
- ``optimizer_update`` steps params and reports a grad norm,
- the VQNSP decoder task-layer rename + legacy-checkpoint remap,
- the shared ``quantize_patch_tokens`` / ``decode_spectrum`` helpers,
- ``ComponentTrainConfig`` / ``CodebookRegConfig`` defaults + round-trip.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from labram.configs.loss_config import LossConfig
from labram.configs.model_config import (
    CodebookRegConfig,
    ComponentTrainConfig,
    QuantizerConfig,
    TransformerArchConfig,
    VQNSPArchConfig,
)
from labram.data import get_channel_indices
from labram.layers import CodeBookBagEmbedder, FeatureEmbedder
from labram.losses import CodebookRegularizedCriterion, LossBreakdown
from labram.models.components import decode_spectrum, quantize_patch_tokens
from labram.models.outputs import PredictorOutput
from labram.models.vqnsp import VQNSP, remap_legacy_task_layer_keys
from labram.train.base import optimizer_update
from labram.utils import NativeScalerWithGradNormCount


# ---------------------------------------------------------------------------
# Tiny VQNSP fixture (mirrors tests/test_losses.py)
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


def _ch():
    return get_channel_indices(['FP1', 'FP2', 'F3', 'F4'])


# ---------------------------------------------------------------------------
# LossConfig.classifier_weight
# ---------------------------------------------------------------------------

class TestClassifierWeight:
    def test_default_is_one(self):
        assert LossConfig().classifier_weight == 1.0


# ---------------------------------------------------------------------------
# CodebookRegularizedCriterion
# ---------------------------------------------------------------------------

def _output(recon: bool = True, B: int = 2, C: int = 3) -> PredictorOutput:
    torch.manual_seed(0)
    logits = torch.randn(B, C)
    if not recon:
        return PredictorOutput(logits=logits)
    n, a, t = 2, 2, 200
    return PredictorOutput(
        logits=logits,
        recon_magnitude=torch.randn(B, n * a, t),
        recon_phase=torch.randn(B, n * a, t),
        quantize_loss=torch.tensor(0.5),
        x_patched=torch.randn(B, n, a, t),
    )


class TestCodebookRegularizedCriterion:
    def test_components_and_weighting(self):
        cfg = LossConfig(classifier_weight=2.0, amplitude_weight=0.5,
                         phase_weight=0.25, embedding_weight=3.0)
        crit = CodebookRegularizedCriterion(nn.CrossEntropyLoss(), cfg)
        breakdown = crit(_output(), torch.tensor([0, 2]))
        assert isinstance(breakdown, LossBreakdown)
        assert set(breakdown.components) == {'classifier', 'magnitude', 'phase', 'quantize'}
        c = breakdown.components
        expected = (2.0 * c['classifier'] + 0.5 * c['magnitude']
                    + 0.25 * c['phase'] + 3.0 * c['quantize'])
        assert torch.allclose(breakdown.total, expected)

    def test_default_weights_are_plain_sum_of_terms(self):
        crit = CodebookRegularizedCriterion(nn.CrossEntropyLoss())  # all weights 1.0 except phase via LossConfig
        breakdown = crit(_output(), torch.tensor([0, 1]))
        c = breakdown.components
        expected = c['classifier'] + c['magnitude'] + c['phase'] + c['quantize']
        assert torch.allclose(breakdown.total, expected)

    def test_classification_only_when_no_recon(self):
        crit = CodebookRegularizedCriterion(nn.CrossEntropyLoss())
        breakdown = crit(_output(recon=False), torch.tensor([0, 1]))
        assert set(breakdown.components) == {'classifier'}
        assert torch.allclose(breakdown.total, breakdown.components['classifier'])

    def test_to_log_dict_keys_and_types(self):
        crit = CodebookRegularizedCriterion(nn.CrossEntropyLoss())
        log = crit(_output(), torch.tensor([0, 1])).to_log_dict('train')
        assert set(log) == {
            'train/total_loss', 'train/classifier_loss',
            'train/magnitude_loss', 'train/phase_loss', 'train/quantize_loss',
        }
        assert all(isinstance(v, float) for v in log.values())


# ---------------------------------------------------------------------------
# Feature embedders
# ---------------------------------------------------------------------------

class TestFeatureEmbedders:
    def test_feature_embedder_mean_reduce(self):
        emb = FeatureEmbedder(in_dim=8, out_dim=5, reduce_dim=1)
        assert emb(torch.randn(3, 10, 8)).shape == (3, 5)

    def test_feature_embedder_linear_identity_passthrough(self):
        emb = FeatureEmbedder(in_dim=8, out_dim=8, is_linear=True)
        assert isinstance(emb.embedder, nn.Identity)
        x = torch.randn(3, 8)
        assert torch.allclose(emb(x), x)

    def test_bag_of_codes_shape(self):
        emb = CodeBookBagEmbedder(n_codes=32, out_dim=6)
        idx = torch.randint(0, 32, (3, 12))
        assert emb(idx).shape == (3, 6)


# ---------------------------------------------------------------------------
# optimizer_update
# ---------------------------------------------------------------------------

class TestOptimizerUpdate:
    def test_steps_params_and_reports_grad_norm(self):
        torch.manual_seed(0)
        model = nn.Linear(4, 1)
        before = model.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        scaler = NativeScalerWithGradNormCount()
        loss = F.mse_loss(model(torch.randn(8, 4)), torch.randn(8, 1))
        grad_norm = optimizer_update(
            loss, optimizer=optimizer, loss_scaler=scaler,
            parameters=model.parameters(), clip_grad=None, update_grad=True)
        assert grad_norm is not None
        assert not torch.equal(before, model.weight.detach())

    def test_no_step_between_boundaries(self):
        torch.manual_seed(0)
        model = nn.Linear(4, 1)
        before = model.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        scaler = NativeScalerWithGradNormCount()
        loss = F.mse_loss(model(torch.randn(8, 4)), torch.randn(8, 1))
        grad_norm = optimizer_update(
            loss, optimizer=optimizer, loss_scaler=scaler,
            parameters=model.parameters(), clip_grad=None, update_grad=False)
        assert grad_norm is None
        assert torch.equal(before, model.weight.detach())


# ---------------------------------------------------------------------------
# VQNSP decoder task-layer rename + legacy remap
# ---------------------------------------------------------------------------

class TestTaskLayerRename:
    def test_state_dict_uses_new_names(self):
        keys = list(_make_tiny_vqnsp().state_dict())
        assert any(k.startswith('decode_task_layer_magnitude.') for k in keys)
        assert any(k.startswith('decode_task_layer_phase.') for k in keys)
        assert not any(k.startswith('decode_task_layer.') for k in keys)
        assert not any(k.startswith('decode_task_layer_angle') for k in keys)

    def test_remap_renames_only_task_layers(self):
        legacy = {
            'decode_task_layer.0.weight': torch.zeros(1),
            'decode_task_layer.0.bias': torch.zeros(1),
            'decode_task_layer_angle.2.weight': torch.zeros(1),
            'encoder.cls_token': torch.zeros(1),
            'quantize.embedding.weight': torch.zeros(1),
        }
        out = remap_legacy_task_layer_keys(legacy)
        assert 'decode_task_layer_magnitude.0.weight' in out
        assert 'decode_task_layer_magnitude.0.bias' in out
        assert 'decode_task_layer_phase.2.weight' in out
        assert 'encoder.cls_token' in out and 'quantize.embedding.weight' in out
        assert not any(k.startswith('decode_task_layer.') or k.startswith('decode_task_layer_angle')
                       for k in out)

    def test_legacy_checkpoint_loads_after_remap(self):
        sd = _make_tiny_vqnsp().state_dict()
        legacy = OrderedDict()
        for k, v in sd.items():
            if k.startswith('decode_task_layer_magnitude.'):
                k = k.replace('decode_task_layer_magnitude.', 'decode_task_layer.', 1)
            elif k.startswith('decode_task_layer_phase.'):
                k = k.replace('decode_task_layer_phase.', 'decode_task_layer_angle.', 1)
            legacy[k] = v
        fresh = _make_tiny_vqnsp()
        fresh.load_state_dict(remap_legacy_task_layer_keys(legacy))  # must not raise

    def test_forward_still_runs(self):
        model = _make_tiny_vqnsp()
        model.train()
        loss, log = model(torch.randn(2, 4, 400) * 0.1, channel_indices=_ch())
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Shared encode/decode helpers
# ---------------------------------------------------------------------------

class TestSharedComponents:
    def test_quantize_and_decode_shapes(self):
        torch.manual_seed(0)
        model = _make_tiny_vqnsp()
        x = (torch.randn(2, 4, 400) * 0.1).reshape(2, 4, 2, 200)
        enc_feats = model.encoder(x, _ch(), return_patch_tokens=True)
        quantized, qloss, idx = quantize_patch_tokens(
            model.encode_task_layer, model.quantize, enc_feats, num_channels=4)
        assert quantized.shape == (2, 8, 4, 2)         # B, quantizer_dim, N, A
        assert idx.shape == (2 * 4 * 2,)               # one index per token
        mag, phase = decode_spectrum(
            model.decoder, model.decode_task_layer_magnitude, model.decode_task_layer_phase,
            quantized, _ch())
        assert mag.shape == phase.shape == (2, 8, 200)  # B, N*A, decoder_out_dim


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

class TestCodebookRegConfig:
    def test_component_defaults(self):
        c = ComponentTrainConfig()
        assert c.trainable is True and c.lr_scale == 1.0 and c.n_last_trainable_layers is None

    def test_defaults(self):
        cfg = CodebookRegConfig()
        assert cfg.enabled is False
        assert cfg.feature_sources == ['encoder_mean']
        assert cfg.encoder.trainable is True
        assert cfg.decoder.trainable is False and cfg.decoder.lr_scale == 0.1
        assert cfg.codebook.trainable is False
        assert cfg.classifier_weight == 1.0

    def test_roundtrip(self, tmp_path):
        cfg = CodebookRegConfig(enabled=True,
                                feature_sources=['encoder_mean', 'bag_of_codes'])
        path = str(tmp_path / 'c.json')
        cfg.save_to(path)
        loaded = CodebookRegConfig.load_from(path)
        assert loaded.enabled is True
        assert loaded.feature_sources == ['encoder_mean', 'bag_of_codes']
        assert loaded.decoder.trainable is False
        assert isinstance(loaded.encoder, ComponentTrainConfig)
