"""Unit tests for the labram.losses package (Phase C), config-driven trunk.

Covers the configurable loss building blocks and pins numeric parity with the
original inline implementations:

- ``LossConfig`` defaults reproduce the old hard-coded behaviour,
- ``get_vqnsp_losses`` weights/sums the components (default == plain sum),
- ``SpectralReconstructionLoss`` matches the original FFT+std-norm+MSE math,
- ``build_classification_criterion`` reproduces the BCE/CE/LabelSmoothing logic,
- ``VQNSP`` threads its arch config (smooth_l1_loss + quantizer.beta) into a
  ``LossConfig`` and the recon-loss module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.loss import LabelSmoothingCrossEntropy

from labram.configs.model_config import QuantizerConfig, TransformerArchConfig, VQNSPArchConfig
from labram.losses import (
    LossConfig,
    SpectralReconstructionLoss,
    build_classification_criterion,
    get_vqnsp_losses,
)
from labram.models.vqnsp import VQNSP


# ---------------------------------------------------------------------------
# LossConfig
# ---------------------------------------------------------------------------

class TestLossConfig:
    def test_defaults_match_original_behaviour(self):
        cfg = LossConfig()
        assert cfg.amplitude_weight == 1.0
        assert cfg.phase_weight == 1.0
        assert cfg.embedding_weight == 1.0
        assert cfg.use_smooth_l1 is False
        assert cfg.vq_commitment_beta == 1.0
        assert cfg.classification_label_smoothing == 0.0


# ---------------------------------------------------------------------------
# get_vqnsp_losses
# ---------------------------------------------------------------------------

class TestGetVQNSPLosses:
    def test_default_total_is_plain_sum(self):
        e, a, p = torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)
        out = get_vqnsp_losses(e, a, p)
        assert torch.allclose(out['total'], torch.tensor(6.0))
        assert out['embedding'] is e and out['amplitude'] is a and out['phase'] is p

    def test_weighted_total(self):
        e, a, p = torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)
        cfg = LossConfig(embedding_weight=0.5, amplitude_weight=2.0, phase_weight=0.0)
        out = get_vqnsp_losses(e, a, p, cfg)
        assert torch.allclose(out['total'], torch.tensor(0.5 * 1 + 2.0 * 2 + 0.0 * 3))

    def test_keys(self):
        out = get_vqnsp_losses(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0))
        assert set(out) == {'embedding', 'amplitude', 'phase', 'total'}


# ---------------------------------------------------------------------------
# SpectralReconstructionLoss  (parity with the original VQNSP math)
# ---------------------------------------------------------------------------

def _ref_std_norm(t):
    mean = torch.mean(t, dim=(1, 2, 3), keepdim=True)
    std = torch.std(t, dim=(1, 2, 3), keepdim=True)
    return (t - mean) / std


class TestSpectralReconstructionLoss:
    def test_spectrum_targets_match_reference(self):
        loss = SpectralReconstructionLoss()
        x = torch.randn(2, 4, 2, 200)
        amp, phase = loss.spectrum_targets(x)
        x_fft = torch.fft.fft(x, dim=-1)
        assert torch.allclose(amp, _ref_std_norm(torch.abs(x_fft)))
        assert torch.allclose(phase, _ref_std_norm(torch.angle(x_fft)))

    def test_reconstruction_loss_matches_mse(self):
        loss = SpectralReconstructionLoss()
        recon = torch.randn(2, 8, 200)
        target = torch.randn(2, 4, 2, 200)
        ref = F.mse_loss(recon, rearrange(target, 'b n a c -> b (n a) c'))
        assert torch.allclose(loss.reconstruction_loss(recon, target), ref)

    def test_smooth_l1_option(self):
        loss = SpectralReconstructionLoss(LossConfig(use_smooth_l1=True))
        recon = torch.randn(2, 8, 200)
        target = torch.randn(2, 4, 2, 200)
        ref = F.smooth_l1_loss(recon, rearrange(target, 'b n a c -> b (n a) c'))
        assert torch.allclose(loss.reconstruction_loss(recon, target), ref)

    def test_freq_fraction_truncates_spectrum(self):
        loss = SpectralReconstructionLoss(LossConfig(freq_fraction=0.5))
        x = torch.randn(2, 4, 2, 200)
        amp, phase = loss.spectrum_targets(x)
        assert amp.shape[-1] == 100  # 200 * 0.5
        assert phase.shape[-1] == 100

    def test_freq_fraction_one_is_full_spectrum(self):
        loss_full = SpectralReconstructionLoss(LossConfig(freq_fraction=1.0))
        loss_default = SpectralReconstructionLoss()
        x = torch.randn(2, 4, 2, 200)
        amp_full, phase_full = loss_full.spectrum_targets(x)
        amp_def, phase_def = loss_default.spectrum_targets(x)
        assert amp_full.shape == amp_def.shape
        assert torch.allclose(amp_full, amp_def)

    def test_freq_fraction_reconstruction_loss_aligned(self):
        loss = SpectralReconstructionLoss(LossConfig(freq_fraction=0.5))
        recon = torch.randn(2, 8, 200)   # full T from decoder
        target = torch.randn(2, 4, 2, 100)  # already-truncated target
        ref = F.mse_loss(recon[..., :100], rearrange(target, 'b n a c -> b (n a) c'))
        assert torch.allclose(loss.reconstruction_loss(recon, target), ref)

    def test_is_parameter_free(self):
        assert list(SpectralReconstructionLoss().state_dict().keys()) == []


# ---------------------------------------------------------------------------
# build_classification_criterion  (parity with the original runner logic)
# ---------------------------------------------------------------------------

class TestBuildClassificationCriterion:
    def test_binary_returns_bce(self):
        assert isinstance(build_classification_criterion(1), nn.BCEWithLogitsLoss)

    def test_multiclass_no_smoothing_returns_ce(self):
        assert isinstance(build_classification_criterion(5), nn.CrossEntropyLoss)

    def test_multiclass_with_smoothing_returns_label_smoothing(self):
        crit = build_classification_criterion(5, LossConfig(classification_label_smoothing=0.1))
        assert isinstance(crit, LabelSmoothingCrossEntropy)

    def test_binary_ignores_smoothing(self):
        crit = build_classification_criterion(1, LossConfig(classification_label_smoothing=0.1))
        assert isinstance(crit, nn.BCEWithLogitsLoss)


# ---------------------------------------------------------------------------
# VQNSP wiring: arch config (smooth_l1_loss + quantizer.beta) -> LossConfig
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


def _make_tiny_vqnsp(smooth_l1_loss=False, beta=1.0):
    enc = _arch_cfg()
    dec = _arch_cfg(eeg_window_size=400 // 200, patch_size=1, in_chans=8, depth=1)
    cfg = VQNSPArchConfig(
        encoder=enc, decoder=dec,
        quantizer=QuantizerConfig(num_codebook_tokens=32, quantizer_dim=8, kmeans_init=False, beta=beta),
        decoder_out_dim=200, smooth_l1_loss=smooth_l1_loss,
    )
    return VQNSP(cfg)


class TestVQNSPLossWiring:
    def test_default_uses_mse_and_configured_beta(self):
        model = _make_tiny_vqnsp(smooth_l1_loss=False, beta=1.0)
        assert model.quantizer.beta == 1.0
        assert model.loss_config.vq_commitment_beta == 1.0
        assert model.loss_config.use_smooth_l1 is False
        assert model.recon_loss._loss_fn is F.mse_loss

    def test_custom_config_propagates(self):
        model = _make_tiny_vqnsp(smooth_l1_loss=True, beta=0.25)
        assert model.quantizer.beta == 0.25
        assert model.loss_config.vq_commitment_beta == 0.25
        assert model.loss_config.use_smooth_l1 is True
        assert model.recon_loss._loss_fn is F.smooth_l1_loss

    def test_forward_runs_and_logs_expected_keys(self):
        from labram.data import get_channel_indices

        model = _make_tiny_vqnsp()
        model.train()
        x = torch.randn(2, 4, 400) * 0.1
        channel_indices = get_channel_indices(['FP1', 'FP2', 'F3', 'F4'])
        loss, log = model(x, channel_indices=channel_indices)
        assert torch.isfinite(loss)
        assert set(log) == {
            'train/quant_loss', 'train/rec_loss', 'train/rec_angle_loss', 'train/total_loss',
        }
