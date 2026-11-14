"""Tests for the LaBraM++ training option (arXiv:2505.16724).

Covers the three opt-in improvements and their wiring:

* ``LaBraMPlusConfig`` -- the master-switch semantics and phase-loss validation,
* CAR + per-patch z-scoring preprocessing helpers (math + config gating),
* the sin/cos circular phase reconstruction loss (parity + wrap-around fix),
* model wiring: VQNSP tokenizer, the fine-tune backbone, and the masked-EEG
  pre-training model preprocess only when enabled and never double-apply,
* the shipped LaBraM++ config files load with the mode enabled, and the
  original default configs keep it disabled (behaviour preserved).
"""
import math

import pytest
import torch
import torch.nn.functional as F
from timm.models import create_model

import labram.models.registry  # noqa: F401  (registers model factories)
from labram.configs.labram_plus_config import LaBraMPlusConfig
from labram.configs.loss_config import LossConfig
from labram.configs.model_config import QuantizerConfig, TransformerArchConfig, VQNSPArchConfig
from labram.data import get_channel_indices
from labram.data.preprocess import (
    apply_labram_plus_preprocess,
    common_average_reference,
    z_score_per_patch,
)
from labram.losses import SpectralReconstructionLoss
from labram.models.vqnsp import VQNSP


# ---------------------------------------------------------------------------
# LaBraMPlusConfig
# ---------------------------------------------------------------------------

class TestLaBraMPlusConfig:
    def test_disabled_by_default_reproduces_original(self):
        cfg = LaBraMPlusConfig()
        assert cfg.enabled is False
        assert cfg.use_car is False
        assert cfg.use_z_score is False
        assert cfg.preprocesses_input is False
        assert cfg.resolved_phase_loss == "angle"

    def test_enabled_turns_on_all_features(self):
        cfg = LaBraMPlusConfig(enabled=True)
        assert cfg.use_car is True
        assert cfg.use_z_score is True
        assert cfg.preprocesses_input is True
        assert cfg.resolved_phase_loss == "sincos"

    def test_sub_features_only_apply_when_enabled(self):
        # sub-flags default True but stay inert while the master switch is off
        cfg = LaBraMPlusConfig(enabled=False, common_average_reference=True, z_score_patches=True)
        assert cfg.use_car is False and cfg.use_z_score is False

    def test_individual_ablation(self):
        cfg = LaBraMPlusConfig(enabled=True, common_average_reference=False)
        assert cfg.use_car is False
        assert cfg.use_z_score is True

    def test_invalid_phase_loss_rejected(self):
        with pytest.raises(ValueError):
            LaBraMPlusConfig(phase_loss="bogus")

    def test_round_trip(self, tmp_path):
        cfg = LaBraMPlusConfig(enabled=True, z_score_patches=False, phase_loss="angle")
        path = str(tmp_path / "lp.json")
        cfg.save_to(path)
        loaded = LaBraMPlusConfig.load_from(path)
        assert loaded.enabled is True
        assert loaded.z_score_patches is False
        assert loaded.phase_loss == "angle"


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_car_removes_across_channel_mean(self):
        x = torch.randn(2, 5, 3, 200)
        out = common_average_reference(x)
        assert out.shape == x.shape
        assert torch.allclose(out.mean(dim=1), torch.zeros_like(out.mean(dim=1)), atol=1e-5)

    def test_car_cancels_shared_signal(self):
        # a component shared by every channel is fully removed by CAR
        base = torch.randn(2, 1, 3, 200)
        per_ch = torch.randn(2, 4, 3, 200)
        x = per_ch + base  # broadcast shared component across channels
        out = common_average_reference(x)
        assert torch.allclose(out, common_average_reference(per_ch), atol=1e-5)

    def test_z_score_standardizes_each_patch(self):
        x = torch.randn(2, 4, 3, 200) * 7 + 5
        out = z_score_per_patch(x)
        assert torch.allclose(out.mean(dim=-1), torch.zeros_like(out.mean(dim=-1)), atol=1e-4)
        assert torch.allclose(out.std(dim=-1), torch.ones_like(out.std(dim=-1)), atol=1e-2)

    def test_z_score_eps_guards_flat_patch(self):
        x = torch.zeros(1, 1, 1, 200)
        out = z_score_per_patch(x, eps=1e-5)
        assert torch.isfinite(out).all()

    def test_apply_none_or_disabled_is_identity(self):
        x = torch.randn(2, 4, 3, 200)
        assert torch.equal(apply_labram_plus_preprocess(x, None), x)
        assert torch.equal(apply_labram_plus_preprocess(x, LaBraMPlusConfig()), x)

    def test_apply_enabled_runs_car_then_zscore(self):
        x = torch.randn(2, 4, 3, 200)
        cfg = LaBraMPlusConfig(enabled=True)
        expected = z_score_per_patch(common_average_reference(x), cfg.z_score_eps)
        assert torch.allclose(apply_labram_plus_preprocess(x, cfg), expected)

    def test_apply_respects_individual_flags(self):
        x = torch.randn(2, 4, 3, 200)
        car_only = LaBraMPlusConfig(enabled=True, z_score_patches=False)
        assert torch.allclose(apply_labram_plus_preprocess(x, car_only), common_average_reference(x))


# ---------------------------------------------------------------------------
# Sin/cos circular phase reconstruction loss
# ---------------------------------------------------------------------------

class TestSincosPhaseLoss:
    def test_angle_mode_matches_original_targets(self):
        loss = SpectralReconstructionLoss(LossConfig(phase_loss="angle"))
        x = torch.randn(2, 4, 2, 200)
        _, phase = loss.spectrum_targets(x)
        # original behaviour: std-normalised angle (mean ~0, std ~1 over dims 1..3)
        assert torch.allclose(phase.mean(dim=(1, 2, 3)),
                              torch.zeros(x.shape[0]), atol=1e-4)

    def test_sincos_mode_returns_raw_angle(self):
        loss = SpectralReconstructionLoss(LossConfig(phase_loss="sincos"))
        x = torch.randn(2, 4, 2, 200)
        _, phase = loss.spectrum_targets(x)
        x_fft = torch.fft.fft(x, dim=-1)
        assert torch.allclose(phase, torch.angle(x_fft))  # raw radians, not std-normed

    def test_sincos_loss_formula(self):
        loss = SpectralReconstructionLoss(LossConfig(phase_loss="sincos"))
        recon = torch.randn(2, 8, 200)
        target = torch.randn(2, 4, 2, 200)
        from einops import rearrange
        t = rearrange(target, 'b n a c -> b (n a) c')
        ref = F.mse_loss(torch.sin(recon), torch.sin(t)) + F.mse_loss(torch.cos(recon), torch.cos(t))
        assert torch.allclose(loss.phase_reconstruction_loss(recon, target), ref)

    def test_sincos_is_continuous_across_pi_boundary(self):
        # phi=+pi and phi_hat=-pi are the same angle: raw-angle MSE sees ~4pi^2,
        # the circular sin/cos loss sees ~0. This is the LaBraM++ fix.
        loss = SpectralReconstructionLoss(LossConfig(phase_loss="sincos"))
        recon = torch.full((1, 1, 1), -math.pi + 1e-4)
        target = torch.full((1, 1, 1, 1), math.pi - 1e-4)
        circular = loss.phase_reconstruction_loss(recon, target)
        raw = F.mse_loss(recon, target.reshape(1, 1, 1))
        assert circular.item() < 1e-3
        assert raw.item() > 30  # ~ (2pi)^2


# ---------------------------------------------------------------------------
# VQNSP tokenizer wiring
# ---------------------------------------------------------------------------

def _arch(**overrides):
    defaults = dict(
        eeg_window_size=400, patch_size=200, in_chans=1, out_chans=8, num_classes=0,
        embed_dim=200, depth=2, num_heads=10, mlp_ratio=4.0, qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, init_values=0.1,
        use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
        use_mean_pooling=True, init_scale=0.001,
    )
    return TransformerArchConfig(**{**defaults, **overrides})


def _tiny_vqnsp(labram_plus=None):
    enc = _arch()
    dec = _arch(eeg_window_size=400 // 200, patch_size=1, in_chans=8, depth=1)
    cfg = VQNSPArchConfig(
        encoder=enc, decoder=dec,
        quantizer=QuantizerConfig(num_codebook_tokens=32, quantizer_dim=8, kmeans_init=False),
        decoder_out_dim=200,
    )
    if labram_plus is not None:
        cfg.labram_plus = labram_plus
    return VQNSP(cfg)


class TestVQNSPWiring:
    def test_default_is_angle_and_no_preprocess(self):
        model = _tiny_vqnsp()
        assert model.loss_config.phase_loss == "angle"
        assert model.labram_plus.preprocesses_input is False

    def test_enabled_selects_sincos(self):
        model = _tiny_vqnsp(LaBraMPlusConfig(enabled=True))
        assert model.loss_config.phase_loss == "sincos"
        assert model.recon_loss.cfg.phase_loss == "sincos"

    def test_internal_encoder_never_double_preprocesses(self):
        # VQNSP owns preprocessing; its inner encoder/decoder must stay disabled
        model = _tiny_vqnsp(LaBraMPlusConfig(enabled=True))
        assert model.encoder.labram_plus.enabled is False
        assert model.decoder.labram_plus.enabled is False

    def test_forward_runs_enabled(self):
        model = _tiny_vqnsp(LaBraMPlusConfig(enabled=True))
        model.train()
        ci = get_channel_indices(['FP1', 'FP2', 'F3', 'F4'])
        loss, log = model(torch.randn(2, 4, 400) * 0.1, channel_indices=ci)
        assert torch.isfinite(loss)
        assert set(log) == {'train/quant_loss', 'train/rec_loss',
                            'train/rec_angle_loss', 'train/total_loss'}

    def test_registry_factory_threads_labram_plus(self):
        model = create_model(
            'vqnsp_encoder_base_decoder_3x200x12', pretrained=False, as_tokenzer=False,
            num_codebook_tokens=32, quantizer_dim=32, eeg_window_size=400,
            quantize_kmeans_init=False, labram_plus=LaBraMPlusConfig(enabled=True))
        assert model.loss_config.phase_loss == "sincos"
        assert model.labram_plus.enabled is True

    def test_registry_default_disabled(self):
        model = create_model(
            'vqnsp_encoder_base_decoder_3x200x12', pretrained=False, as_tokenzer=False,
            num_codebook_tokens=32, quantizer_dim=32, eeg_window_size=400,
            quantize_kmeans_init=False)
        assert model.loss_config.phase_loss == "angle"
        assert model.labram_plus.enabled is False


# ---------------------------------------------------------------------------
# Backbone / masked-EEG preprocessing wiring
# ---------------------------------------------------------------------------

class TestBackboneWiring:
    def _finetune(self, labram_plus=None):
        kw = dict(num_classes=2, use_mean_pooling=True, use_abs_pos_emb=True,
                  use_rel_pos_bias=False, qkv_bias=False, init_values=0.1)
        if labram_plus is not None:
            kw['labram_plus'] = labram_plus
        return create_model('labram_base_patch200_200', pretrained=False, **kw)

    def test_disabled_backbone_is_identity_preprocess(self):
        model = self._finetune()
        x = torch.randn(2, 4, 4, 200)
        assert torch.equal(model.maybe_preprocess_input(x), x)

    def test_enabled_backbone_preprocesses(self):
        model = self._finetune(LaBraMPlusConfig(enabled=True))
        x = torch.randn(2, 4, 4, 200)
        out = model.maybe_preprocess_input(x)
        assert not torch.equal(out, x)
        expected = z_score_per_patch(common_average_reference(x), model.labram_plus.z_score_eps)
        assert torch.allclose(out, expected)

    def test_enabled_backbone_forward(self):
        model = self._finetune(LaBraMPlusConfig(enabled=True))
        ci = get_channel_indices(['FP1', 'FP2', 'F3', 'F4'])
        out = model(torch.randn(2, 4, 4, 200) * 0.1, channel_indices=ci)
        assert out.shape == (2, 2)

    def test_masked_eeg_preprocesses_when_enabled(self):
        model = create_model('labram_base_patch200_1600_8k_vocab', pretrained=False,
                             vocab_size=32, init_values=0.1,
                             labram_plus=LaBraMPlusConfig(enabled=True))
        # the student backbone carries the preprocessing config
        x = torch.randn(2, 4, 8, 200)
        assert not torch.equal(model.student.maybe_preprocess_input(x), x)


# ---------------------------------------------------------------------------
# Config files
# ---------------------------------------------------------------------------

class TestConfigFiles:
    @pytest.mark.parametrize("path,cls_name", [
        ("labram/configs/defaults/vqnsp_labram_plus_plus.json", "VQNSPRunConfig"),
        ("labram/configs/defaults/pretrain_labram_plus_plus.json", "PretrainRunConfig"),
        ("labram/configs/defaults/finetune_tuab_labram_plus_plus.json", "FinetuneRunConfig"),
    ])
    def test_labram_plus_configs_enable_mode(self, path, cls_name):
        from labram.configs import run_configs
        cls = getattr(run_configs, cls_name)
        cfg = cls.load_config(path)
        assert cfg.labram_plus.enabled is True
        assert cfg.labram_plus.resolved_phase_loss == "sincos"

    @pytest.mark.parametrize("path,cls_name", [
        ("labram/configs/defaults/vqnsp.json", "VQNSPRunConfig"),
        ("labram/configs/defaults/pretrain.json", "PretrainRunConfig"),
        ("labram/configs/defaults/finetune_tuab.json", "FinetuneRunConfig"),
    ])
    def test_default_configs_keep_mode_disabled(self, path, cls_name):
        from labram.configs import run_configs
        cls = getattr(run_configs, cls_name)
        cfg = cls.load_config(path)
        assert cfg.labram_plus.enabled is False
        assert cfg.labram_plus.resolved_phase_loss == "angle"


# ---------------------------------------------------------------------------
# Device-robustness options: band-limit and amplitude normalization
# ---------------------------------------------------------------------------

def _tone(freq_hz, amp=1.0, n_ch=3, seconds=10, fs=200):
    t = torch.arange(seconds * fs, dtype=torch.float32) / fs
    sig = amp * torch.sin(2 * math.pi * freq_hz * t)
    return sig.repeat(1, n_ch, 1).reshape(1, n_ch, seconds, fs)       # [B, N, A, T]


class TestDeviceRobustness:
    def test_band_limit_keeps_passband_and_removes_stopband(self):
        from labram.data.preprocess import band_limit
        x = _tone(10.0) + _tone(60.0) + _tone(0.2)
        y = band_limit(x, 1.0, 45.0)
        spec = torch.fft.rfft(y.reshape(1, 3, -1)[0, 0])
        f = torch.fft.rfftfreq(2000, 1 / 200)
        mag = spec.abs() / 1000
        assert mag[(f - 10).abs().argmin()] == pytest.approx(1.0, abs=0.02)
        assert mag[(f - 60).abs().argmin()] < 0.02
        assert mag[(f - 0.2).abs().argmin()] < 0.01
        assert y.shape == x.shape and y.dtype == x.dtype

    def test_band_limit_without_highpass(self):
        from labram.data.preprocess import band_limit
        x = _tone(0.5) + _tone(70.0)
        y = band_limit(x, 0.0, 45.0)
        mag = torch.fft.rfft(y.reshape(1, 3, -1)[0, 0]).abs() / 1000
        f = torch.fft.rfftfreq(2000, 1 / 200)
        assert mag[(f - 0.5).abs().argmin()] == pytest.approx(1.0, abs=0.02)
        assert mag[(f - 70).abs().argmin()] < 0.02

    def test_amplitude_normalize_is_gain_invariant(self):
        from labram.data.preprocess import amplitude_normalize
        torch.manual_seed(0)
        x = torch.randn(2, 5, 10, 200) * torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])[None, :, None, None]
        x[:, 4] = 0.0                                               # a missing (all-zero) channel
        y1 = amplitude_normalize(x, target=0.1)
        y2 = amplitude_normalize(7.5 * x, target=0.1)
        assert torch.allclose(y1, y2, atol=1e-5)
        std = y1.reshape(2, 5, -1).std(dim=-1)[:, :4]
        assert torch.allclose(std.median(dim=1).values, torch.full((2,), 0.1), atol=0.01)
        assert torch.all(y1[:, 4] == 0)

    def test_options_are_off_by_default_and_validated(self):
        cfg = LaBraMPlusConfig(enabled=True, common_average_reference=False, z_score_patches=False)
        assert not cfg.use_band and not cfg.use_amplitude_norm and not cfg.preprocesses_input
        cfg = LaBraMPlusConfig(enabled=True, band_hz=[1.0, 45.0], amplitude_norm="window")
        assert cfg.use_band and cfg.use_amplitude_norm and cfg.preprocesses_input
        assert not LaBraMPlusConfig(enabled=False, band_hz=[1.0, 45.0]).use_band
        with pytest.raises(ValueError):
            LaBraMPlusConfig(amplitude_norm="recording")
        with pytest.raises(ValueError):
            LaBraMPlusConfig(band_hz=[45.0, 1.0])

    def test_preprocess_applies_band_and_amplitude(self):
        # identical channels: CAR off, or it would cancel the test signal
        cfg = LaBraMPlusConfig(enabled=True, common_average_reference=False, z_score_patches=False,
                               band_hz=[1.0, 45.0], amplitude_norm="window", amplitude_target=0.1)
        torch.manual_seed(0)
        x = 3.0 * (_tone(10.0, n_ch=4) + _tone(60.0, n_ch=4))
        x = x + torch.randn_like(x) * 0.001
        y = apply_labram_plus_preprocess(x, cfg)
        assert torch.allclose(apply_labram_plus_preprocess(5 * x, cfg), y, atol=1e-4)
        mag = torch.fft.rfft(y.reshape(1, 4, -1)[0, 0]).abs()
        f = torch.fft.rfftfreq(2000, 1 / 200)
        assert mag[(f - 60).abs().argmin()] < 0.02 * mag[(f - 10).abs().argmin()]


class TestDeviceAugmentations:
    def test_random_gain_range_and_probability(self):
        from labram.train.augment import random_gain
        torch.manual_seed(0)
        x = torch.ones(2000, 3, 50)
        y = random_gain(x, 1.0, 0.55, 1.8)
        g = y[:, 0, 0]
        assert g.min() >= 0.55 - 1e-6 and g.max() <= 1.8 + 1e-6
        assert torch.allclose(y, g[:, None, None].expand_as(y))           # one gain per sample
        assert torch.equal(random_gain(x, 0.0, 0.55, 1.8), x)

    def test_random_filter_low_and_high(self):
        from labram.train.augment import random_filter
        t = torch.arange(2000, dtype=torch.float32) / 200
        x = (torch.sin(2 * math.pi * 10 * t) + torch.sin(2 * math.pi * 70 * t)
             + torch.sin(2 * math.pi * 0.05 * t)).repeat(4, 2, 1)
        f = torch.fft.rfftfreq(2000, 1 / 200)
        low = torch.fft.rfft(random_filter(x, 1.0, 40.0, 45.0, "low")[0, 0]).abs() / 1000
        assert low[(f - 10).abs().argmin()] == pytest.approx(1.0, abs=0.02)
        assert low[(f - 70).abs().argmin()] < 0.02
        high = torch.fft.rfft(random_filter(x, 1.0, 0.5, 1.0, "high")[0, 0]).abs() / 1000
        assert high[(f - 10).abs().argmin()] == pytest.approx(1.0, abs=0.02)
        assert high[(f - 0.05).abs().argmin()] < 0.02
        assert torch.equal(random_filter(x, 0.0, 40.0, 45.0, "low"), x)

    def test_augment_batch_runs_device_options_only_when_enabled(self):
        from labram.configs.train_config import AugmentConfig
        from labram.train.augment import augment_batch
        x = torch.randn(4, 3, 2000)
        off = AugmentConfig(enabled=True, hemisphere_flip_prob=0, sign_flip_prob=0, ft_surrogate_prob=0,
                            channel_dropout_prob=0, time_mask_prob=0)
        assert torch.equal(augment_batch(x, off), x)
        on = AugmentConfig(enabled=True, hemisphere_flip_prob=0, sign_flip_prob=0, ft_surrogate_prob=0,
                           channel_dropout_prob=0, time_mask_prob=0, gain_prob=1.0, lowpass_prob=1.0)
        y = augment_batch(x, on)
        assert y.shape == x.shape and not torch.equal(y, x)
