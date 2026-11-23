"""Handcrafted window features inside the model (mil.window_features; labram/layers/window_features.py)."""
import glob
import os

import numpy as np
import pytest
import torch

from labram.layers.window_features import (BANDS, FILTERBANK, N_CHANNELS, HandcraftedWindowFeatures,
                                           WindowFeatureBranch, riemann_mean)


def _windows(n=6, seed=0):
    """``[n, 19, 2000]`` µV: 1/f-like noise shared between channels plus a posterior 10 Hz rhythm."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(size=(n, 1, 2000)), axis=-1) * 0.8
    x = common + np.cumsum(rng.normal(size=(n, N_CHANNELS, 2000)), axis=-1) * 0.5 + rng.normal(size=(n, N_CHANNELS, 2000)) * 4
    x[:, 6:10] += 20 * np.sin(2 * np.pi * 10 * np.arange(2000) / 200.0 + rng.uniform(0, 6, size=(n, 1, 1)))
    return (x - x.mean(-1, keepdims=True)).astype(np.float32)


def test_bands_match_the_feature_toolkit():
    fx = pytest.importorskip("labram.features.extractors")
    assert BANDS == fx.BANDS and FILTERBANK == fx.FILTERBANK


def test_features_match_the_numpy_reference():
    """Spectral, spectrum-bin, connectivity and covariance values equal labram/features/extractors.py."""
    fx = pytest.importorskip("labram.features.extractors")
    x = _windows()
    m = HandcraftedWindowFeatures()
    core, cov, conn = m.raw(torch.from_numpy(x))
    car = x - x.mean(1, keepdims=True)
    freqs, psd = fx.welch_psd(car.astype(np.float64))
    core = core.reshape(len(x), N_CHANNELS, -1).numpy()
    np.testing.assert_allclose(core[..., :17], fx.spectral_features(freqs, psd)[0], rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(core[..., 17:61], fx.psd_bin_features(freqs, psd)[0], rtol=2e-3, atol=2e-3)
    # time domain: the same definitions on the same 1-45 Hz signal (FFT band limit of the window)
    full = np.fft.rfft(car.astype(np.float64), axis=-1)
    ff = np.fft.rfftfreq(2000, 1 / 200.0)
    xb = np.fft.irfft(full * ((ff >= 1.0) & (ff <= 45.0)), n=2000, axis=-1)
    np.testing.assert_allclose(core[..., 61:], fx.time_features(xb)[0], rtol=5e-3, atol=5e-3)
    f, csd = fx.cross_spectra(car.astype(np.float64))
    np.testing.assert_allclose(conn.numpy(), fx.band_connectivity(f, csd)[0], rtol=5e-3, atol=5e-4)
    ref = fx.band_covariance(f, csd)                                   # [n, 6, 19, 19]
    b = m.basis.numpy()
    ref = b.T @ ref @ b
    ref = (1 - m.shrink) * ref + m.shrink * np.trace(ref, axis1=-2, axis2=-1)[..., None, None] / 18 * np.eye(18)
    np.testing.assert_allclose(cov.numpy(), ref, rtol=2e-3, atol=1e-6)
    assert m.n_features == core.shape[1] * core.shape[2] + 6 * 171 + conn.shape[1] == 4028


def test_fit_standardizes_and_round_trips():
    x = torch.from_numpy(_windows(n=40, seed=1))
    m = HandcraftedWindowFeatures().fit(x)
    z = m(x)
    assert bool(m.fitted) and z.shape == (40, m.n_features) and torch.isfinite(z).all()
    assert z.mean(0).abs().median() < 0.05 and (z.std(0).median() - 1).abs() < 0.1 and z.abs().max() <= 6
    # tangent vectors at the fitted reference are centred: the mean covariance maps to ~0
    core, cov, conn = m.raw(x)
    assert m._tangent(cov).mean(0).abs().max() < 0.05
    other = HandcraftedWindowFeatures()
    other.load_state_dict(m.state_dict())
    torch.testing.assert_close(other(x), z)


def test_riemann_mean_of_scaled_copies():
    a = torch.randn(18, 18, dtype=torch.float64)
    c = a @ a.T + torch.eye(18, dtype=torch.float64)
    mean = riemann_mean(torch.stack([c * 0.5, c * 2.0]))               # geometric mean of 0.5 c and 2 c is c
    torch.testing.assert_close(mean, c, rtol=1e-6, atol=1e-8)


def test_branch_places_a_channel_subset_and_freezes():
    x = torch.from_numpy(_windows(n=8, seed=2))
    branch = WindowFeatureBranch(dim=16, dropout=0.0)
    branch.extract.fit(x)
    ci = [0] + list(range(1, 20))                                       # class token + 19 channels
    branch.set_channels(ci)
    x4 = (x / 100).reshape(8, N_CHANNELS, 10, 200)
    full = branch(x4)
    assert full.shape == (8, 16) and full.requires_grad
    # the same windows given without channel 3 (absent), with the indices that are there
    keep = [c for c in range(N_CHANNELS) if c != 3]
    zeroed = x4.clone()
    zeroed[:, 3] = 0
    torch.testing.assert_close(branch(x4[:, keep], [0] + [ci[c + 1] for c in keep]), branch(zeroed))
    assert not branch(x4, frozen=True).requires_grad
    with pytest.raises(ValueError):
        WindowFeatureBranch().set_channels([0] + list(range(1, 24)))    # 23 channels: drop A1/A2/T1/T2 first


def test_run_finetune_with_window_features(tmp_path):
    """Phased bag training with [class token, features] as the window input, end to end on the npy fixture."""
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs import run_finetune
    from test_window_qc import _corpus
    root = _corpus(tmp_path / "data")
    out = tmp_path / "out"
    cfg = FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab_age.json", **{
        "data.data_path": str(root), "data.data_format": "npy", "data.trim_start_sec": 0.0, "data.trim_end_sec": 0.0,
        "data.eval_minutes": 0.0, "data.num_workers": 0, "data.drop_channels": ["A1", "A2", "T1", "T2"],
        "trainer.batch_size": 2, "trainer.epochs": 3, "optimizer.warmup_epochs": 0, "distributed.device": "cpu",
        "finetune_checkpoint.finetune": "", "clearml.enabled": False, "output.output_dir": str(out),
        "output.log_dir": str(out), "logging.log_model_graph": False, "evaluation.agg_windows": "mean",
        "channel_drop.enabled": True, "channel_drop.prob": 0.0, "channel_drop.consistency_weight": 0.0,
        "mixup.enabled": True, "mil.enabled": True, "mil.bag_size": 3, "mil.pool": "gated", "mil.gated": False,
        "mil.window_epochs": 1, "mil.window_minutes_per_epoch": 0.5, "mil.window_batch_size": 6,
        "mil.freeze_encoder_epochs": 1, "mil.freeze_head": True, "mil.window_loss_weight": 0.5,
        "mil.head_lr_scale": 5.0, "mil.window_features": "handcrafted", "mil.feature_dim": 8,
        "mil.feature_fit_batches": 3})
    summary = run_finetune.main(cfg)
    assert np.isfinite(summary["best_val_stats"]["mae"]) and "window_mae" in summary["best_val_stats"]
    ckpt = glob.glob(os.path.join(str(out) + "*", "checkpoint-best.pth"))
    state = torch.load(ckpt[0], map_location="cpu", weights_only=False)["model"]
    assert bool(state["window_feat.extract.fitted"]) and int(state["window_feat.channel_index"].min()) > 0
    model = run_finetune.get_model(cfg)
    model.load_state_dict(state)
    assert model.head.in_features == model.num_features + 8 == model.mil.pool.dim


# ------------------------------------------------------------------ optional blocks (mil.feature_extra,
# mil.recording_features)
def test_extra_blocks_have_their_sizes_and_fit():
    from labram.layers.window_features import N_EXTRA
    x = torch.from_numpy(_windows(n=30, seed=3))
    m = HandcraftedWindowFeatures(extra=("microstates", "aec", "complexity")).fit(x)
    assert m.n_features == 4028 + sum(N_EXTRA.values())
    torch.testing.assert_close(m.ms_maps.norm(dim=1), torch.ones(5), rtol=1e-4, atol=1e-4)
    z = m(x)
    assert z.shape == (30, m.n_features) and torch.isfinite(z).all()
    v = m.vector(x)
    ms = v[:, 4028:4028 + 36]
    torch.testing.assert_close(ms[:, :5].sum(1), torch.ones(30), rtol=1e-5, atol=1e-5)   # coverage sums to 1
    aec = v[:, 4028 + 36:4028 + 36 + 855]
    assert (aec.abs() <= 1 + 1e-4).all()


def test_aec_matches_a_direct_computation():
    x = torch.from_numpy(_windows(n=2, seed=4))
    m = HandcraftedWindowFeatures(extra=("aec",))
    v = m._aec(m.car(x))
    xc = m.car(x)
    f = torch.fft.fftfreq(2000, 1 / 200.0)
    z = torch.fft.ifft(torch.fft.fft(xc, dim=-1) * (((f >= 8) & (f < 13)).float() * 2), dim=-1)

    def corr(i, j):
        yp = (z[1, j] * (z[1, i] / z[1, i].abs()).conj()).imag.abs()
        return torch.corrcoef(torch.stack([z[1, i].abs(), yp]))[0, 1]
    pairs = torch.triu_indices(19, 19, 1)
    k = int(((pairs[0] == 3) & (pairs[1] == 11)).nonzero())
    torch.testing.assert_close(v[1, 2 * 171 + k], (corr(3, 11) + corr(11, 3)) / 2, rtol=1e-3, atol=1e-4)


def test_learned_filterbank_trains():
    from labram.layers.window_features import LearnedFilterbankCov
    x = torch.from_numpy(_windows(n=8, seed=5))
    g = LearnedFilterbankCov(n_filters=4, rank=3, out_dim=5)
    out = g(x - x.mean(1, keepdim=True))
    assert out.shape == (8, 5)
    out.pow(2).sum().backward()
    for p in (g.log_f, g.log_bw, g.bimap):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_recording_features_are_looked_up_and_standardised(tmp_path):
    from labram.layers.window_features import RecordingFeatures
    path = tmp_path / "rec.npz"
    np.savez(path, stems=np.array(["a", "b", "c"]), features=np.array([[1., 2.], [3., np.nan], [5., 6.]], np.float32))
    r = RecordingFeatures(2, out_dim=3)
    assert r.load_table(str(path)) == 3
    raw = r.lookup(["a", "b", "c", "missing"], "cpu")
    assert torch.isnan(raw[3]).all() and torch.isnan(raw[1, 1])
    r.fit(raw[:3])
    torch.testing.assert_close(r.mean, torch.tensor([3.0, 4.0]))
    out = r(raw)
    torch.testing.assert_close(out[3], r.proj.bias)                 # unknown recording -> the training mean


def test_slow_dynamics_dfa_exponent():
    from labram.features.slow_dynamics import dfa_exponent, slow_dynamics
    rng = np.random.default_rng(0)
    white = rng.normal(size=(3, 20000))
    assert np.allclose(dfa_exponent(white, 25.0), 0.5, atol=0.08)
    walk = np.cumsum(white, axis=1)
    assert np.allclose(dfa_exponent(walk, 25.0), 1.5, atol=0.1)
    f = slow_dynamics(rng.normal(size=(19, 200 * 120)) * 10)
    assert f.shape == (19 * 3 * 4,) and np.isfinite(f).all()


def test_run_finetune_with_every_extra_block(tmp_path):
    """The phased run with all optional blocks and a per-recording table; the checkpoint restores them."""
    import json
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs import run_finetune
    from test_window_qc import _corpus
    root = _corpus(tmp_path / "data")
    recs = json.load(open(os.path.join(root, "manifest.json")))["recordings"]
    stems = [r["stem"] for r in (recs if isinstance(recs, list) else recs.values())]
    np.savez(os.path.join(root, "slow_test.npz"), stems=np.array(stems),
             features=np.random.default_rng(0).normal(size=(len(stems), 228)).astype(np.float32))
    out = tmp_path / "out"
    cfg = FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab_age.json", **{
        "data.data_path": str(root), "data.data_format": "npy", "data.trim_start_sec": 0.0, "data.trim_end_sec": 0.0,
        "data.eval_minutes": 0.0, "data.num_workers": 0, "data.drop_channels": ["A1", "A2", "T1", "T2"],
        "trainer.batch_size": 2, "trainer.epochs": 3, "optimizer.warmup_epochs": 0, "distributed.device": "cpu",
        "finetune_checkpoint.finetune": "", "clearml.enabled": False, "output.output_dir": str(out),
        "output.log_dir": str(out), "logging.log_model_graph": False, "evaluation.agg_windows": "mean",
        "evaluation.test_sets": [],
        "channel_drop.enabled": True, "channel_drop.prob": 0.0, "channel_drop.consistency_weight": 0.0,
        "mixup.enabled": True, "mil.enabled": True, "mil.bag_size": 3, "mil.pool": "gated", "mil.gated": False,
        "mil.window_epochs": 1, "mil.window_minutes_per_epoch": 0.5, "mil.window_batch_size": 6,
        "mil.freeze_encoder_epochs": 1, "mil.freeze_head": True, "mil.window_loss_weight": 0.5,
        "mil.head_lr_scale": 5.0, "mil.window_features": "handcrafted", "mil.feature_dim": 8,
        "mil.feature_fit_batches": 3, "mil.feature_extra": ["microstates", "aec", "complexity", "green"],
        "mil.recording_features": "slow_test.npz", "mil.feature_norm": "layernorm"})
    summary = run_finetune.main(cfg)
    assert np.isfinite(summary["best_val_stats"]["mae"])
    ckpt = glob.glob(os.path.join(str(out) + "*", "checkpoint-best.pth"))
    state = torch.load(ckpt[0], map_location="cpu", weights_only=False)["model"]
    assert bool(state["window_feat.rec.fitted"]) and state["window_feat.extract.ms_maps"].abs().sum() > 0
    model = run_finetune.get_model(cfg)
    model.load_state_dict(state)
    assert model.head.in_features == model.num_features + 8 + 32 + 16 == model.mil.pool.dim
    assert model.window_feat.out_norm is not None


def test_learned_filterbank_handles_a_single_window_in_training():
    """The channel-drop path can hand the model a group of one window."""
    from labram.layers.window_features import LearnedFilterbankCov
    x = torch.from_numpy(_windows(n=6, seed=6))
    xc = x - x.mean(1, keepdim=True)
    g = LearnedFilterbankCov(n_filters=4, rank=3, out_dim=5)
    g.init_stats(xc)
    g.train()
    out = g(xc[:1])
    assert out.shape == (1, 5) and torch.isfinite(out).all()
    z = (g.vectors(xc) - g.run_mean) / torch.sqrt(g.run_var + 1e-5)
    assert z.mean(0).abs().max() < 1e-3                                # initialised on these windows


def test_feature_families_select_columns_of_the_full_vector():
    x = torch.from_numpy(_windows(n=4, seed=7))
    full = HandcraftedWindowFeatures().vector(x)
    slim = HandcraftedWindowFeatures(families=("bins", "covariance", "connectivity"))
    v = slim.vector(x)
    assert slim.n_features == 19 * 44 + 1026 + 1710 == v.shape[1]
    torch.testing.assert_close(v[:, :44], full[:, 17:61])                     # channel 1's bins come first
    torch.testing.assert_close(v[:, 19 * 44:], full[:, 19 * 68:])            # covariance + connectivity
    with pytest.raises(ValueError):
        HandcraftedWindowFeatures(families=("alpha",))
