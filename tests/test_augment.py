"""Label-preserving EEG augmentations (augment.*) and the model-owned input clip."""
import pytest
import torch

from labram.configs.train_config import AugmentConfig
from labram.data.eeg_constants import TUH_EEG_CH_NAMES, normalize_ch_names
from labram.train.augment import (
    augment_batch, channel_dropout, ft_surrogate, hemisphere_flip, hemisphere_permutation,
    sign_flip, time_mask)

CH = normalize_ch_names(TUH_EEG_CH_NAMES)


def _x(b=8, n=23, t=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, t, generator=g)


def _power(x):
    return torch.fft.rfft(x, dim=-1).abs() ** 2


def test_hemisphere_permutation_pairs_tuh_channels_and_is_an_involution():
    perm = hemisphere_permutation(CH)
    pairs = {(CH[i], CH[j]) for i, j in enumerate(perm) if i < j}
    assert pairs == {("FP1", "FP2"), ("F3", "F4"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2"),
                     ("F7", "F8"), ("T3", "T4"), ("T5", "T6"), ("A1", "A2"), ("T1", "T2")}
    assert [CH[i] for i, j in enumerate(perm) if i == j] == ["FZ", "CZ", "PZ"]
    assert [perm[j] for j in perm] == list(range(len(CH)))


def test_hemisphere_flip_swaps_channels_and_keeps_midline():
    x = _x()
    y = hemisphere_flip(x, hemisphere_permutation(CH), prob=1.0)
    assert torch.equal(y[:, CH.index("FP1")], x[:, CH.index("FP2")])
    assert torch.equal(y[:, CH.index("CZ")], x[:, CH.index("CZ")])


def test_sign_flip_keeps_the_power_spectrum():
    x = _x()
    y = sign_flip(x, prob=1.0)
    assert torch.equal(y, -x)
    assert torch.allclose(_power(y), _power(x))


def test_ft_surrogate_keeps_power_and_cross_channel_phase_but_changes_the_waveform():
    torch.manual_seed(0)
    x = _x()
    y = ft_surrogate(x, prob=1.0)
    assert y.dtype == x.dtype and y.shape == x.shape
    assert torch.allclose(_power(y), _power(x), rtol=1e-3, atol=1e-2)
    fx, fy = torch.fft.rfft(x, dim=-1), torch.fft.rfft(y, dim=-1)
    # The phase added is shared by all channels, so channel-to-channel phase
    # differences (coherence structure) are preserved.
    cross_x = fx[:, 0] * fx[:, 1].conj()
    cross_y = fy[:, 0] * fy[:, 1].conj()
    assert torch.allclose(cross_x, cross_y, rtol=1e-3, atol=1e-1)
    assert not torch.allclose(x, y, atol=1e-2)


def test_channel_dropout_zeroes_whole_channels_and_never_all_of_them():
    torch.manual_seed(0)
    x = _x(b=64)
    y = channel_dropout(x, prob=1.0, p_channel=0.1)
    zeroed = (y == 0).all(dim=-1)
    assert 0.05 < zeroed.float().mean() < 0.15
    assert torch.equal(y[~zeroed], x[~zeroed])
    assert (~channel_dropout(x, prob=1.0, p_channel=1.0).eq(0).all(dim=-1)).sum(dim=1).min() == 1


def test_time_mask_zeroes_segments_across_all_channels():
    torch.manual_seed(0)
    x = _x()
    y = time_mask(x, prob=1.0, max_count=2, min_len=100, max_len=200)
    masked = (y == 0).all(dim=1)                      # (B, T): every channel zero
    per_sample = masked.sum(dim=1)
    assert (per_sample >= 100).all() and (per_sample <= 400).all()
    assert torch.equal(y[~masked[:, None, :].expand_as(y)], x[~masked[:, None, :].expand_as(y)])


def test_disabled_or_zero_probabilities_are_the_identity():
    x = _x()
    assert augment_batch(x, AugmentConfig(enabled=False), CH) is x
    off = AugmentConfig(enabled=True, hemisphere_flip_prob=0, sign_flip_prob=0,
                        ft_surrogate_prob=0, channel_dropout_prob=0, time_mask_prob=0)
    assert torch.equal(augment_batch(x, off, CH), x)


def test_full_augmentation_keeps_shape_and_changes_about_the_configured_fraction():
    torch.manual_seed(0)
    x = _x(b=256, t=400)
    only_sign = AugmentConfig(enabled=True, hemisphere_flip_prob=0, sign_flip_prob=0.5,
                              ft_surrogate_prob=0, channel_dropout_prob=0, time_mask_prob=0)
    y = augment_batch(x, only_sign, CH)
    flipped = torch.isclose(y, -x).all(dim=(1, 2)).float().mean()
    assert 0.4 < flipped < 0.6
    y = augment_batch(x, AugmentConfig(enabled=True), CH)
    assert y.shape == x.shape and torch.isfinite(y).all()


def test_model_clips_its_input_before_car_and_only_when_enabled():
    from timm.models import create_model
    import labram.models.registry  # noqa: F401
    x = torch.randn(1, 2, 1, 200) * 20                # model-input units (uV / 100)
    m = create_model("labram_base_patch200_200", num_classes=1, init_values=0.1, input_clip=8.0)
    assert m.maybe_preprocess_input(x).abs().max() <= 8.0
    assert torch.equal(create_model("labram_base_patch200_200", num_classes=1, init_values=0.1)
                       .maybe_preprocess_input(x), x)


def test_input_clip_uv_reaches_the_model_and_is_validated():
    from labram.configs.model_config import CodebookRegConfig, FinetuneModelConfig
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs.run_finetune import get_model
    cfg = FinetuneRunConfig.load_config(
        "labram/configs/defaults/finetune_tuab_age.json",
        **{"model.input_clip_uv": 800.0, "finetune_checkpoint.finetune": ""})
    cfg.model.nb_classes = 1
    assert get_model(cfg).input_clip == pytest.approx(8.0)
    with pytest.raises(ValueError, match="input_clip_uv"):
        FinetuneModelConfig(task="regression", input_clip_uv=-1).validate()
    with pytest.raises(ValueError, match="codebook_reg"):
        FinetuneModelConfig(task="regression", input_clip_uv=800,
                            codebook_reg=CodebookRegConfig(enabled=True)).validate()


def test_train_one_epoch_with_augmentation_and_mixup_runs():
    from torch import nn
    from labram.configs.train_config import MixupConfig
    from labram.train.train_finetune import train_one_epoch
    from test_anti_memorization import _regression_loader
    from test_finetune import _make_epoch_args, _make_model
    model = _make_model(num_classes=1)
    args = _make_epoch_args(model, _regression_loader(), nn.HuberLoss(), is_binary=False)
    args.update(task="regression", nb_classes=1, target_stats=(10.0, 5.0),
                ch_names=["FP1", "FP2", "CZ", "O1"],
                mixup_cfg=MixupConfig(enabled=True, alpha=0.4),
                augment_cfg=AugmentConfig(enabled=True))
    stats = train_one_epoch(**args)
    assert stats["mae"] >= 0
