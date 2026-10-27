"""channel_drop: missing (all-zero) channels are left out per sample, an
augmented batch drops n ~ U{min..max} candidate channels from every sample,
the consistency loss pulls the dropped feature towards a frozen all-channels
pass, and mixup never half-mixes a missing channel."""
from collections import Counter

import numpy as np
import pytest
import torch

import labram.models.registry  # noqa: F401
from labram.configs.model_config import TransformerArchConfig
from labram.configs.train_config import ChannelDropConfig
from labram.data.eeg_constants import TUH_EEG_CH_NAMES, normalize_ch_names
from labram.models.neural_transformer import NeuralTransformer
from labram.train.channel_drop import ChannelDropper

CH = normalize_ch_names(TUH_EEG_CH_NAMES)
T12 = [CH.index("T1"), CH.index("T2")]


def _model(**kw):
    torch.manual_seed(0)
    return NeuralTransformer(TransformerArchConfig(
        eeg_window_size=200, patch_size=200, in_chans=1, out_chans=8, num_classes=1,
        embed_dim=200, depth=2, num_heads=10, init_values=0.1, init_scale=1.0,
        drop_path_rate=0.2, **kw)).eval()


def _batch(missing_rows=(1, 3), n=5):
    torch.manual_seed(1)
    x = torch.randn(n, 23, 1, 200)
    for r in missing_rows:
        x[r, T12] = 0
    return x


def _dropper(**kw):
    cfg = dict(enabled=True, channels=["T1", "T2"], min_drop=1, max_drop=2)
    cfg.update(kw)
    return ChannelDropper(ChannelDropConfig(**cfg), CH)


def test_present_channels_per_sample():
    p = ChannelDropper.present(_batch())
    assert p.shape == (5, 23) and p[[0, 2, 4]].all()
    assert not p[1, T12].any() and p[1].sum() == 21


def test_draw_respects_the_range_and_candidates():
    d = _dropper()
    g = torch.Generator().manual_seed(0)
    draws = [tuple(d.draw(g)) for _ in range(400)]
    assert set(draws) == {(T12[0],), (T12[1],), tuple(T12)}
    assert 0.25 < Counter(len(x) for x in draws)[2] / 400 < 0.75
    any_ch = _dropper(channels=[], min_drop=3, max_drop=3)
    assert len(any_ch.candidates) == 23 and len(any_ch.draw(g)) == 3


@pytest.mark.parametrize("kw, match", [(dict(min_drop=0), "min_drop"), (dict(max_drop=3), "exceeds"),
                                       (dict(channels=["XX"]), "XX")])
def test_bad_settings(kw, match):
    with pytest.raises(ValueError, match=match):
        _dropper(**kw)


def test_off_unless_enabled():
    assert ChannelDropper.from_config(ChannelDropConfig(channels=["T1"]), CH) is None
    assert ChannelDropper.from_config(ChannelDropConfig(enabled=True), CH) is not None


def test_grouped_forward_equals_a_per_sample_forward():
    model, d, x = _model(), _dropper(), _batch()
    present = d.present(x)
    present[0, 5] = False                                   # a third pattern
    out, feat = d.forward(model, x, present, return_features=True)
    for i in range(len(x)):
        chans = present[i].nonzero().squeeze(1)
        ci = [0] + [d.ci_all[int(c) + 1] for c in chans]
        ref, ref_f = model(x[i:i + 1][:, chans], ci, return_features=True)
        assert torch.allclose(out[i], ref[0], atol=1e-5) and torch.allclose(feat[i], ref_f[0], atol=1e-5)


def test_dropping_is_not_the_same_as_zeros():
    model, d = _model(), _dropper()
    x = _batch(missing_rows=(0,))
    assert not torch.allclose(d.forward(model, x[:1], d.present(x[:1])), model(x[:1], d.ci_all))


def test_frozen_full_pass_consistency():
    model, d, x = _model(), _dropper(), _batch()
    model.train()
    present = d.present(x)
    dropped = d.drop_mask(present, T12)
    run = present & ~dropped
    _, feat = d.forward(model, x, run, return_features=True)
    cons = d.consistency(model, x, present, present & ~run, feat)
    assert model.training                                    # mode restored
    model.eval()
    with torch.no_grad():
        _, f_full = d.forward(model, x[[0, 2, 4]], present[[0, 2, 4]], return_features=True)
    assert torch.allclose(cons, (f_full - feat[[0, 2, 4]]).pow(2).mean(), rtol=1e-3)
    # gradient flows only through the dropped pass
    model.train()
    model.zero_grad()
    _, feat = d.forward(model, x, run, return_features=True)
    cons = d.consistency(model, x, present, present & ~run, feat)
    cons.backward()
    assert model.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
    # nobody lost a channel -> 0
    assert d.consistency(model, x[[1, 3]], present[[1, 3]], (present & ~run)[[1, 3]],
                         feat[[1, 3]]) == 0


def test_mixup_keeps_only_channels_both_partners_have():
    from labram.train.mixup import mixup_batch
    d, x = _dropper(), _batch()
    had = d.present(x)
    torch.manual_seed(3)
    mixed, _, lam, perm = mixup_batch(x, torch.zeros(len(x), 1), 0.4, return_perm=True)
    mixed = d.zero_missing(mixed, had & had[perm])
    assert torch.equal(d.present(mixed), had & had[perm])


def test_run_finetune_with_channel_drop_end_to_end(tmp_path):
    """One recording lacks T1/T2; class-token head; every batch augmented with
    1-2 of T1/T2 dropped; val/test report the dropped-channel metrics."""
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs import run_finetune
    from test_window_qc import _corpus
    root = _corpus(tmp_path / "data")
    rec = root / "recordings" / "bbb_s001_t000.npy"
    arr = np.load(rec)
    arr[:, T12] = 0
    np.save(rec, arr)
    out = tmp_path / "out"
    cfg = FinetuneRunConfig.load_config(
        "labram/configs/defaults/finetune_tuab_age.json",
        **{"data.data_path": str(root), "data.data_format": "npy",
           "data.trim_start_sec": 0.0, "data.trim_end_sec": 0.0, "data.eval_minutes": 0.0,
           "data.num_workers": 0, "trainer.debug": True, "trainer.epochs": 1,
           "distributed.device": "cpu", "finetune_checkpoint.finetune": "",
           "clearml.enabled": False, "output.output_dir": str(out), "output.log_dir": str(out),
           "model.use_mean_pooling": False, "mixup.enabled": True,
           "channel_drop.enabled": True, "channel_drop.channels": ["T1", "T2"],
           "channel_drop.min_drop": 1, "channel_drop.max_drop": 2, "channel_drop.prob": 1.0,
           "logging.log_model_graph": False})
    summary = run_finetune.main(cfg)
    val = summary["best_val_stats"]
    assert "channel_drop_consistency" in summary["best_train_stats"]
    assert "channel_drop_window_mae" in val and "channel_drop_consistency" in val


@pytest.mark.parametrize("start_epoch, dropped", [(0, True), (1, False)])
def test_random_drop_starts_at_start_epoch(tmp_path, monkeypatch, start_epoch, dropped):
    """channel_drop.start_epoch: with prob 1, epoch 0 drops channels only when
    the augmentation already started; missing channels are handled either way."""
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs import run_finetune
    from labram.train import train_finetune
    from test_window_qc import _corpus
    seen = []
    real = train_finetune.train_channel_drop_batch

    def spy(model, samples, target, criterion, dropper, drop_all, **kw):
        seen.append(drop_all)
        return real(model, samples, target, criterion, dropper, drop_all, **kw)

    monkeypatch.setattr(train_finetune, "train_channel_drop_batch", spy)
    root = _corpus(tmp_path / "data")
    out = tmp_path / "out"
    cfg = FinetuneRunConfig.load_config(
        "labram/configs/defaults/finetune_tuab_age.json",
        **{"data.data_path": str(root), "data.data_format": "npy",
           "data.trim_start_sec": 0.0, "data.trim_end_sec": 0.0, "data.eval_minutes": 0.0,
           "data.num_workers": 0, "trainer.debug": True, "trainer.epochs": 1,
           "distributed.device": "cpu", "finetune_checkpoint.finetune": "",
           "clearml.enabled": False, "output.output_dir": str(out), "output.log_dir": str(out),
           "channel_drop.enabled": True, "channel_drop.channels": [],
           "channel_drop.min_drop": 1, "channel_drop.max_drop": 3, "channel_drop.prob": 1.0,
           "channel_drop.start_epoch": start_epoch, "logging.log_model_graph": False})
    run_finetune.main(cfg)
    assert seen and all(d is dropped for d in seen)
