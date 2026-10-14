"""channel_drop: recordings without T1/T2 run without those tokens, a whole
training batch can drop them with a full-vs-dropped consistency loss, and
mixup never half-mixes a missing channel."""
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
        embed_dim=200, depth=2, num_heads=10, init_values=0.1, init_scale=1.0, **kw)).eval()


def _batch(missing_rows=(1, 3), n=5):
    x = torch.randn(n, 23, 1, 200)
    for r in missing_rows:
        x[r, T12] = 0
    return x


def _dropper(**kw):
    return ChannelDropper(ChannelDropConfig(channels=["T1", "T2"], **kw), CH)


def test_rows_without_the_channels_are_detected():
    d = _dropper()
    assert d.drop == T12 and len(d.keep) == 21 and len(d.ci_kept) == 22
    assert d.has_channels(_batch()).tolist() == [True, False, True, False, True]


def test_split_forward_equals_a_per_sample_forward():
    model, d, x = _model(), _dropper(), _batch()
    has = d.has_channels(x)
    out, feat = d.forward(model, x, has, return_features=True)
    for i in range(len(x)):
        if has[i]:
            ref, ref_f = model(x[i:i + 1], d.ci_full, return_features=True)
        else:
            ref, ref_f = model(x[i:i + 1, d.keep], d.ci_kept, return_features=True)
        assert torch.allclose(out[i], ref[0], atol=1e-5) and torch.allclose(feat[i], ref_f[0], atol=1e-5)


def test_dropping_is_not_the_same_as_zeros():
    model, d = _model(), _dropper()
    x = _batch(missing_rows=(0,))
    dropped = d.forward(model, x[:1], torch.tensor([False]))
    zeros = model(x[:1], d.ci_full)
    assert not torch.allclose(dropped, zeros)


def test_consistency_is_the_mean_squared_feature_gap_and_trains_both_passes():
    model, d, x = _model(), _dropper(), _batch()
    model.train()
    has = d.has_channels(x)
    _, feat = d.forward(model, x, torch.zeros_like(has), return_features=True)
    cons = d.consistency(model, x, has, feat)
    _, f_full = model(x[has], d.ci_full, return_features=True)
    assert torch.allclose(cons, (f_full - feat[has]).pow(2).mean(), rtol=1e-4)
    cons.backward()
    assert model.blocks[0].attn.qkv.weight.grad.abs().sum() > 0

    model.zero_grad()
    d2 = _dropper(detach_full=True)
    _, feat = d2.forward(model, x, torch.zeros_like(has), return_features=True)
    d2.consistency(model, x, has, feat).backward()          # still trains the dropped pass
    assert model.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
    assert d.consistency(model, x, torch.zeros_like(has), feat) == 0


def test_mixup_never_half_mixes_a_missing_channel():
    from labram.train.mixup import mixup_batch
    d = _dropper()
    x = _batch()
    had = d.has_channels(x)
    torch.manual_seed(3)
    mixed, _, lam, perm = mixup_batch(x, torch.zeros(len(x), 1), 0.4, return_perm=True)
    mixed = d.zero_rows(mixed, ~(had & had[perm]))
    both = (had & had[perm])
    assert torch.equal(d.has_channels(mixed), both)
    assert torch.allclose(mixed[both], (lam * x + (1 - lam) * x[perm])[both])


def test_unknown_channel_and_off():
    with pytest.raises(ValueError, match="XX"):
        ChannelDropper(ChannelDropConfig(channels=["XX"]), CH)
    assert ChannelDropper.from_config(ChannelDropConfig(), CH) is None


def test_run_finetune_with_channel_drop_end_to_end(tmp_path):
    """A cohort where one recording lacks T1/T2 (zero-filled), class-token head,
    every batch dropped (prob 1) with the consistency loss."""
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs import run_finetune
    from test_window_qc import _corpus
    root = _corpus(tmp_path / "data")
    rec = root / "recordings" / "bbb_s001_t000.npy"
    x = np.load(rec)
    x[:, T12] = 0
    np.save(rec, x)
    out = tmp_path / "out"
    cfg = FinetuneRunConfig.load_config(
        "labram/configs/defaults/finetune_tuab_age.json",
        **{"data.data_path": str(root), "data.data_format": "npy",
           "data.trim_start_sec": 0.0, "data.trim_end_sec": 0.0, "data.eval_minutes": 0.0,
           "data.num_workers": 0, "trainer.debug": True, "trainer.epochs": 1,
           "distributed.device": "cpu", "finetune_checkpoint.finetune": "",
           "clearml.enabled": False, "output.output_dir": str(out), "output.log_dir": str(out),
           "model.use_mean_pooling": False, "mixup.enabled": True,
           "channel_drop.channels": ["T1", "T2"], "channel_drop.prob": 1.0,
           "logging.log_model_graph": False})
    summary = run_finetune.main(cfg)
    assert "mae" in summary["best_val_stats"]
    assert "channel_drop_consistency" in summary["best_train_stats"]
