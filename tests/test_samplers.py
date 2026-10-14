"""RecordingSubsetSampler: at most k samples per recording each epoch, redrawn
every epoch, every recording kept, sharded across ranks without overlap."""
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from labram.data.samplers import RecordingSubsetSampler


def _ds(counts):
    files = [f"train/rec{r}_s001_t000_{k}.pkl" for r, n in enumerate(counts) for k in range(n)]
    return SimpleNamespace(files=files)


def _rec(ds, i):
    return ds.files[i].split("/")[1].rsplit("_", 1)[0]


def test_at_most_k_per_recording_and_every_recording_kept():
    ds = _ds([120, 50, 3, 30])
    s = RecordingSubsetSampler(ds, 30)
    idx = list(s)
    per = Counter(_rec(ds, i) for i in idx)
    assert per == {"rec0_s001_t000": 30, "rec1_s001_t000": 30, "rec2_s001_t000": 3,
                   "rec3_s001_t000": 30}
    assert len(idx) == len(s) == 93 and len(set(idx)) == 93


def test_redrawn_every_epoch_and_reproducible():
    ds = _ds([120, 120])
    s = RecordingSubsetSampler(ds, 30, seed=1)
    e0 = list(s)
    assert list(s) == e0                                   # same epoch -> same draw
    s.set_epoch(1)
    e1 = list(s)
    assert set(e1) != set(e0)
    s.set_epoch(0)
    assert list(s) == e0


def test_shards_cover_the_epoch_without_overlap():
    ds = _ds([40, 40, 7])
    shards = [list(RecordingSubsetSampler(ds, 10, num_replicas=2, rank=r, seed=3)) for r in (0, 1)]
    assert len(shards[0]) == len(shards[1]) == 14           # 27 padded to 28
    assert set(shards[0]) | set(shards[1]) == set(RecordingSubsetSampler(ds, 10, seed=3).indices())


def test_debug_subset_and_bad_k():
    ds = _ds([20, 20])
    sub = torch.utils.data.Subset(type("D", (torch.utils.data.Dataset,), {
        "files": ds.files, "__len__": lambda self: len(self.files),
        "__getitem__": lambda self, i: i})(), [0, 1, 2, 25])
    assert len(RecordingSubsetSampler(sub, 2)) == 3
    with pytest.raises(ValueError):
        RecordingSubsetSampler(ds, 0)


def test_fused_attention_matches_the_explicit_path():
    from labram.layers.attention import Attention, set_fused_attention
    torch.manual_seed(0)
    for window in (None, (4, 5)):
        att = Attention(200, num_heads=10, qkv_bias=True, qk_norm=torch.nn.LayerNorm,
                        window_size=window).eval()
        if window:
            torch.nn.init.normal_(att.relative_position_bias_table, std=0.5)
        x = torch.randn(3, 21 if window else 31, 200)
        rel = torch.randn(1, 10, x.shape[1], x.shape[1]) if window else None
        ref = att(x, rel_pos_bias=rel)
        holder = torch.nn.Sequential(att)
        assert set_fused_attention(holder, True) == 1
        assert torch.allclose(att(x, rel_pos_bias=rel), ref, atol=1e-5)
        assert att(x, return_attention=True).shape[-1] == x.shape[1]   # maps still explicit
        set_fused_attention(holder, False)


def test_fused_attention_trains_the_same_gradients():
    from labram.layers.attention import Attention
    torch.manual_seed(0)
    att = Attention(200, num_heads=10, qkv_bias=True)
    x = torch.randn(2, 24, 200)
    att(x).pow(2).sum().backward()
    g_ref = att.qkv.weight.grad.clone()
    att.zero_grad()
    att.fused_attn = True
    att(x).pow(2).sum().backward()
    assert torch.allclose(att.qkv.weight.grad, g_ref, atol=1e-4)


def test_run_finetune_with_fast_settings(tmp_path):
    """Fused attention + a 1-minute (6-sample) per-recording draw per epoch."""
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs import run_finetune
    from test_window_qc import _corpus
    root = _corpus(tmp_path / "data")
    out = tmp_path / "out"
    cfg = FinetuneRunConfig.load_config(
        "labram/configs/defaults/finetune_tuab_age.json",
        **{"data.data_path": str(root), "data.data_format": "npy",
           "data.trim_start_sec": 0.0, "data.trim_end_sec": 0.0, "data.eval_minutes": 0.0,
           "data.num_workers": 0, "data.train_minutes_per_epoch": 1.0,
           "model.fused_attention": True, "trainer.batch_size": 4,
           "trainer.debug": True, "trainer.epochs": 2, "optimizer.warmup_epochs": 0,
           "distributed.device": "cpu", "finetune_checkpoint.finetune": "",
           "clearml.enabled": False, "output.output_dir": str(out), "output.log_dir": str(out),
           "logging.log_model_graph": False})
    summary = run_finetune.main(cfg)
    assert "mae" in summary["best_val_stats"]
    model = run_finetune.get_model(cfg)
    assert all(b.attn.fused_attn for b in model.blocks)


def test_whole_numbers_are_accepted_for_float_fields():
    from labram.configs.run_configs import FinetuneRunConfig
    cfg = FinetuneRunConfig.load_config(
        "labram/configs/defaults/finetune_tuab_age.json",
        **{"channel_drop.consistency_weight": 1, "channel_drop.prob": 0, "optimizer.lr": 1})
    assert cfg.channel_drop.consistency_weight == 1.0 and isinstance(cfg.channel_drop.prob, float)
    with pytest.raises(ValueError):
        FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab_age.json",
                                      **{"channel_drop.min_drop": 1.5})
