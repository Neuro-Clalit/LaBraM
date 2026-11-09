"""End-to-end recording-level MIL fine-tuning (mil.enabled) on the npy fixture."""
import glob
import os

import pandas as pd
import pytest
import torch


def _cfg(tmp_path, **over):
    from labram.configs.run_configs import FinetuneRunConfig
    from test_window_qc import _corpus
    root = _corpus(tmp_path / "data")
    out = tmp_path / "out"
    base = {"data.data_path": str(root), "data.data_format": "npy",
            "data.trim_start_sec": 0.0, "data.trim_end_sec": 0.0, "data.eval_minutes": 0.0,
            "data.num_workers": 0, "trainer.batch_size": 2, "trainer.epochs": 2,
            "optimizer.warmup_epochs": 0, "distributed.device": "cpu",
            "finetune_checkpoint.finetune": "", "clearml.enabled": False,
            "output.output_dir": str(out), "output.log_dir": str(out),
            "logging.log_model_graph": False, "evaluation.agg_windows": "mean",
            "mil.enabled": True, "mil.bag_size": 3}
    base.update(over)
    return FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab_age.json", **base), out


@pytest.mark.parametrize("pool,score_input", [("mean", "h"), ("gated", "hq"), ("sigmoid", "q")])
def test_mil_run_finetune(tmp_path, pool, score_input):
    from labram.runs import run_finetune
    cfg, out = _cfg(tmp_path, **{"mil.pool": pool, "mil.score_input": score_input,
                                 "channel_drop.enabled": True, "channel_drop.prob": 0.0,
                                 "channel_drop.consistency_weight": 0.0,
                                 "mixup.enabled": True, "mil.freeze_encoder_epochs": 1,
                                 "mil.retention_weight": 0.001, "mil.head_lr_scale": 5.0,
                                 "mil.descriptor_fit_batches": 2})
    summary = run_finetune.main(cfg)
    val = summary["best_val_stats"]
    assert "mae" in val and "case_mean_mae" in val and "window_mae" in val
    files = glob.glob(os.path.join(str(out) + "*", "analysis", "val_bag_weights_epoch*.parquet"))
    assert files
    w = pd.read_parquet(files[0])
    assert abs(w.groupby("recording")["alpha"].sum() - 1).max() < 1e-3
    model = run_finetune.get_model(cfg)
    assert hasattr(model, "mil") and type(model.mil.pool).__name__.lower().startswith(
        {"mean": "mean", "gated": "gated", "sigmoid": "sigmoid"}[pool])


def test_mil_random_bag_sizes_with_retention(tmp_path):
    """Random K per recording (padded bags), ungated ABMIL on h + q, retention after a frozen epoch."""
    from labram.runs import run_finetune
    cfg, out = _cfg(tmp_path, **{"mil.pool": "gated", "mil.gated": False, "mil.score_input": "hq",
                                 "mil.bag_size_choices": [1, 2, 3, 4], "trainer.batch_size": 3,
                                 "mixup.enabled": True, "mil.freeze_encoder_epochs": 1,
                                 "mil.retention_weight": 0.1, "mil.descriptor_fit_batches": 2})
    val = run_finetune.main(cfg)["best_val_stats"]
    assert torch.isfinite(torch.tensor(val["mae"]))


def test_bag_dataset_draws_and_collate(tmp_path):
    from labram.data.bags import RecordingBagDataset, collate_bags
    from labram.data import get_dataset_bundle
    from labram.data.window_selection import WindowSelection, apply_window_selection
    from test_window_qc import _corpus
    from labram.configs.run_configs import FinetuneRunConfig
    root = _corpus(tmp_path / "data")
    cfg = FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab_age.json",
                                        **{"data.trim_start_sec": 0.0, "data.trim_end_sec": 0.0,
                                           "data.eval_minutes": 0.0})
    bundle = apply_window_selection(get_dataset_bundle("TUAB_AGE", str(root), data_format="npy"),
                                    WindowSelection.from_data_config(cfg.data))
    bags = RecordingBagDataset(bundle.train, k=2, train=True)
    x, y, stem, minutes = bags[0]
    assert x.shape[0] == min(2, len(bags.windows[0])) and x.dim() == 3
    ev = RecordingBagDataset(bundle.val, train=False)
    xb, yb, stems, mb, mask = collate_bags([ev[i] for i in range(min(2, len(ev)))])
    assert xb.shape[:2] == mask.shape and mask.any(1).all()
    assert torch.all(mb[mask] >= 0)


def test_warm_start_from_a_finetuned_checkpoint_loads_weights(tmp_path):
    """A fine-tuned checkpoint (plain keys) loads despite the default 'student.' filter."""
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs import run_finetune
    from labram.runs.finetune_setup import load_finetune_checkpoint
    cfg = FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab_age.json",
                                        **{"model.nb_classes": 1, "model.task": "regression"})
    src = run_finetune.get_model(cfg)
    torch.nn.init.constant_(src.head.bias, 3.0)
    path = tmp_path / "ft.pth"
    torch.save({"model": src.state_dict()}, path)
    cfg.finetune_checkpoint.finetune = str(path)
    assert cfg.finetune_checkpoint.model_filter_name != ""
    dst = run_finetune.get_model(cfg)
    load_finetune_checkpoint(dst, cfg.finetune_checkpoint)
    assert torch.equal(dst.blocks[0].attn.qkv.weight, src.blocks[0].attn.qkv.weight)
    assert float(dst.head.bias) == 3.0
