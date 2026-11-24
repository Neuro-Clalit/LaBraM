"""End-to-end fine-tuning with a foreign encoder (CBraMod, REVE) on the npy fixture."""
import glob
import os
import sys

import pytest
import torch

DROP = ["A1", "A2", "T1", "T2"]                      # the fixture has 23 channels; the encoders take the 19


def _cfg(tmp_path, model, **over):
    from labram.configs.run_configs import FinetuneRunConfig
    from test_window_qc import _corpus
    root = _corpus(tmp_path / "data")
    out = tmp_path / "out"
    base = {"data.data_path": str(root), "data.data_format": "npy", "data.drop_channels": DROP,
            "data.trim_start_sec": 0.0, "data.trim_end_sec": 0.0, "data.eval_minutes": 0.0,
            "data.num_workers": 0, "trainer.batch_size": 2, "trainer.epochs": 2,
            "optimizer.warmup_epochs": 0, "distributed.device": "cpu",
            "finetune_checkpoint.finetune": "", "clearml.enabled": False,
            "output.output_dir": str(out), "output.log_dir": str(out),
            "logging.log_model_graph": False, "evaluation.agg_windows": "mean",
            "model.model": model}
    base.update(over)
    return FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab_age.json", **base), out


@pytest.mark.parametrize("model", ["cbramod_tiny_patch200", "reve_tiny_patch200"])
def test_window_level_finetune_runs(tmp_path, model):
    from labram.runs import run_finetune
    cfg, _ = _cfg(tmp_path, model, **{"channel_drop.enabled": True, "channel_drop.prob": 0.0,
                                      "channel_drop.consistency_weight": 0.0, "mixup.enabled": True,
                                      "data.train_minutes_per_epoch": 1.0})
    val = run_finetune.main(cfg)["best_val_stats"]
    assert torch.isfinite(torch.tensor(val["mae"]))


@pytest.mark.parametrize("model", ["cbramod_tiny_patch200", "reve_tiny_patch200"])
def test_p4_recipe_runs_and_the_run_is_rescored_from_its_folder(tmp_path, model):
    """P4: window phase, an attention-only epoch, joint bag epochs with a per-window loss."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import eval_age_by_cohort as ev
    from labram.runs import run_finetune
    cfg, out = _cfg(tmp_path, model, **{
        "mil.enabled": True, "mil.bag_size": 3, "mil.pool": "gated", "mil.gated": False,
        "mil.score_input": "h", "trainer.epochs": 3, "mil.window_epochs": 1,
        "mil.window_minutes_per_epoch": 0.3334, "mil.window_bag_repeats": 2, "mil.window_batch_size": 4,
        "mil.freeze_encoder_epochs": 1, "mil.freeze_head": True, "mil.head_lr_scale": 5.0,
        "mil.window_loss_weight": 0.5, "mil.bag_repeats": 2,
        "channel_drop.enabled": True, "channel_drop.prob": 0.0, "channel_drop.consistency_weight": 0.0,
        "mixup.enabled": True, "output.save_ckpt": True})
    summary = run_finetune.main(cfg)
    assert summary["best_val_stats"]["mae"] > 0
    run_dir = [d for d in glob.glob(str(out) + "*") if os.path.exists(os.path.join(d, "run_config.yaml"))][0]
    assert glob.glob(os.path.join(run_dir, "analysis", "val_bag_weights_epoch*.parquet"))
    ckpt = "checkpoint-best.pth" if os.path.exists(os.path.join(run_dir, "checkpoint-best.pth")) else "checkpoint.pth"
    res = ev.evaluate_run(run_dir, cfg.data.data_path, ckpt, ["test"], torch.device("cpu"),
                          batch_size=4, pred_stats=(50.0, 10.0), stage_dir="none", num_workers=0)
    preds = res["splits"]["test"]["predictions"]
    assert len(preds) == 1 and all(torch.isfinite(torch.tensor(v)) for v in preds.values())
