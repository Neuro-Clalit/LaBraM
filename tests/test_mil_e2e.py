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


def test_mil_cross_validation_fold_with_held_out_test(tmp_path):
    """One-step bag training (no warm start) as CV fold 0: folds over train+val by
    subject, the original test split held out as the fold's test set."""
    import json
    from labram.runs import finetune_cv
    cfg, out = _cfg(tmp_path, **{"mil.pool": "gated", "mil.gated": False, "mil.score_input": "h",
                                 "cross_validation.enabled": True, "cross_validation.n_folds": 3,
                                 "cross_validation.fold": 0, "cross_validation.pool": "train_val",
                                 "cross_validation.test_set": "held_out",
                                 "cross_validation.split_by": "subject", "trainer.epochs": 1})
    finetune_cv.run_cross_validation(cfg)
    fold = glob.glob(os.path.join(str(out) + "*", "fold_0"))
    assert fold, os.listdir(tmp_path)
    split = json.load(open(glob.glob(os.path.join(fold[0], "cv_split.json"))[0]))
    assert split["pool"] == "train_val"
    metrics = json.load(open(os.path.join(fold[0], "fold_metrics.json")))
    assert metrics
    log = open(os.path.join(fold[0], "log.txt")).read()
    assert '"test_mae"' in log or "test_mae" in log


def test_mil_phased_schedule_window_then_pool_then_joint(tmp_path):
    """mil.window_epochs: a window-level epoch first (pool untouched), then a
    frozen-encoder epoch training only the pool (freeze_head), then joint bag training."""
    from labram.runs import run_finetune
    from labram.train import train_finetune, train_mil
    cfg, out = _cfg(tmp_path, **{"mil.pool": "gated", "mil.score_input": "h", "trainer.epochs": 3,
                                 "mil.window_epochs": 1, "mil.window_batch_size": 4,
                                 "mil.window_minutes_per_epoch": 1.0, "mil.freeze_encoder_epochs": 1,
                                 "mil.freeze_head": True, "mil.retention_weight": 0.1,
                                 "mil.head_lr_scale": 5.0, "mixup.enabled": True,
                                 # the SageMaker jobs' settings (submit_grid.sh BASE)
                                 "channel_drop.enabled": True, "channel_drop.prob": 0.0,
                                 "channel_drop.consistency_weight": 0.0,
                                 "channel_drop.eval_consistency": False})
    snaps = []

    def snap(model, tag):
        m = getattr(model, "module", model)
        snaps.append((tag, {k: v.detach().clone() for k, v in m.state_dict().items()}))

    real_win, real_bag = train_finetune.train_one_epoch, train_mil.train_one_epoch_bags

    def win(model, *a, **k):
        snap(model, "window_start")
        out_ = real_win(model, *a, **k)
        return out_

    def bag(model, *a, **k):
        snap(model, f"bag_start_{a[4]}")
        return real_bag(model, *a, **k)
    train_finetune.train_one_epoch, train_mil.train_one_epoch_bags = win, bag
    try:
        summary = run_finetune.main(cfg)
    finally:
        train_finetune.train_one_epoch, train_mil.train_one_epoch_bags = real_win, real_bag
    assert summary["best_val_stats"]["mae"] > 0
    tags = [t for t, _ in snaps]
    assert tags == ["window_start", "bag_start_1", "bag_start_2"], tags
    s = dict(snaps)

    def changed(a, b, prefix):
        return any(not torch.equal(s[a][k], s[b][k]) for k in s[a]
                   if k.startswith(prefix) and s[a][k].is_floating_point())
    assert not changed("window_start", "bag_start_1", "mil.")      # window phase: pool untouched
    assert changed("window_start", "bag_start_1", "blocks.")       # ... encoder trained
    assert changed("window_start", "bag_start_1", "head.")         # ... head trained
    assert changed("bag_start_1", "bag_start_2", "mil.")           # frozen epoch: pool trained
    assert not changed("bag_start_1", "bag_start_2", "blocks.")    # ... encoder frozen
    assert not changed("bag_start_1", "bag_start_2", "head.")      # ... head frozen (freeze_head)
    assert glob.glob(os.path.join(str(out) + "*", "retention_reference.pth"))
    assert train_mil.bag_phase(cfg.mil, 0) == (True, False)
    assert train_mil.bag_phase(cfg.mil, 1) == (False, True)
    assert train_mil.bag_phase(cfg.mil, 2) == (False, False)


def test_collate_bag_windows_flattens_bags_into_window_samples():
    from labram.data.bags import collate_bag_windows
    a = (torch.randn(3, 2, 10), torch.tensor(40.0), "rec_a", torch.zeros(3))
    b = (torch.randn(2, 2, 10), torch.tensor(70.0), "rec_b", torch.zeros(2))
    x, y, ids = collate_bag_windows([a, b])
    assert x.shape == (5, 2, 10) and torch.equal(x[:3], a[0]) and torch.equal(x[3:], b[0])
    assert y.tolist() == [40.0, 40.0, 40.0, 70.0, 70.0]
    assert ids == ["rec_a"] * 3 + ["rec_b"] * 2
    # multitask targets [age, abnormal] repeat row-wise
    c = (torch.randn(2, 2, 10), torch.tensor([50.0, 1.0]), "rec_c", torch.zeros(2))
    _, yc, _ = collate_bag_windows([c])
    assert yc.shape == (2, 2) and yc[1].tolist() == [50.0, 1.0]
