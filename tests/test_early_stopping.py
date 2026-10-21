"""trainer.early_stopping: stop on a validation plateau or worsening over the
last k epochs, never during warm-up or before k post-warm-up epochs exist."""
import pytest

from labram.train.early_stopping import should_stop


def _first_stop(hist, **kw):
    for e in range(len(hist)):
        if should_stop(hist[:e + 1], **kw):
            return e
    return None


def test_stops_after_k_epochs_without_improvement():
    hist = [9, 8, 7.5, 7.4, 7.45, 7.42, 7.41, 7.5]
    # best 7.4 at epoch 3; epochs 4,5,6 do not beat it -> stop at 6
    assert _first_stop(hist, patience=3) == 6


def test_never_during_warmup_nor_before_k_post_warmup_epochs():
    hist = [9, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7]       # always worse
    assert _first_stop(hist, patience=3, warmup_epochs=0) == 3
    assert _first_stop(hist, patience=3, warmup_epochs=5) == 7   # first check: epochs 5-7
    assert not should_stop(hist[:7], patience=3, warmup_epochs=5)


def test_still_improving_never_stops():
    hist = [9, 8.5, 8.2, 8.0, 7.9, 7.85, 7.84, 7.83]
    assert _first_stop(hist, patience=3) is None
    # ...unless the window's gain over the earlier best is below min_delta:
    # epoch 7 reaches 7.83, only 0.07 below the earlier best 7.90
    assert _first_stop(hist, patience=3, min_delta=0.1) == 7


def test_higher_is_better_and_non_finite():
    assert _first_stop([0.5, 0.6, 0.6, 0.59, 0.58], patience=3, lower_is_better=False) == 4
    assert should_stop([7, 6, 6, 6, 6], patience=3)               # a tie is a plateau
    assert should_stop([9, 8, float("nan"), float("nan"), float("nan")], patience=3)
    with pytest.raises(ValueError):
        should_stop([1, 2], patience=0)


def test_run_finetune_reports_early_stopping(tmp_path):
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs import run_finetune
    from test_window_qc import _corpus
    root = _corpus(tmp_path / "data")
    out = tmp_path / "out"
    cfg = FinetuneRunConfig.load_config(
        "labram/configs/defaults/finetune_tuab_age.json",
        **{"data.data_path": str(root), "data.data_format": "npy",
           "data.trim_start_sec": 0.0, "data.trim_end_sec": 0.0, "data.eval_minutes": 0.0,
           "data.num_workers": 0, "data.stage_dir": "off", "trainer.batch_size": 8,
           "trainer.debug": False, "trainer.epochs": 6,                   # (debug caps at 2)
           "optimizer.warmup_epochs": 0, "optimizer.lr": 0.0,          # no learning: a plateau
           "optimizer.min_lr": 0.0, "optimizer.warmup_lr": 0.0,
           "trainer.early_stopping": True, "trainer.early_stopping_patience": 2,
           "distributed.device": "cpu", "finetune_checkpoint.finetune": "",
           "clearml.enabled": False, "output.output_dir": str(out), "output.log_dir": str(out),
           "logging.log_model_graph": False})
    summary = run_finetune.main(cfg)
    # lr 0: every epoch ties epoch 0, so patience 2 stops after epoch 2
    assert summary["stopped_early"] and len(summary["val_history"]) == 3
