"""Tests for the config fields added for grad-clip/scheduler, detailed
evaluation, window aggregation, EC2 shutdown, and ClearML artifact upload."""

import pytest

from labram.configs.optim_config import OptimizerConfig
from labram.configs.run_configs import (
    FinetuneRunConfig,
    PretrainRunConfig,
    VQNSPRunConfig,
)
from labram.configs.train_config import (
    ClearMLConfig,
    EvaluationConfig,
    ShutdownConfig,
)


def test_optimizer_scheduler_and_clip_defaults():
    o = OptimizerConfig()
    assert o.sched == "cosine"
    assert o.decay_epochs == 30
    assert o.decay_rate == 0.1
    assert o.decay_milestones is None
    assert o.clip_grad is None  # grad clip is an optimizer param, off by default


def test_shutdown_defaults():
    s = ShutdownConfig()
    assert s.stop_instance_on_finish is False
    assert s.stop_delay_minutes == 5
    assert s.stop_method == "ec2"


def test_evaluation_defaults():
    e = EvaluationConfig()
    assert e.detailed_metrics is True
    assert e.log_confusion_matrix is True
    assert e.log_curves is True
    assert e.log_grad_components is False
    assert e.log_grad_freq == 50
    assert e.agg_windows == "none"
    assert e.agg_case_by == "recording"


def test_clearml_artifact_defaults():
    c = ClearMLConfig()
    assert c.upload_model_artifact is True
    assert c.artifact_name == ""


def test_run_configs_compose_new_subconfigs():
    # shutdown on every run config; evaluation only on finetune.
    for cls in (PretrainRunConfig, VQNSPRunConfig, FinetuneRunConfig):
        cfg = cls()
        assert isinstance(cfg.shutdown, ShutdownConfig)
    assert isinstance(FinetuneRunConfig().evaluation, EvaluationConfig)


def test_finetune_json_round_trip_preserves_new_fields(tmp_path):
    cfg = FinetuneRunConfig()
    cfg.optimizer.sched = "step"
    cfg.optimizer.clip_grad = 1.5
    cfg.shutdown.stop_instance_on_finish = True
    cfg.evaluation.agg_windows = "mean"
    cfg.clearml.upload_model_artifact = False
    path = str(tmp_path / "cfg.json")
    cfg.save_to(path)

    loaded = FinetuneRunConfig.load_from(path)
    assert loaded.optimizer.sched == "step"
    assert loaded.optimizer.clip_grad == 1.5
    assert loaded.shutdown.stop_instance_on_finish is True
    assert loaded.evaluation.agg_windows == "mean"
    assert loaded.clearml.upload_model_artifact is False


@pytest.mark.parametrize("path,cls", [
    ("labram/configs/defaults/finetune_tuab.json", FinetuneRunConfig),
    ("labram/configs/defaults/finetune_tuev.json", FinetuneRunConfig),
    ("labram/configs/defaults/finetune_tuab_codebook.json", FinetuneRunConfig),
    ("labram/configs/defaults/pretrain.json", PretrainRunConfig),
    ("labram/configs/defaults/vqnsp.json", VQNSPRunConfig),
])
def test_default_jsons_load_with_new_fields(path, cls):
    cfg = cls.load_config(path)
    assert cfg.optimizer.sched in ("cosine", "step", "multistep", "linear", "constant")
    assert isinstance(cfg.shutdown, ShutdownConfig)
    assert cfg.clearml.upload_model_artifact in (True, False)
