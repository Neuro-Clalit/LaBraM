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
    LoggingConfig,
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


def test_logging_and_output_defaults():
    lg = LoggingConfig()
    assert lg.log_model_graph is True
    assert lg.model_graph_format == "svg"
    assert lg.log_data_split is True
    from labram.configs.train_config import OutputConfig
    assert OutputConfig().save_only_final_model is False


def test_run_configs_compose_new_subconfigs():
    # shutdown + logging on every run config; evaluation only on finetune.
    for cls in (PretrainRunConfig, VQNSPRunConfig, FinetuneRunConfig):
        cfg = cls()
        assert isinstance(cfg.shutdown, ShutdownConfig)
        assert isinstance(cfg.logging, LoggingConfig)
    assert isinstance(FinetuneRunConfig().evaluation, EvaluationConfig)


def test_flatten_config_produces_scalar_dotted_keys():
    from labram.runs.common import flatten_config
    flat = flatten_config(FinetuneRunConfig().as_dict())
    assert "optimizer/lr" in flat and "trainer/epochs" in flat
    # every value is a scalar/str (lists rendered as strings) -> tabular-friendly
    assert all(not isinstance(v, (dict, list, tuple)) for v in flat.values())


def test_paper_tuab_config_matches_recipe():
    cfg = FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab_paper.json")
    assert cfg.model.model == "labram_base_patch200_200"
    assert cfg.data.dataset == "TUAB"
    assert cfg.optimizer.lr == 5e-4
    assert cfg.optimizer.layer_decay == 0.65
    assert cfg.optimizer.warmup_epochs == 5
    assert cfg.trainer.epochs == 50
    assert cfg.model.abs_pos_emb is True
    assert cfg.model.rel_pos_bias is False
    assert cfg.model.qkv_bias is False
    assert cfg.finetune_checkpoint.finetune.endswith("labram-base.pth")


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
    ("labram/configs/defaults/finetune_tuab_paper.json", FinetuneRunConfig),
    ("labram/configs/defaults/pretrain.json", PretrainRunConfig),
    ("labram/configs/defaults/vqnsp.json", VQNSPRunConfig),
])
def test_default_jsons_load_with_new_fields(path, cls):
    cfg = cls.load_config(path)
    assert cfg.optimizer.sched in ("cosine", "step", "multistep", "linear", "constant")
    assert isinstance(cfg.shutdown, ShutdownConfig)
    assert isinstance(cfg.logging, LoggingConfig)
    assert cfg.output.save_only_final_model in (True, False)
    assert cfg.clearml.upload_model_artifact in (True, False)
