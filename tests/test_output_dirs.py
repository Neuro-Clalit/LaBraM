"""Fresh output directory per run: no auto-resume by default, a run timestamp on
output_dir/log_dir, and resuming that continues after the saved epoch."""
import os
import types

import pytest
import torch

from labram.configs.run_configs import FinetuneRunConfig
from labram.configs.train_config import OutputConfig, TrainerConfig
from labram.runs import common
from labram.runs.common import RUN_STAMP_ENV, prepare_output_dir, run_stamp, stamp_output_dirs

STAMP = "20261004_120000_123"


class TestDefaults:
    def test_new_runs_neither_resume_nor_share_a_directory(self):
        out = OutputConfig()
        assert out.auto_resume is False and out.append_timestamp is True

    @pytest.mark.parametrize("name", sorted(os.listdir("labram/configs/defaults")))
    def test_every_shipped_config_starts_fresh(self, name):
        import json
        with open(os.path.join("labram/configs/defaults", name)) as fh:
            output = json.load(fh)["output"]
        assert output["auto_resume"] is False and output["append_timestamp"] is True


class TestStampOutputDirs:
    def test_separate_log_dir_is_stamped_too(self):
        out = OutputConfig(output_dir="./checkpoints/age/", log_dir="./log/age")
        assert stamp_output_dirs(out, STAMP)
        assert out.output_dir == f"./checkpoints/age_{STAMP}"
        assert out.log_dir == f"./log/age_{STAMP}"

    @pytest.mark.parametrize("log_dir, expected", [
        ("ckpt/run", f"ckpt/run_{STAMP}"),
        ("ckpt/run/tensorboard", f"ckpt/run_{STAMP}/tensorboard"),
        ("", ""),
    ])
    def test_log_dir_inside_output_dir_follows_it(self, log_dir, expected):
        out = OutputConfig(output_dir="ckpt/run", log_dir=log_dir)
        stamp_output_dirs(out, STAMP)
        assert out.log_dir == expected

    def test_applied_once_then_cleared(self):
        out = OutputConfig(output_dir="ckpt/run")
        assert stamp_output_dirs(out, STAMP)
        assert out.append_timestamp is False
        assert not stamp_output_dirs(out, "20991231_000000_000")
        assert out.output_dir == f"ckpt/run_{STAMP}"

    @pytest.mark.parametrize("kwargs", [dict(auto_resume=True), dict(resume="ckpt/run/checkpoint.pth")])
    def test_resuming_keeps_the_existing_directory(self, kwargs):
        out = OutputConfig(output_dir="ckpt/run", **kwargs)
        assert not stamp_output_dirs(out, STAMP)
        assert out.output_dir == "ckpt/run"

    def test_disabled_or_no_output_dir_is_a_no_op(self):
        out = OutputConfig(output_dir="ckpt/run", append_timestamp=False)
        assert not stamp_output_dirs(out, STAMP) and out.output_dir == "ckpt/run"
        assert not stamp_output_dirs(OutputConfig(), STAMP)


def test_run_stamp_can_be_pinned(monkeypatch):
    monkeypatch.setenv(RUN_STAMP_ENV, STAMP)
    assert run_stamp() == STAMP
    monkeypatch.delenv(RUN_STAMP_ENV)
    assert run_stamp() != STAMP


def test_consecutive_runs_get_different_directories(tmp_path, monkeypatch):
    dirs = []
    for stamp in ("20261004_120000_001", "20261004_120000_002"):
        monkeypatch.setenv(RUN_STAMP_ENV, stamp)
        config = FinetuneRunConfig()
        config.output.output_dir = str(tmp_path / "age")
        prepare_output_dir(config)
        dirs.append(config.output.output_dir)
    assert dirs[0] != dirs[1] and all(os.path.isdir(d) for d in dirs)


def test_saved_run_config_reloads_to_the_same_directory(tmp_path, monkeypatch):
    monkeypatch.setenv(RUN_STAMP_ENV, STAMP)
    config = FinetuneRunConfig()
    config.output.output_dir = str(tmp_path / "age")
    prepare_output_dir(config)
    saved = os.path.join(config.output.output_dir, "run_config.yaml")
    reloaded = FinetuneRunConfig.load_config(saved)
    # Resuming from the saved config must find this directory, not a new one.
    reloaded.output.auto_resume = True
    prepare_output_dir(reloaded)
    assert reloaded.output.output_dir == config.output.output_dir
    assert reloaded.output.append_timestamp is False


class TestClearMLTaskName:
    @pytest.fixture
    def fake_clearml(self, monkeypatch):
        clearml = pytest.importorskip("clearml")
        created = {}

        class _Task:
            @staticmethod
            def init(**kwargs):
                created.update(kwargs)
                return types.SimpleNamespace(add_tags=lambda tags: None,
                                             connect=lambda *a, **k: None,
                                             connect_configuration=lambda *a, **k: None)

        monkeypatch.setattr(clearml, "Task", _Task, raising=False)
        return created

    def _run_config(self, output_dir):
        return types.SimpleNamespace(debug=False, output=OutputConfig(output_dir=output_dir),
                                     model=None)

    def test_task_and_directory_share_the_stamp(self, fake_clearml):
        from labram.configs.train_config import ClearMLConfig
        common.init_clearml_task(ClearMLConfig(enabled=True),
                                 self._run_config(f"ckpt/age_{STAMP}"), global_rank=0)
        assert fake_clearml["task_name"] == f"age_{STAMP}"

    def test_explicit_task_name_gets_the_directory_stamp(self, fake_clearml):
        from labram.configs.train_config import ClearMLConfig
        common.init_clearml_task(ClearMLConfig(enabled=True, task_name="debug_visuals"),
                                 self._run_config(f"ckpt/age_{STAMP}"), global_rank=0)
        assert fake_clearml["task_name"] == f"debug_visuals_{STAMP}"


class TestCrossValidation:
    def _config(self, fold=-1):
        config = FinetuneRunConfig()
        config.output.output_dir = "ckpt/finetune_tuab"
        config.cross_validation.enabled = True
        config.cross_validation.fold = fold
        return config

    def test_in_process_study_stamps_the_base_not_the_folds(self, monkeypatch):
        from labram.runs.finetune_cv import cv_base_dir, derive_fold_config, stamp_cv_base_dir
        monkeypatch.setenv(RUN_STAMP_ENV, STAMP)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        config = self._config()
        stamp_cv_base_dir(config)
        assert cv_base_dir(config) == f"ckpt/finetune_tuab_cv5_{STAMP}"
        fold = derive_fold_config(config, 2)
        assert fold.output.output_dir == f"ckpt/finetune_tuab_cv5_{STAMP}/fold_2"
        assert fold.output.append_timestamp is False

    def test_a_single_fold_job_keeps_the_shared_base(self, monkeypatch):
        from labram.runs.finetune_cv import cv_base_dir, stamp_cv_base_dir
        monkeypatch.delenv(RUN_STAMP_ENV, raising=False)
        config = self._config(fold=1)
        stamp_cv_base_dir(config)
        assert cv_base_dir(config) == "ckpt/finetune_tuab_cv5"

    def test_a_pinned_stamp_lets_fold_jobs_share_a_stamped_base(self, monkeypatch):
        from labram.runs.finetune_cv import cv_base_dir, stamp_cv_base_dir
        monkeypatch.setenv(RUN_STAMP_ENV, STAMP)
        config = self._config(fold=1)
        stamp_cv_base_dir(config)
        assert cv_base_dir(config) == f"ckpt/finetune_tuab_cv5_{STAMP}"


class TestResumeEpoch:
    def _resume(self, tmp_path, epoch):
        from labram.utils.checkpoint import auto_load_model
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        path = tmp_path / "checkpoint.pth"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch}, path)
        trainer = TrainerConfig(start_epoch=0)
        auto_load_model(OutputConfig(output_dir=str(tmp_path), resume=str(path)), trainer,
                        model, model, optimizer, loss_scaler=None)
        return trainer.start_epoch

    def test_continues_after_the_saved_epoch(self, tmp_path):
        assert self._resume(tmp_path, 7) == 8

    def test_named_checkpoint_keeps_the_configured_start(self, tmp_path):
        assert self._resume(tmp_path, "best") == 0
