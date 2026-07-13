"""Tests for explicit ClearML model-artifact upload (Task 7):
labram.utils.clearml_artifacts."""

import sys
import types

import pytest

from labram.utils.clearml_artifacts import (
    get_clearml_task,
    resolve_final_checkpoint,
    upload_model_artifact,
)
from labram.utils.logging import ClearMLLogger, MultiWriter, TensorboardLogger


def test_get_task_from_clearml_logger():
    logger = ClearMLLogger(task="TASK", clearml_logger=object())
    assert get_clearml_task(logger) == "TASK"


def test_get_task_from_multiwriter(tmp_path):
    tb = TensorboardLogger(log_dir=str(tmp_path))
    cl = ClearMLLogger(task="TASK", clearml_logger=object())
    assert get_clearml_task(MultiWriter([tb, cl])) == "TASK"


def test_get_task_none_when_absent(tmp_path):
    assert get_clearml_task(None) is None
    assert get_clearml_task(TensorboardLogger(log_dir=str(tmp_path))) is None


def test_resolve_prefers_best_then_rolling_then_epoch(tmp_path):
    assert resolve_final_checkpoint(str(tmp_path)) is None
    (tmp_path / "checkpoint-3.pth").write_text("")
    (tmp_path / "checkpoint-10.pth").write_text("")
    assert resolve_final_checkpoint(str(tmp_path)).endswith("checkpoint-10.pth")
    (tmp_path / "checkpoint.pth").write_text("")
    assert resolve_final_checkpoint(str(tmp_path)).endswith("checkpoint.pth")
    (tmp_path / "checkpoint-best.pth").write_text("")
    assert resolve_final_checkpoint(str(tmp_path)).endswith("checkpoint-best.pth")


def test_upload_model_artifact_registers_output_model(tmp_path, monkeypatch):
    ckpt = tmp_path / "checkpoint-best.pth"
    ckpt.write_text("weights")

    captured = {}

    class FakeOutputModel:
        def __init__(self, task=None, name=None, framework=None):
            captured["task"] = task
            captured["name"] = name
            captured["framework"] = framework

        def update_weights(self, weights_filename=None, auto_delete_file=True):
            captured["weights"] = weights_filename
            return "s3://bucket/models/checkpoint-best.pth"

    fake_clearml = types.ModuleType("clearml")
    fake_clearml.OutputModel = FakeOutputModel
    monkeypatch.setitem(sys.modules, "clearml", fake_clearml)

    uri = upload_model_artifact("TASK", str(ckpt), name="best")
    assert uri == "s3://bucket/models/checkpoint-best.pth"
    assert captured["task"] == "TASK"
    assert captured["name"] == "best"
    assert captured["weights"] == str(ckpt)


def test_upload_skips_when_no_task_or_missing_file(tmp_path):
    assert upload_model_artifact(None, str(tmp_path / "x.pth")) is None
    assert upload_model_artifact("TASK", str(tmp_path / "missing.pth")) is None
