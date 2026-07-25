"""Tests for explicit ClearML model-artifact upload (Task 7):
labram.utils.clearml_artifacts."""

import sys
import types


from labram.utils.clearml_artifacts import (
    finalize_clearml_task,
    get_clearml_task,
    resolve_final_checkpoint,
    upload_model_artifact,
)
from labram.utils.logging import ClearMLLogger, MultiWriter, TensorboardLogger


class _FakeTask:
    def __init__(self):
        self.calls = []

    def flush(self, wait_for_uploads=False):
        self.calls.append(("flush", wait_for_uploads))

    def mark_completed(self, force=False):
        self.calls.append(("mark_completed", force))

    def close(self):
        self.calls.append(("close", None))


def test_finalize_clearml_task_flushes_marks_and_closes():
    task = _FakeTask()
    assert finalize_clearml_task(task) is True
    names = [c[0] for c in task.calls]
    # Flush waits for uploads, records Completed, then closes — in that order.
    assert names == ["flush", "mark_completed", "close"]
    assert ("flush", True) in task.calls


def test_finalize_clearml_task_none_is_noop():
    assert finalize_clearml_task(None) is False


def test_finalize_clearml_task_survives_errors():
    class _Boom:
        def flush(self, wait_for_uploads=False):
            raise RuntimeError("no server")

        def close(self):
            raise RuntimeError("no server")

    # Best-effort: never raises, returns False on a failed close.
    assert finalize_clearml_task(_Boom()) is False


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
