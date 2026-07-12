"""Tests for the logging utilities: Python-logging setup, ClearMLLogger, and
MultiWriter fan-out. All are exercised without the optional ``clearml`` package
installed, so they double as a check that the integration degrades gracefully.
"""
import importlib.util
import logging
import types

import pytest
import torch

from labram.utils.logging import (
    ClearMLLogger,
    MultiWriter,
    TensorboardLogger,
    configure_logging,
    get_logger,
)

# ClearML is an optional dependency; the debug-routing tests below are skipped
# when it (its config) is not available, mirroring how the runtime degrades to
# TensorBoard-only logging.
clearml_available = importlib.util.find_spec("clearml") is not None


class TestGetLogger:
    def test_returns_labram_root(self):
        assert get_logger().name == "labram"
        assert get_logger(None).name == "labram"
        assert get_logger("labram").name == "labram"

    def test_dotted_child_is_under_labram(self):
        assert get_logger("labram.runs.common").name == "labram.runs.common"

    def test_bare_name_becomes_child(self):
        assert get_logger("runs").name == "labram.runs"


class TestConfigureLogging:
    def _clear(self):
        logger = logging.getLogger("labram")
        for h in list(logger.handlers):
            logger.removeHandler(h)

    def teardown_method(self):
        self._clear()

    def test_attaches_stream_handler_and_level(self):
        self._clear()
        logger = configure_logging(level=logging.INFO, rank=0)
        assert logger.handlers, "expected a handler to be attached"
        assert logger.level == logging.INFO
        assert logger.propagate is False

    def test_non_zero_rank_is_raised_to_warning(self):
        self._clear()
        logger = configure_logging(level=logging.INFO, rank=3)
        assert logger.level == logging.WARNING

    def test_idempotent_does_not_stack_handlers(self):
        self._clear()
        configure_logging(rank=0)
        n_first = len(logging.getLogger("labram").handlers)
        configure_logging(rank=0)
        n_second = len(logging.getLogger("labram").handlers)
        assert n_first == n_second

    def test_writes_file_on_rank_zero(self, tmp_path):
        self._clear()
        log_file = tmp_path / "sub" / "run.log"
        logger = configure_logging(rank=0, log_file=str(log_file))
        logger.info("hello-clearml")
        for h in logger.handlers:
            h.flush()
        assert log_file.exists()
        assert "hello-clearml" in log_file.read_text(encoding="utf-8")

    def test_no_file_on_non_zero_rank(self, tmp_path):
        self._clear()
        log_file = tmp_path / "run.log"
        configure_logging(rank=1, log_file=str(log_file))
        assert not log_file.exists()


class _FakeClearMLLogger:
    """Records report_scalar / report_image / flush calls."""

    def __init__(self):
        self.scalars = []
        self.images = []
        self.flushed = 0

    def report_scalar(self, title, series, value, iteration):
        self.scalars.append((title, series, value, iteration))

    def report_image(self, title, series, iteration, image):
        self.images.append((title, series, iteration, image))

    def flush(self):
        self.flushed += 1


class TestClearMLLogger:
    def test_noop_when_logger_unavailable(self):
        writer = ClearMLLogger(task=None)
        # Every method is a safe no-op — no exception, nothing to assert beyond that.
        writer.set_step(5)
        writer.update(loss=1.0, head="loss")
        writer.update_image(sample=torch.zeros(3, 4, 4), head="img")
        writer.flush()

    def test_update_forwards_scalars_with_step_as_iteration(self):
        fake = _FakeClearMLLogger()
        writer = ClearMLLogger(clearml_logger=fake)
        writer.set_step(7)
        writer.update(loss=0.5, acc=0.9, head="train")
        assert ("train", "loss", 0.5, 7) in fake.scalars
        assert ("train", "acc", 0.9, 7) in fake.scalars

    def test_explicit_step_overrides_internal_step(self):
        fake = _FakeClearMLLogger()
        writer = ClearMLLogger(clearml_logger=fake)
        writer.set_step(1)
        writer.update(loss=0.1, head="train", step=42)
        assert fake.scalars == [("train", "loss", 0.1, 42)]

    def test_tensor_values_are_unwrapped(self):
        fake = _FakeClearMLLogger()
        writer = ClearMLLogger(clearml_logger=fake)
        writer.update(loss=torch.tensor(2.0), head="train")
        assert fake.scalars[0][2] == 2.0

    def test_none_values_skipped(self):
        fake = _FakeClearMLLogger()
        writer = ClearMLLogger(clearml_logger=fake)
        writer.update(loss=None, acc=1.0, head="train")
        assert [s[1] for s in fake.scalars] == ["acc"]

    def test_flush_forwards(self):
        fake = _FakeClearMLLogger()
        writer = ClearMLLogger(clearml_logger=fake)
        writer.flush()
        assert fake.flushed == 1

    def test_builds_logger_from_task(self):
        fake = _FakeClearMLLogger()

        class _Task:
            def get_logger(self_inner):
                return fake

        writer = ClearMLLogger(task=_Task())
        writer.update(loss=0.3, head="train")
        assert fake.scalars[0] == ("train", "loss", 0.3, 0)


class _RecordingWriter:
    def __init__(self):
        self.calls = []

    def set_step(self, step=None):
        self.calls.append(("set_step", step))

    def update(self, head="scalar", step=None, **kwargs):
        self.calls.append(("update", head, step, kwargs))

    def update_image(self, head="images", step=None, **kwargs):
        self.calls.append(("update_image", head, step, kwargs))

    def flush(self):
        self.calls.append(("flush",))


class TestMultiWriter:
    def test_drops_none_writers(self):
        w = _RecordingWriter()
        mw = MultiWriter([w, None, None])
        assert mw.writers == [w]

    def test_fans_out_update_to_all(self):
        a, b = _RecordingWriter(), _RecordingWriter()
        mw = MultiWriter([a, b])
        mw.update(loss=1.0, head="loss", step=3)
        assert a.calls == [("update", "loss", 3, {"loss": 1.0})]
        assert b.calls == [("update", "loss", 3, {"loss": 1.0})]

    def test_fans_out_set_step_and_flush(self):
        a, b = _RecordingWriter(), _RecordingWriter()
        mw = MultiWriter([a, b])
        mw.set_step(9)
        mw.flush()
        assert ("set_step", 9) in a.calls and ("set_step", 9) in b.calls
        assert ("flush",) in a.calls and ("flush",) in b.calls

    def test_update_image_only_on_supporting_writers(self):
        class _NoImage:
            def __init__(self):
                self.updated = False

            def set_step(self, step=None):
                pass

            def update(self, head="scalar", step=None, **kwargs):
                pass

        supporting = _RecordingWriter()
        no_image = _NoImage()
        mw = MultiWriter([supporting, no_image])
        # Must not raise even though _NoImage has no update_image.
        mw.update_image(sample=torch.zeros(3, 2, 2), head="img")
        assert supporting.calls[0][0] == "update_image"

    def test_combines_tensorboard_and_clearml(self, tmp_path):
        tb = TensorboardLogger(log_dir=str(tmp_path / "tb"))
        fake = _FakeClearMLLogger()
        cm = ClearMLLogger(clearml_logger=fake)
        mw = MultiWriter([tb, cm])
        mw.set_step(0)
        mw.update(loss=0.25, head="loss")
        mw.flush()
        # ClearML received the scalar; TensorBoard wrote its event file.
        assert fake.scalars[0] == ("loss", "loss", 0.25, 0)


class _FakeTask:
    """Stand-in for ``clearml.Task`` that records init kwargs and tags instead
    of contacting a ClearML server, so the debug-routing logic can be tested
    without credentials."""

    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tags = []

    @classmethod
    def init(cls, **kwargs):
        inst = cls(**kwargs)
        _FakeTask.last = inst
        return inst

    @staticmethod
    def set_offline(offline_mode=False):
        pass

    def add_tags(self, tags):
        self.tags.extend(tags)

    def connect_configuration(self, *args, **kwargs):  # pragma: no cover - unused here
        pass


@pytest.mark.skipif(
    not clearml_available,
    reason="clearml is not installed/configured; ClearML tests skipped",
)
class TestInitClearMLTaskDebug:
    """``init_clearml_task`` routes debug runs to a '<project>/debug' subfolder
    and tags them 'debug'. The ClearML server is mocked via ``_FakeTask``."""

    def _cfg(self, **overrides):
        from labram.configs.train_config import ClearMLConfig
        params = dict(enabled=True, project_name="eeg", output_uri="s3://clearml-eeg")
        params.update(overrides)
        return ClearMLConfig(**params)

    def test_debug_run_is_tagged_and_routed_to_debug_subfolder(self, monkeypatch):
        import clearml
        from labram.runs import common
        monkeypatch.setattr(clearml, "Task", _FakeTask, raising=False)

        task = common.init_clearml_task(
            self._cfg(), types.SimpleNamespace(debug=True), global_rank=0)

        assert task is _FakeTask.last
        assert task.kwargs["project_name"] == "eeg"
        assert task.kwargs["output_uri"] == "s3://clearml-eeg/eeg/debug"
        assert "debug" in task.tags

    def test_non_debug_run_keeps_base_uri_and_no_debug_tag(self, monkeypatch):
        import clearml
        from labram.runs import common
        monkeypatch.setattr(clearml, "Task", _FakeTask, raising=False)

        task = common.init_clearml_task(
            self._cfg(), types.SimpleNamespace(debug=False), global_rank=0)

        assert task.kwargs["output_uri"] == "s3://clearml-eeg"
        assert "debug" not in task.tags

    def test_disabled_returns_none(self):
        from labram.runs import common
        task = common.init_clearml_task(
            self._cfg(enabled=False), types.SimpleNamespace(debug=True), global_rank=0)
        assert task is None

    def test_off_rank_returns_none(self):
        from labram.runs import common
        task = common.init_clearml_task(
            self._cfg(), types.SimpleNamespace(debug=True), global_rank=1)
        assert task is None
        assert any((tmp_path / "tb").iterdir())
