"""Finished runs must end: bounded ClearML finalize, no PyTorch checkpoint
auto-uploads, and a forced exit when exit handlers hang."""
import subprocess
import sys
import textwrap
import time
import types

import pytest

from labram.utils.clearml_artifacts import finalize_clearml_task


def _task(flush_sleep=0.0):
    calls = []

    def flush(wait_for_uploads=True):
        calls.append("flush")
        time.sleep(flush_sleep)
    return types.SimpleNamespace(flush=flush, mark_completed=lambda force=False: calls.append("done"),
                                 close=lambda: calls.append("close")), calls


def test_finalize_flushes_completes_and_closes():
    task, calls = _task()
    assert finalize_clearml_task(task, timeout_sec=5)
    assert calls == ["flush", "done", "close"]


def test_finalize_gives_up_after_the_timeout():
    task, _ = _task(flush_sleep=30)
    t0 = time.time()
    assert not finalize_clearml_task(task, timeout_sec=0.5)
    assert time.time() - t0 < 5


def test_pytorch_checkpoint_auto_upload_is_off():
    from labram.runs.common import clearml_frameworks
    assert clearml_frameworks(True) == {"pytorch": False}
    assert clearml_frameworks(False) is False


def test_task_init_receives_the_framework_map(monkeypatch):
    clearml = pytest.importorskip("clearml")
    from labram.configs.train_config import ClearMLConfig
    from labram.runs import common
    seen = {}

    class _Task:
        @staticmethod
        def init(**kw):
            seen.update(kw)
            return types.SimpleNamespace(add_tags=lambda t: None, connect=lambda *a, **k: None,
                                         connect_configuration=lambda *a, **k: None)
    monkeypatch.setattr(clearml, "Task", _Task, raising=False)
    common.init_clearml_task(ClearMLConfig(enabled=True), None, global_rank=0)
    assert seen["auto_connect_frameworks"] == {"pytorch": False}


@pytest.mark.parametrize("body, expected", [("pass", 0), ("raise RuntimeError('boom')", 1)])
def test_exit_guard_ends_a_process_whose_exit_handlers_hang(body, expected):
    script = textwrap.dedent(f"""
        import atexit, time
        from labram.utils.exit_guard import run_and_exit
        atexit.register(lambda: time.sleep(60))     # e.g. ClearML waiting on an upload
        def main():
            {body}
        run_and_exit(main, grace_sec=1)
    """)
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert time.time() - t0 < 15
    assert proc.returncode == expected
    assert "forcing exit" in proc.stderr
