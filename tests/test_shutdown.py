"""Tests for post-training machine shutdown (labram.utils.shutdown) and the
ClearML task finalization that must precede an EC2 stop."""


from labram.configs.run_configs import FinetuneRunConfig
from labram.configs.train_config import ShutdownConfig
from labram.utils import shutdown


def test_disabled_by_default_is_noop():
    assert shutdown.maybe_stop_instance(ShutdownConfig()) is False
    assert shutdown.maybe_stop_instance(None) is False


def test_env_kill_switch(monkeypatch):
    monkeypatch.setenv("LABRAM_DISABLE_SHUTDOWN", "1")
    cfg = ShutdownConfig(stop_instance_on_finish=True, stop_method="os")
    assert shutdown.maybe_stop_instance(cfg) is False


def test_os_method_invokes_shutdown_command(monkeypatch):
    calls = []
    monkeypatch.setattr(shutdown.subprocess, "Popen", lambda args: calls.append(args))
    cfg = ShutdownConfig(stop_instance_on_finish=True, stop_method="os", stop_delay_minutes=7)
    assert shutdown.maybe_stop_instance(cfg) is True
    assert calls == [["shutdown", "-h", "+7"]]


def test_ec2_method_without_imds_is_safe(monkeypatch):
    monkeypatch.setattr(shutdown, "get_instance_identity", lambda: None)
    cfg = ShutdownConfig(stop_instance_on_finish=True, stop_method="ec2")
    # Attempted (returns True) but no metadata -> no boto3 call, no raise.
    assert shutdown.maybe_stop_instance(cfg) is True


def test_ec2_method_calls_boto3(monkeypatch):
    monkeypatch.setattr(shutdown, "get_instance_identity",
                        lambda: {"instance_id": "i-123", "region": "eu-west-1"})
    monkeypatch.setattr(shutdown.time, "sleep", lambda s: None)

    stopped = {}

    class FakeClient:
        def stop_instances(self, InstanceIds):
            stopped["ids"] = InstanceIds

    class FakeBoto3:
        def client(self, service, region_name=None):
            stopped["service"] = service
            stopped["region"] = region_name
            return FakeClient()

    import sys
    monkeypatch.setitem(sys.modules, "boto3", FakeBoto3())
    cfg = ShutdownConfig(stop_instance_on_finish=True, stop_method="ec2", stop_delay_minutes=0)
    assert shutdown.maybe_stop_instance(cfg) is True
    assert stopped["ids"] == ["i-123"]
    assert stopped["service"] == "ec2"
    assert stopped["region"] == "eu-west-1"


def test_unknown_method_returns_false():
    cfg = ShutdownConfig(stop_instance_on_finish=True, stop_method="bogus")
    assert shutdown.maybe_stop_instance(cfg) is False


def test_finalize_run_closes_clearml_before_ec2_stop(monkeypatch):
    """When shutdown is enabled and ClearML is on, finalize_run must flush+close
    the ClearML task *before* stopping the instance, so the run is persisted."""
    import labram.runs.common as common
    from labram.utils.logging import ClearMLLogger

    order = []

    class _Task:
        def flush(self, wait_for_uploads=False):
            order.append("flush")

        def mark_completed(self, force=False):
            order.append("mark_completed")

        def close(self):
            order.append("close")

    def _fake_stop(cfg):
        order.append("stop_instance")
        return True

    # finalize_run imports maybe_stop_instance from the source module inside the
    # function, so patch it there.
    monkeypatch.setattr(shutdown, "maybe_stop_instance", _fake_stop)
    # Avoid touching a real model dir / OutputModel.
    monkeypatch.setattr(common.utils, "is_main_process", lambda: True)

    config = FinetuneRunConfig()
    config.clearml.enabled = True
    config.clearml.upload_model_artifact = False
    config.shutdown.stop_instance_on_finish = True

    log_writer = ClearMLLogger(task=_Task(), clearml_logger=object())
    common.finalize_run(config, log_writer)

    # The task is finalized (flush -> close) strictly before the machine stops.
    assert "close" in order and "stop_instance" in order
    assert order.index("close") < order.index("stop_instance")
    assert order[0] == "flush"


def test_finalize_run_no_clearml_finalize_when_not_stopping(monkeypatch):
    """Without a shutdown, finalize_run must NOT force-close the ClearML task
    (it closes naturally at process exit)."""
    import labram.runs.common as common
    from labram.utils.logging import ClearMLLogger

    closed = {"v": False}

    class _Task:
        def flush(self, wait_for_uploads=False):
            pass

        def close(self):
            closed["v"] = True

    monkeypatch.setattr(common.utils, "is_main_process", lambda: True)
    config = FinetuneRunConfig()
    config.clearml.enabled = True
    config.clearml.upload_model_artifact = False
    config.shutdown.stop_instance_on_finish = False

    common.finalize_run(config, ClearMLLogger(task=_Task(), clearml_logger=object()))
    assert closed["v"] is False
