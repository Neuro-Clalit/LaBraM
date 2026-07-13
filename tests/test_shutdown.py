"""Tests for post-training machine shutdown (labram.utils.shutdown)."""

import pytest

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
