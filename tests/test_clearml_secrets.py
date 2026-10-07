"""ClearML credentials never leave the process in a run config: the submitter
passes a Secrets Manager secret name, the container reads it, and every config
copy (run_config.yaml, S3 upload, ClearML, console log) is redacted."""
import json
import logging
import types

import pytest

from labram.configs.run_configs import FinetuneRunConfig
from labram.runs import submit_sagemaker as sub
from labram.utils import secrets
from labram.utils.secrets import CLEARML_SECRET_ENV, REDACTED, redact, redacted_copy

PLANTED = "sk-PLANTED-0123456789"


def _config_with_secret():
    c = FinetuneRunConfig()
    c.clearml.enabled = True
    c.sagemaker.environment.update({"CLEARML_API_SECRET_KEY": PLANTED,
                                    "CLEARML_API_HOST": "https://api.clear.ml",
                                    "MY_TOKEN": PLANTED, "PLAIN": "keep"})
    return c


class TestRedaction:
    def test_secret_like_keys_are_masked_recursively(self):
        out = redact({"A_SECRET": "x", "nested": {"api_token": "y", "ok": 1},
                      "list": [{"PASSWORD": "z"}], "EMPTY_SECRET": ""})
        assert out == {"A_SECRET": REDACTED, "nested": {"api_token": REDACTED, "ok": 1},
                       "list": [{"PASSWORD": REDACTED}], "EMPTY_SECRET": ""}

    def test_redacted_copy_leaves_the_live_config_intact(self):
        c = _config_with_secret()
        safe = redacted_copy(c)
        assert safe.sagemaker.environment["CLEARML_API_SECRET_KEY"] == REDACTED
        assert safe.sagemaker.environment["PLAIN"] == "keep"
        assert safe.sagemaker.environment["CLEARML_API_HOST"] == "https://api.clear.ml"
        assert c.sagemaker.environment["CLEARML_API_SECRET_KEY"] == PLANTED

    def test_console_form_has_no_secret(self):
        assert PLANTED not in str(redacted_copy(_config_with_secret()))

    def test_saved_run_config_has_no_secret(self, tmp_path):
        from labram.runs.common import prepare_output_dir
        c = _config_with_secret()
        c.output.output_dir = str(tmp_path / "run")
        c.output.append_timestamp = False
        prepare_output_dir(c)
        text = (tmp_path / "run" / "run_config.yaml").read_text()
        assert PLANTED not in text and REDACTED in text

    def test_uploaded_config_has_no_secret(self, tmp_path):
        uploaded = {}

        def fake_upload(session, local, key_prefix, extra_args=None):
            uploaded["text"] = open(local).read()
            return "s3://bucket/run_config.yaml"
        c = _config_with_secret()
        launcher = types.SimpleNamespace(_get_session=lambda: object())
        orig = sub._upload_data
        sub._upload_data = fake_upload
        try:
            sub.upload_run_config(launcher, c)
        finally:
            sub._upload_data = orig
        assert PLANTED not in uploaded["text"]

    def test_clearml_receives_no_secret(self, monkeypatch):
        clearml = pytest.importorskip("clearml")
        from labram.configs.train_config import ClearMLConfig
        from labram.runs import common
        seen = []

        class _Task:
            @staticmethod
            def init(**kw):
                return types.SimpleNamespace(
                    add_tags=lambda t: None,
                    connect=lambda d, name=None: seen.append(json.dumps(d)),
                    connect_configuration=lambda d, name=None: seen.append(json.dumps(d)))
        monkeypatch.setattr(clearml, "Task", _Task, raising=False)
        common.init_clearml_task(ClearMLConfig(enabled=True), _config_with_secret(), global_rank=0)
        assert seen and not any(PLANTED in s for s in seen)


class TestSubmitter:
    def test_secret_name_replaces_credentials_in_the_job_env(self):
        c = _config_with_secret()
        forwarded = sub.forward_clearml_env(c)
        env = c.sagemaker.environment
        assert forwarded == {CLEARML_SECRET_ENV: "labram/clearml"}
        assert env[CLEARML_SECRET_ENV] == "labram/clearml"
        assert "CLEARML_API_SECRET_KEY" not in env
        # Other variables the user set explicitly still reach the job (and are
        # redacted from every config copy, see TestRedaction).
        assert env["PLAIN"] == "keep"

    def test_missing_secret_stops_the_submission(self, monkeypatch):
        monkeypatch.setattr(secrets, "secret_exists", lambda name, session=None: False)
        with pytest.raises(SystemExit, match="clearml_secret put"):
            sub.forward_clearml_env(_config_with_secret(), boto_session=object())

    def test_disabled_clearml_forwards_nothing(self):
        c = FinetuneRunConfig()
        assert sub.forward_clearml_env(c) == {} and CLEARML_SECRET_ENV not in c.sagemaker.environment


class _FakeSecrets:
    class exceptions:
        ResourceNotFoundException = KeyError

    def __init__(self, store):
        self.store = store

    def get_secret_value(self, SecretId):
        return {"SecretString": self.store[SecretId]}

    def describe_secret(self, SecretId):
        if SecretId not in self.store:
            raise KeyError(SecretId)

    def put_secret_value(self, SecretId, SecretString):
        if SecretId not in self.store:
            raise KeyError(SecretId)
        self.store[SecretId] = SecretString
        return {"ARN": f"arn:{SecretId}"}

    def create_secret(self, Name, SecretString, Description=""):
        self.store[Name] = SecretString
        return {"ARN": f"arn:{Name}"}


class TestContainerSide:
    def test_secret_round_trip_into_the_environment(self, monkeypatch):
        store = {}
        monkeypatch.setattr(secrets, "_client", lambda session=None, region=None: _FakeSecrets(store))
        secrets.put_clearml_secret("labram/clearml", {"CLEARML_API_ACCESS_KEY": "AK",
                                                      "CLEARML_API_SECRET_KEY": PLANTED,
                                                      "CLEARML_API_HOST": "https://api"})
        assert secrets.secret_exists("labram/clearml")
        for k in secrets.CLEARML_SECRET_KEYS:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(CLEARML_SECRET_ENV, "labram/clearml")
        loaded = secrets.load_clearml_secret_into_env()
        import os
        assert sorted(loaded) == ["CLEARML_API_ACCESS_KEY", "CLEARML_API_HOST",
                                  "CLEARML_API_SECRET_KEY"]
        assert os.environ["CLEARML_API_SECRET_KEY"] == PLANTED

    def test_entry_point_fails_clearly_without_access(self, monkeypatch):
        from labram.runs import sagemaker_entry

        def denied(name=None, session=None):
            raise PermissionError("AccessDenied")
        monkeypatch.setattr(secrets, "load_clearml_secret_into_env", denied)
        monkeypatch.setenv(CLEARML_SECRET_ENV, "labram/clearml")
        with pytest.raises(RuntimeError, match="GetSecretValue"):
            sagemaker_entry.load_clearml_credentials()

    def test_entry_point_is_a_no_op_without_the_variable(self, monkeypatch):
        from labram.runs import sagemaker_entry
        monkeypatch.delenv(CLEARML_SECRET_ENV, raising=False)
        sagemaker_entry.load_clearml_credentials()


class TestCommandLine:
    def test_known_values_and_secret_assignments_are_masked(self):
        cmd = ('-m labram.runs.run_finetune --set clearml.enabled=true '
               'sagemaker.environment={"CLEARML_API_SECRET_KEY": "sk-1", "MY_DB_PASSWORD": "pw-2"} '
               f'X_TOKEN={PLANTED} other=fine')
        out = secrets.redact_text(cmd, secrets.secret_values(_config_with_secret()))
        assert "sk-1" not in out and "pw-2" not in out and PLANTED not in out
        assert "clearml.enabled=true" in out and "other=fine" in out

    def test_entry_point_is_rewritten_on_the_task(self):
        from labram.runs.common import _scrub_task_entry_point
        calls = {}
        task = types.SimpleNamespace(
            data=types.SimpleNamespace(script=types.SimpleNamespace(
                entry_point=f"-m x --set a.MY_TOKEN={PLANTED} b=1")),
            set_script=lambda **kw: calls.update(kw))
        _scrub_task_entry_point(task, _config_with_secret())
        assert PLANTED not in calls["entry_point"] and "b=1" in calls["entry_point"]
