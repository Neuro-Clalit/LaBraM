# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Keep credentials out of run configs: redaction for everything a config is
# written to (run_config.yaml, S3, ClearML, logs), and the Secrets Manager hop
# that hands ClearML credentials to SageMaker jobs.
# ---------------------------------------------------------
import copy
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

REDACTED = "***"
# A mapping key whose name looks like a credential has its value masked.
_SECRET_NAME = re.compile(r"(SECRET|PASSWORD|PASSWD|TOKEN|ACCESS_KEY|PRIVATE_KEY|API_KEY|CREDENTIAL)",
                          re.IGNORECASE)

# Job env var naming the Secrets Manager secret that holds the ClearML credentials.
CLEARML_SECRET_ENV = "LABRAM_CLEARML_SECRET"
DEFAULT_CLEARML_SECRET = "labram/clearml"
CLEARML_SECRET_KEYS = ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY", "CLEARML_API_HOST",
                       "CLEARML_WEB_HOST", "CLEARML_FILES_HOST")


def is_secret_name(name: Any) -> bool:
    return isinstance(name, str) and bool(_SECRET_NAME.search(name))


def redact(obj: Any) -> Any:
    """Copy of ``obj`` (nested dicts/lists) with every non-empty value under a
    secret-looking key replaced by ``***``."""
    if isinstance(obj, dict):
        return {k: (REDACTED if is_secret_name(k) and v not in (None, "") else redact(v))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact(v) for v in obj)
    return obj


def redacted_copy(config):
    """Deep copy of a ``ConfigBase`` tree whose dict-valued fields (e.g.
    ``sagemaker.environment``) are redacted -- what is safe to save, upload,
    log or hand to ClearML."""
    clone = copy.deepcopy(config)
    _redact_fields(clone)
    return clone


def _redact_fields(node) -> None:
    for name, value in vars(node).items():
        if isinstance(value, dict):
            setattr(node, name, redact(value))
        elif hasattr(value, "__dataclass_fields__"):
            _redact_fields(value)


def secret_values(config) -> List[str]:
    """Values held under secret-looking keys anywhere in a ``ConfigBase`` tree."""
    found: List[str] = []

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if is_secret_name(k) and isinstance(v, str) and v:
                    found.append(v)
                else:
                    walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)
        elif hasattr(obj, "__dataclass_fields__"):
            walk(vars(obj))
    walk(config)
    return found


# NAME=value / "NAME": "value" pairs whose name looks like a credential.
_SECRET_ASSIGNMENT = re.compile(
    r"""(?P<name>[\w.-]*(?:SECRET|PASSWORD|PASSWD|TOKEN|ACCESS_KEY|PRIVATE_KEY|API_KEY|CREDENTIAL)[\w.-]*)"""
    r"""(?P<sep>["']?\s*[=:]\s*["']?)(?P<value>[^\s"',}]+)""", re.IGNORECASE)


def redact_text(text: str, values: Iterable[str] = ()) -> str:
    """Mask known secret ``values`` and secret-looking ``NAME=value`` pairs in
    free text such as a recorded command line."""
    for v in sorted(set(values), key=len, reverse=True):
        if len(v) >= 4:
            text = text.replace(v, REDACTED)
    return _SECRET_ASSIGNMENT.sub(lambda m: m.group("name") + m.group("sep") + REDACTED, text)


# ------------------------------------------------------------ Secrets Manager
def _client(session=None, region: Optional[str] = None):
    import boto3
    session = session or boto3.Session()
    return session.client("secretsmanager", region_name=region or _region(session))


def _region(session) -> Optional[str]:
    """The session's region, else the one in ``$TRAINING_JOB_ARN`` -- SageMaker
    containers set neither AWS_REGION nor a config file, only that ARN."""
    if session.region_name:
        return session.region_name
    parts = os.environ.get("TRAINING_JOB_ARN", "").split(":")
    return parts[3] if len(parts) > 3 and parts[3] else None


def load_clearml_secret_into_env(name: Optional[str] = None, session=None) -> List[str]:
    """Read the ClearML credentials secret and export its ``CLEARML_*`` entries
    (values already in the environment win). Returns the names set; values are
    never logged. ``name`` defaults to ``$LABRAM_CLEARML_SECRET``."""
    name = name or os.environ.get(CLEARML_SECRET_ENV)
    if not name:
        return []
    raw = _client(session).get_secret_value(SecretId=name)["SecretString"]
    values = json.loads(raw)
    set_names = []
    for key in CLEARML_SECRET_KEYS:
        if values.get(key) and not os.environ.get(key):
            os.environ[key] = str(values[key])
            set_names.append(key)
    return set_names


def secret_exists(name: str, session=None) -> bool:
    client = _client(session)
    try:
        client.describe_secret(SecretId=name)
        return True
    except client.exceptions.ResourceNotFoundException:
        return False


def put_clearml_secret(name: str, values: Dict[str, str], session=None) -> str:
    """Create or update the secret with the given ``CLEARML_*`` values; returns its ARN."""
    payload = json.dumps({k: values[k] for k in CLEARML_SECRET_KEYS if values.get(k)})
    client = _client(session)
    try:
        return client.put_secret_value(SecretId=name, SecretString=payload)["ARN"]
    except client.exceptions.ResourceNotFoundException:
        return client.create_secret(
            Name=name, SecretString=payload,
            Description="ClearML API credentials for LaBraM SageMaker jobs")["ARN"]


def missing_keys(values: Dict[str, str], required: Iterable[str]) -> List[str]:
    return [k for k in required if not values.get(k)]
