# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Optionally stop the (EC2) machine after training finishes, with a delay.
# ---------------------------------------------------------

import os
import subprocess
import time
import urllib.request
from typing import Any, Optional

from labram.utils.logging import get_logger

logger = get_logger(__name__)

_IMDS_BASE = "http://169.254.169.254/latest"
_IMDS_TIMEOUT = 2.0


def _imds_token() -> Optional[str]:
    """Fetch an IMDSv2 session token (falls back to IMDSv1 if this fails)."""
    try:
        req = urllib.request.Request(
            f"{_IMDS_BASE}/api/token", method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        with urllib.request.urlopen(req, timeout=_IMDS_TIMEOUT) as resp:
            return resp.read().decode()
    except Exception:
        return None


def _imds_get(path: str, token: Optional[str]) -> Optional[str]:
    try:
        headers = {"X-aws-ec2-metadata-token": token} if token else {}
        req = urllib.request.Request(f"{_IMDS_BASE}/meta-data/{path}", headers=headers)
        with urllib.request.urlopen(req, timeout=_IMDS_TIMEOUT) as resp:
            return resp.read().decode().strip()
    except Exception:
        return None


def get_instance_identity() -> Optional[dict]:
    """Return ``{'instance_id', 'region'}`` from EC2 IMDS, or ``None`` off-EC2."""
    token = _imds_token()
    instance_id = _imds_get("instance-id", token)
    if not instance_id:
        return None
    region = _imds_get("placement/region", token)
    if not region:
        az = _imds_get("placement/availability-zone", token)
        region = az[:-1] if az else None
    return {"instance_id": instance_id, "region": region}


def _stop_via_ec2(delay_seconds: int) -> None:
    identity = get_instance_identity()
    if identity is None:
        logger.warning(
            "stop_instance_on_finish: could not read EC2 instance metadata "
            "(not on EC2 or IMDS unreachable); skipping shutdown.")
        return
    logger.warning(
        "Training finished. Stopping EC2 instance %s (region %s) in %d seconds.",
        identity["instance_id"], identity["region"], delay_seconds)
    time.sleep(max(0, delay_seconds))
    try:
        import boto3
        client = boto3.client("ec2", region_name=identity["region"])
        client.stop_instances(InstanceIds=[identity["instance_id"]])
        logger.warning("stop_instances issued for %s.", identity["instance_id"])
    except Exception as exc:  # pragma: no cover - depends on AWS creds/permissions
        logger.error("Failed to stop EC2 instance: %s", exc)


def _stop_via_os(delay_minutes: int) -> None:
    logger.warning(
        "Training finished. Powering off this machine in %d minute(s) via "
        "`shutdown -h`.", delay_minutes)
    try:
        subprocess.Popen(["shutdown", "-h", f"+{delay_minutes}"])
    except Exception as exc:  # pragma: no cover - needs privileges / shutdown bin
        logger.error("Failed to schedule OS shutdown: %s", exc)


def maybe_stop_instance(shutdown_cfg: Any) -> bool:
    """Stop the machine after training if ``shutdown_cfg`` enables it.

    ``stop_method='ec2'`` waits ``stop_delay_minutes`` then issues a boto3
    ``stop_instances`` for this instance (id/region from EC2 IMDS); ``'os'`` runs
    ``shutdown -h +<minutes>``. Best-effort: any failure is logged, never
    raised, so it can't turn a successful run into a crash. Returns True when a
    stop was attempted. The caller is responsible for the rank-0 guard.
    """
    if shutdown_cfg is None or not getattr(shutdown_cfg, "stop_instance_on_finish", False):
        return False

    delay_minutes = max(0, int(getattr(shutdown_cfg, "stop_delay_minutes", 5)))
    method = getattr(shutdown_cfg, "stop_method", "ec2")

    if os.environ.get("LABRAM_DISABLE_SHUTDOWN"):
        logger.warning("LABRAM_DISABLE_SHUTDOWN set; skipping machine stop.")
        return False

    if method == "os":
        _stop_via_os(delay_minutes)
    elif method == "ec2":
        _stop_via_ec2(delay_minutes * 60)
    else:
        logger.error("Unknown stop_method %r; expected 'ec2' or 'os'.", method)
        return False
    return True
