# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Explicitly register the trained model as a ClearML artifact in the (S3)
# output_uri, rather than relying solely on framework auto-capture.
# ---------------------------------------------------------

import os
from typing import Any, Optional

from labram.utils.logging import get_logger

logger = get_logger(__name__)


def get_clearml_task(log_writer: Any) -> Optional[Any]:
    """Recover the ClearML ``Task`` from a log writer, if one is present.

    Handles a bare ``ClearMLLogger`` and a ``MultiWriter`` fanning out to one.
    Returns ``None`` when ClearML is not in use.
    """
    if log_writer is None:
        return None
    task = getattr(log_writer, "task", None)
    if task is not None:
        return task
    for writer in getattr(log_writer, "writers", []) or []:
        task = getattr(writer, "task", None)
        if task is not None:
            return task
    return None


def resolve_final_checkpoint(output_dir: str) -> Optional[str]:
    """Best available checkpoint to publish: prefer ``checkpoint-best.pth``,
    else the rolling ``checkpoint.pth``, else the highest-epoch checkpoint."""
    if not output_dir:
        return None
    best = os.path.join(output_dir, "checkpoint-best.pth")
    if os.path.exists(best):
        return best
    final = os.path.join(output_dir, "checkpoint-final.pth")
    if os.path.exists(final):
        return final
    rolling = os.path.join(output_dir, "checkpoint.pth")
    if os.path.exists(rolling):
        return rolling
    import glob
    candidates = []
    for path in glob.glob(os.path.join(output_dir, "checkpoint-*.pth")):
        tag = path.rsplit("-", 1)[-1].split(".")[0]
        if tag.isdigit():
            candidates.append((int(tag), path))
    if candidates:
        return max(candidates)[1]
    return None


def upload_file_artifact(task: Any, name: str, path: str) -> bool:
    """Attach a file (e.g. the data-split JSON or the model graph) to ``task``
    as a ClearML artifact. Best-effort: returns True on success, else False."""
    if task is None or not path or not os.path.exists(path):
        return False
    try:
        task.upload_artifact(name=name, artifact_object=path)
        logger.info("Uploaded ClearML artifact %r <- %s", name, path)
        return True
    except Exception as exc:  # pragma: no cover - depends on clearml availability
        logger.error("Failed to upload ClearML artifact %r: %s", name, exc)
        return False


def upload_model_artifact(
    task: Any,
    model_path: str,
    name: Optional[str] = None,
) -> Optional[str]:
    """Register ``model_path`` as a ClearML ``OutputModel`` on ``task``.

    ``OutputModel.update_weights`` uploads the checkpoint to the task's
    configured ``output_uri`` (e.g. the S3 bucket) and records it in the model
    registry. Best-effort: returns the uploaded URI on success, else ``None``
    (logging, never raising, so tracking never fails a finished run).
    """
    if task is None or not model_path or not os.path.exists(model_path):
        logger.warning(
            "Skipping ClearML model upload: task=%s model_path=%r exists=%s",
            task is not None, model_path, os.path.exists(model_path) if model_path else False)
        return None
    try:
        from clearml import OutputModel
        output_model = OutputModel(task=task, name=name or "trained_model", framework="PyTorch")
        uri = output_model.update_weights(weights_filename=model_path, auto_delete_file=False)
        logger.info("Registered ClearML model artifact %s -> %s", model_path, uri)
        return uri
    except Exception as exc:  # pragma: no cover - depends on clearml/S3 availability
        logger.error("Failed to upload ClearML model artifact: %s", exc)
        return None
