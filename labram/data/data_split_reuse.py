# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Reuse a recorded ``data_split.json`` (see labram.utils.data_split) so several
# fine-tune runs share the *exact* same train/val/test case assignment — e.g. to
# compare models on an identical split. Reads local paths or ``s3://`` URIs.
# ---------------------------------------------------------

import json
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch.utils.data

from labram.data.bundles import DatasetBundle
from labram.data.cross_validation import CARRIED_SOURCE_ATTRS
from labram.utils.logging import get_logger

logger = get_logger(__name__)


def load_data_split_json(path: str) -> Dict[str, Any]:
    """Read a ``data_split.json`` from a local path or an ``s3://`` URI.

    S3 is handled by the shared :class:`FileSystem` (download to a temp file,
    then parse), so this works both on the submitting machine and inside a
    SageMaker container that has S3 access via its execution role.
    """
    if str(path).startswith("s3://"):
        from labram.file_system import FileSystem
        fs = FileSystem()
        return fs.load_object(path, lambda p: json.load(open(p, encoding="utf-8")))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _file_to_source(bundle: DatasetBundle) -> Dict[str, Any]:
    """Map each window filename -> the loader (root/class/rate) that holds it,
    scanning the bundle's train/val/test loaders."""
    index: Dict[str, Any] = {}
    for ds in (bundle.train, bundle.val, bundle.test):
        if ds is None:
            continue
        for f in getattr(ds, "files", []) or []:
            index.setdefault(f, ds)
    return index


def _build_split_dataset(index: Dict[str, Any], files: List[str], split_name: str):
    """Rebuild one split's dataset from recorded ``files``, grouping each file
    under its physical source loader (so a file keeps its real root even if the
    recorded split moved it across the original train/val/test dirs)."""
    per_source: "OrderedDict[int, Any]" = OrderedDict()
    missing: List[str] = []
    for f in files:
        ds = index.get(f)
        if ds is None:
            missing.append(f)
            continue
        per_source.setdefault(id(ds), (ds, []))[1].append(f)
    if missing:
        logger.warning(
            "data_split reuse [%s]: %d recorded file(s) not found under data_path "
            "(e.g. %s)", split_name, len(missing), missing[:3])

    parts = []
    for _key, (ds, fs_) in per_source.items():
        loader = type(ds)(ds.root, sorted(fs_), getattr(ds, "sampling_rate", 200))
        # Carry over state this positional rebuild cannot pass (notably a
        # regression target's normalization stats); see cross_validation.
        for attr in CARRIED_SOURCE_ATTRS:
            if hasattr(ds, attr):
                setattr(loader, attr, getattr(ds, attr))
        parts.append(loader)
    if not parts:
        raise ValueError(
            f"data_split reuse produced an empty '{split_name}' split; none of its "
            f"recorded files were found under the current data_path.")
    return parts[0] if len(parts) == 1 else torch.utils.data.ConcatDataset(parts)


def apply_data_split(bundle: DatasetBundle, split: Dict[str, Any]) -> DatasetBundle:
    """Return a new bundle whose train/val/test are rebuilt from the recorded
    ``split`` (a parsed ``data_split.json``), keeping the bundle's channel
    names / class count / metrics / task. Splits absent from the record become
    ``None``.
    """
    index = _file_to_source(bundle)
    out: Dict[str, Optional[Any]] = {}
    for name in ("train", "val", "test"):
        entry = split.get(name)
        files = entry.get("files") if isinstance(entry, dict) else None
        out[name] = _build_split_dataset(index, files, name) if files else None

    counts = {k: (len(v) if v is not None else 0) for k, v in out.items()}
    logger.info("Reused recorded data split (window counts: %s)", counts)
    # ``task``/``target_stats`` must travel with the bundle: without them a reused
    # split would silently downgrade a regression run to classification (both are
    # nb_classes == 1) and drop the target normalization.
    return DatasetBundle(
        train=out["train"], val=out["val"], test=out["test"],
        ch_names=bundle.ch_names, nb_classes=bundle.nb_classes, metrics=bundle.metrics,
        task=bundle.task, target_stats=bundle.target_stats)


def bundle_from_data_split(dataset_name: str, data_path: str, split_json: str) -> DatasetBundle:
    """Build the dataset's default bundle, then re-partition it to match the
    recorded ``split_json``."""
    from labram.data.bundles import get_dataset_bundle
    base = get_dataset_bundle(dataset_name, data_path)
    return apply_data_split(base, load_data_split_json(split_json))
