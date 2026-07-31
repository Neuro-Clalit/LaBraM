# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# K-fold cross-validation for fine-tuning: partition the data pool into
# group-disjoint folds (grouped by subject / recording / window), materialize a
# fold's train/val/test datasets, and record the partition to a reproducible
# ``cv_split.json`` artifact. See docs/cross_validation.md.
# ---------------------------------------------------------

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch.utils.data

from labram.data.bundles import DatasetBundle
from labram.utils.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------ group ids


def _strip_ext(filename: str) -> str:
    # A window file may be given as a path relative to the loader root (the age
    # split stores 'train/<name>.pkl'), so group ids come off the basename.
    base = os.path.basename(filename)
    return base[:-4] if base.endswith(".pkl") else base


def group_id_for(source, filename: str, split_by: str) -> str:
    """Group id of a window file under a source loader.

    ``subject`` -> leading underscore token; ``recording`` -> filename with the
    trailing ``<sep><window_index>`` stripped (``_`` for TUAB, ``-`` for TUEV);
    ``window`` -> the file itself (no grouping, plain K-fold over windows).
    """
    base = _strip_ext(filename)
    if split_by == "window":
        return base
    if split_by == "subject":
        return base.split("_")[0]
    sep = getattr(source, "_recording_sep", "_")
    return base.rsplit(sep, 1)[0]


# ------------------------------------------------------------------ sources


def _iter_pool_sources(bundle: DatasetBundle, pool: str):
    """Yield ``(name, dataset)`` for each source loader in the CV pool.

    ``train_val`` keeps the original test set out of the folds; ``all`` pools
    every split. ``None`` splits are skipped.
    """
    named = [("train", bundle.train), ("val", bundle.val)]
    if pool == "all":
        named.append(("test", bundle.test))
    for name, ds in named:
        if ds is not None:
            yield name, ds


def _source_descriptor(dataset) -> Dict[str, Any]:
    return {
        "loader": type(dataset).__name__,
        "root": getattr(dataset, "root", None),
        "sampling_rate": getattr(dataset, "sampling_rate", 200),
        "n_files": len(getattr(dataset, "files", []) or []),
    }


# ------------------------------------------------------------------ folds


@dataclass
class GroupedFolds:
    """A group-disjoint K-fold partition plus everything needed to materialize
    each fold's train/val/test datasets."""

    n_folds: int
    split_by: str
    pool: str
    seed: int
    shuffle: bool
    dataset_name: Optional[str]
    # Ordered source loaders that were pooled (raw dataset objects).
    sources: List[Any] = field(default_factory=list)
    # Per-source descriptor dicts (loader/root/sampling_rate/n_files).
    source_meta: List[Dict[str, Any]] = field(default_factory=list)
    # The K group partitions (each a sorted list of group ids).
    fold_groups: List[List[str]] = field(default_factory=list)
    # group id -> list of (source_index, filename).
    group_files: Dict[str, List[Tuple[int, str]]] = field(default_factory=dict)
    # Summary of the untouched held-out test set (pool == 'train_val').
    held_out_test: Optional[Dict[str, Any]] = None

    def fold_indices(self, k: int) -> Tuple[int, int]:
        """(test_fold, val_fold) for fold ``k``: test = k, val = next fold."""
        return k, (k + 1) % self.n_folds


def build_grouped_folds(bundle: DatasetBundle, cv_cfg, dataset_name: Optional[str] = None) -> GroupedFolds:
    """Partition the CV pool of ``bundle`` into ``cv_cfg.n_folds`` group-disjoint
    folds deterministically from ``(split_by, seed, shuffle)``."""
    sources: List[Any] = []
    source_meta: List[Dict[str, Any]] = []
    group_files: Dict[str, List[Tuple[int, str]]] = {}

    for _name, ds in _iter_pool_sources(bundle, cv_cfg.pool):
        idx = len(sources)
        sources.append(ds)
        source_meta.append(_source_descriptor(ds))
        for f in getattr(ds, "files", []) or []:
            g = group_id_for(ds, f, cv_cfg.split_by)
            group_files.setdefault(g, []).append((idx, f))

    groups = sorted(group_files.keys())
    if not groups:
        raise ValueError(
            "Cross-validation found no files to split; check the dataset path "
            "and that the loaders expose a `.files` list.")
    if len(groups) < cv_cfg.n_folds:
        raise ValueError(
            f"Cannot make {cv_cfg.n_folds} folds from only {len(groups)} "
            f"{cv_cfg.split_by} group(s). Reduce n_folds or change split_by.")

    order = np.arange(len(groups))
    if cv_cfg.shuffle:
        np.random.RandomState(cv_cfg.seed).shuffle(order)
    fold_groups = [sorted(groups[i] for i in chunk)
                   for chunk in np.array_split(order, cv_cfg.n_folds)]

    held_out_test = None
    if cv_cfg.pool == "train_val" and bundle.test is not None:
        held_out_test = _split_summary(bundle.test, cv_cfg.split_by)

    return GroupedFolds(
        n_folds=cv_cfg.n_folds, split_by=cv_cfg.split_by, pool=cv_cfg.pool,
        seed=cv_cfg.seed, shuffle=cv_cfg.shuffle, dataset_name=dataset_name,
        sources=sources, source_meta=source_meta, fold_groups=fold_groups,
        group_files=group_files, held_out_test=held_out_test)


def _split_summary(dataset, split_by: str) -> Dict[str, Any]:
    files = sorted(getattr(dataset, "files", []) or [])
    groups = sorted({group_id_for(dataset, f, split_by) for f in files})
    return {"n_windows": len(files), "n_groups": len(groups), "groups": groups}


# ------------------------------------------------------------------ materialize


# Loader attributes that must survive the positional rebuild in
# _build_split_dataset (which cannot pass keyword arguments).
_CARRIED_SOURCE_ATTRS = ("target_stats",)


def _build_split_dataset(folds: GroupedFolds, groups: List[str]):
    """ConcatDataset of one per-source loader restricted to ``groups``' files."""
    per_source: Dict[int, List[str]] = {}
    for g in groups:
        for src_idx, fname in folds.group_files.get(g, []):
            per_source.setdefault(src_idx, []).append(fname)

    parts = []
    for src_idx, files in sorted(per_source.items()):
        src = folds.sources[src_idx]
        # Rebuild the same loader class over just this split's files from the
        # source's root, preserving its sampling rate.
        loader = type(src)(src.root, sorted(files), getattr(src, "sampling_rate", 200))
        # Carry over state the positional constructor cannot receive. Notably a
        # regression target's normalization stats: without them the fold would
        # yield raw targets that the eval path then de-normalizes a second time.
        for attr in _CARRIED_SOURCE_ATTRS:
            if hasattr(src, attr):
                setattr(loader, attr, getattr(src, attr))
        parts.append(loader)

    if not parts:
        raise ValueError("A cross-validation split resolved to zero samples; "
                         "check n_folds vs. the number of groups.")
    if len(parts) == 1:
        return parts[0]
    return torch.utils.data.ConcatDataset(parts)


def materialize_fold(folds: GroupedFolds, k: int, base_bundle: DatasetBundle) -> DatasetBundle:
    """Build the ``k``-th fold's train/val/test :class:`DatasetBundle`.

    test = fold ``k``; val = fold ``(k+1) % K``; train = the remaining folds.
    When ``pool == 'train_val'`` the original test set is discarded from CV (it
    stays available for a separate final evaluation); the held-out fold is used
    as the fold's test set so every case is tested exactly once across folds.
    """
    if not 0 <= k < folds.n_folds:
        raise ValueError(f"fold {k} out of range 0..{folds.n_folds - 1}")

    test_fold, val_fold = folds.fold_indices(k)
    test_groups = folds.fold_groups[test_fold]
    val_groups = folds.fold_groups[val_fold]
    train_groups = [g for i, chunk in enumerate(folds.fold_groups)
                    if i not in (test_fold, val_fold) for g in chunk]

    dataset_train = _build_split_dataset(folds, train_groups)
    dataset_val = _build_split_dataset(folds, val_groups)
    dataset_test = _build_split_dataset(folds, test_groups)

    return DatasetBundle(
        train=dataset_train, val=dataset_val, test=dataset_test,
        ch_names=base_bundle.ch_names, nb_classes=base_bundle.nb_classes,
        metrics=base_bundle.metrics, task=base_bundle.task,
        target_stats=base_bundle.target_stats)


# ------------------------------------------------------------------ artifact


def cv_split_to_dict(folds: GroupedFolds) -> Dict[str, Any]:
    """Serializable record of the fold partition (the reproducibility artifact)."""
    fold_entries = []
    for k in range(folds.n_folds):
        test_fold, val_fold = folds.fold_indices(k)
        groups = folds.fold_groups[k]
        n_windows = sum(len(folds.group_files.get(g, [])) for g in groups)
        fold_entries.append({
            "fold": k,
            "n_groups": len(groups),
            "n_windows": n_windows,
            "groups": groups,
            "role_as_test_for_fold": k,
            "role_as_val_for_fold": (k - 1) % folds.n_folds,
        })
    return {
        "dataset": folds.dataset_name,
        "n_folds": folds.n_folds,
        "split_by": folds.split_by,
        "pool": folds.pool,
        "seed": folds.seed,
        "shuffle": folds.shuffle,
        "convention": "fold k -> test; fold (k+1)%K -> val; remainder -> train",
        "sources": folds.source_meta,
        "folds": fold_entries,
        "held_out_test": folds.held_out_test,
    }


def save_cv_split(folds: GroupedFolds, output_dir: str,
                  filename: str = "cv_split.json") -> Optional[str]:
    """Write the fold partition to ``output_dir/filename``; return the path."""
    if not output_dir:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cv_split_to_dict(folds), f, indent=2, ensure_ascii=False)
    logger.info("Saved CV split (%d folds, split_by=%s) -> %s",
                folds.n_folds, folds.split_by, path)
    return path


def load_cv_split_dict(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_cv_split(bundle: DatasetBundle, split: Dict[str, Any],
                   dataset_name: Optional[str] = None) -> GroupedFolds:
    """Rebuild :class:`GroupedFolds` from a saved ``cv_split.json`` and the
    current ``bundle``, so a separately-dispatched fold job re-uses the exact
    same partition. Group membership is recomputed from the bundle's files;
    the fold assignment comes verbatim from ``split['folds']``.
    """
    split_by = split["split_by"]
    pool = split.get("pool", "train_val")

    sources: List[Any] = []
    source_meta: List[Dict[str, Any]] = []
    group_files: Dict[str, List[Tuple[int, str]]] = {}
    for _name, ds in _iter_pool_sources(bundle, pool):
        idx = len(sources)
        sources.append(ds)
        source_meta.append(_source_descriptor(ds))
        for f in getattr(ds, "files", []) or []:
            g = group_id_for(ds, f, split_by)
            group_files.setdefault(g, []).append((idx, f))

    fold_groups = [sorted(entry["groups"]) for entry in split["folds"]]
    return GroupedFolds(
        n_folds=split["n_folds"], split_by=split_by, pool=pool,
        seed=split.get("seed", 0), shuffle=split.get("shuffle", True),
        dataset_name=dataset_name or split.get("dataset"),
        sources=sources, source_meta=source_meta, fold_groups=fold_groups,
        group_files=group_files, held_out_test=split.get("held_out_test"))


def subject_overlap(folds: GroupedFolds) -> Dict[str, List[str]]:
    """Cross-fold group overlap; empty by construction (a leakage self-check)."""
    overlap: Dict[str, List[str]] = {}
    for i in range(folds.n_folds):
        for j in range(i + 1, folds.n_folds):
            common = sorted(set(folds.fold_groups[i]) & set(folds.fold_groups[j]))
            if common:
                overlap[f"fold{i}∩fold{j}"] = common
    return overlap
