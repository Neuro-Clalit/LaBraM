# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Stage npy recordings onto fast local storage (instance NVMe) before training.
# ---------------------------------------------------------
"""Copy the recordings a run reads onto the instance-store NVMe, then train
from there.

The GPU box's ``/data`` is a gp3 EBS volume (125 MB/s); the instance store at
``/opt/dlami/nvme`` is several times faster but ephemeral (wiped when the
instance stops) and smaller (~400 GB) than a full corpus. So staging is

* **selective** -- only the recordings the run's train/val/test splits use
  (after window selection), not the whole corpus;
* **prioritised** -- train first, then val, then test: when the NVMe cannot
  hold everything, what does not fit is read from ``/data`` through a symlink,
  so the run is never blocked, only partly accelerated;
* **reused** -- a staged mirror persists across runs (until the instance stops)
  and a recording already copied is not copied again.

The mirror (``<stage_dir>/<source path>/``) holds copies of the source's small
JSON files (manifest, age sidecars) and ``recordings/<stem>.npy`` as either a
copy or a symlink to the source; loaders are simply re-rooted at it, so every
other behaviour (splits, window selection, CV rebuilds) is unchanged.
"""

import concurrent.futures as cf
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch.distributed as dist

from labram.data.tuh_datasets import NpyWindowsMixin
from labram.data.tuh_metadata import recording_stem
from labram.data.window_selection import _leaves

logger = logging.getLogger(__name__)

AUTO = "auto"
NVME_CANDIDATES = ("/opt/dlami/nvme",)
STAGE_SUBDIR = "labram_stage"
SPLIT_PRIORITY = ("train", "val", "test")
GB = 1e9


@dataclass
class StageReport:
    mirrors: Dict[str, str] = field(default_factory=dict)        # source root -> mirror
    staged: Dict[str, int] = field(default_factory=dict)         # split -> recordings on NVMe
    linked: Dict[str, int] = field(default_factory=dict)         # split -> read from source
    copied_bytes: int = 0
    seconds: float = 0.0


def resolve_stage_dir(stage_dir: str) -> Optional[str]:
    """``auto`` -> the instance NVMe when present and writable (never inside a
    SageMaker container, whose File-mode channels are local already); ``""`` /
    ``none`` -> disabled; anything else is used as given."""
    if not stage_dir or stage_dir.lower() == "none":
        return None
    if stage_dir != AUTO:
        return os.path.join(stage_dir, STAGE_SUBDIR)
    if os.environ.get("SM_TRAINING_ENV") or os.path.isdir("/opt/ml/input/data"):
        return None
    for cand in NVME_CANDIDATES:
        if os.path.isdir(cand) and os.access(cand, os.W_OK):
            return os.path.join(cand, STAGE_SUBDIR)
    return None


def mirror_root(stage_root: str, source_root: str) -> str:
    return os.path.join(stage_root, os.path.abspath(source_root).strip(os.sep))


def _npy_leaves(dataset) -> List[NpyWindowsMixin]:
    return [leaf for leaf in _leaves(dataset) if isinstance(leaf, NpyWindowsMixin)]


def _ordered_needs(bundle, splits):
    """[(split, source_root, stem)] in priority order, each recording once."""
    seen, out = set(), []
    for split in splits:
        for leaf in _npy_leaves(getattr(bundle, split, None)):
            for f in leaf.files:
                key = (leaf.root, recording_stem(f))
                if key not in seen:
                    seen.add(key)
                    out.append((split, leaf.root, key[1]))
    return out


def _copy_atomic(src, dst):
    tmp = f"{dst}.tmp{os.getpid()}"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)            # never leave a half-written recording


def _link(src, dst):
    tmp = f"{dst}.lnk{os.getpid()}"
    os.symlink(src, tmp)
    os.replace(tmp, dst)


def _prepare_mirror(source, mirror):
    os.makedirs(os.path.join(mirror, "recordings"), exist_ok=True)
    for name in os.listdir(source):            # manifest + sidecars (small JSON)
        path = os.path.join(source, name)
        if name.endswith(".json") and os.path.isfile(path):
            _copy_atomic(path, os.path.join(mirror, name))


def _do_stage(bundle, stage_root, reserve_gb, threads, splits) -> StageReport:
    t0 = time.time()
    report = StageReport()
    needs = _ordered_needs(bundle, splits)
    for _, root, _ in needs:
        if root not in report.mirrors:
            report.mirrors[root] = mirror_root(stage_root, root)
            _prepare_mirror(root, report.mirrors[root])
    budget = shutil.disk_usage(stage_root).free - reserve_gb * GB
    to_copy = []
    for split, root, stem in needs:
        src = os.path.join(root, "recordings", f"{stem}.npy")
        dst = os.path.join(report.mirrors[root], "recordings", f"{stem}.npy")
        size = os.path.getsize(src)
        if os.path.isfile(dst) and not os.path.islink(dst) and os.path.getsize(dst) == size:
            report.staged[split] = report.staged.get(split, 0) + 1     # from an earlier run
        elif size <= budget:
            budget -= size
            to_copy.append((src, dst, size))
            report.staged[split] = report.staged.get(split, 0) + 1
        else:
            if not (os.path.islink(dst) and os.readlink(dst) == src):
                _link(src, dst)
            report.linked[split] = report.linked.get(split, 0) + 1
    with cf.ThreadPoolExecutor(threads) as pool:
        for _ in pool.map(lambda job: _copy_atomic(job[0], job[1]), to_copy):
            pass
    report.copied_bytes = sum(size for *_, size in to_copy)
    report.seconds = time.time() - t0
    return report


def stage_bundle(bundle, stage_dir: str = AUTO, *, reserve_gb: float = 20.0,
                 threads: int = 8, splits=SPLIT_PRIORITY) -> Optional[StageReport]:
    """Stage the npy recordings ``bundle`` uses and re-root its loaders at the
    mirror. Returns None (and changes nothing) when staging is disabled, there
    is nothing npy-backed to stage, or the data is already on the stage disk.

    ``splits`` is the priority order and the set staged: a run that only
    evaluates passes ``("val", "test")`` so no train recording is copied. Only
    the loaders of those splits are re-rooted; any other split keeps reading
    its source.

    Under DDP only rank 0 copies; the others wait at a barrier and re-root."""
    stage_root = resolve_stage_dir(stage_dir)
    if stage_root is None:
        return None
    leaves = [leaf for split in splits
              for leaf in _npy_leaves(getattr(bundle, split, None))]
    if not leaves or all(os.path.abspath(leaf.root).startswith(os.path.dirname(stage_root))
                         for leaf in leaves):
        return None
    os.makedirs(stage_root, exist_ok=True)
    distributed = dist.is_available() and dist.is_initialized()
    report = None
    if not distributed or dist.get_rank() == 0:
        report = _do_stage(bundle, stage_root, reserve_gb, threads, splits)
        for split in splits:
            n_s, n_l = report.staged.get(split, 0), report.linked.get(split, 0)
            if n_s or n_l:
                logger.info("staging [%s]: %d recording(s) on %s, %d read from source",
                            split, n_s, stage_root, n_l)
        logger.info("staging: copied %.1f GB in %.0f s (reserve %.0f GB kept free)",
                    report.copied_bytes / GB, report.seconds, reserve_gb)
    if distributed:
        dist.barrier()
    for leaf in leaves:
        leaf.root = mirror_root(stage_root, leaf.root)
        leaf._mmaps.clear()
    return report
