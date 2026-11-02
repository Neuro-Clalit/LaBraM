"""Which source dataset each recording comes from, for per-dataset evaluation.

A cohort can mix corpora (``dataset_maker/make_joint_cohort.py``: Temple
University's TUH + the Harvard EEG Database), and every evaluation must be
reported per dataset as well as overall. The dataset of a recording is its
manifest row's ``corpus``, mapped to a family name:

* ``harvard``: ``heedb`` (Harvard EEG Database, BDSP);
* ``temple``: ``tuh`` (joint cohorts) and the TUH sub-corpora ``tuab``,
  ``tueg``, ``tuep``, ``tusz``, ``tuev``, ``tusl``, ``tuar``.

Any other corpus name is used as is; recordings without one are ``unknown``.

``train_eval_subset`` picks a fixed, seeded sample of training recordings per
dataset, so training-set metrics can be reported per epoch next to val/test.
"""
import copy
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from labram.data.tuh_metadata import recording_stem

DATASET_FAMILIES = {
    "heedb": "harvard",
    "tuh": "temple", "tuab": "temple", "tueg": "temple", "tuep": "temple",
    "tusz": "temple", "tuev": "temple", "tusl": "temple", "tuar": "temple",
}
UNKNOWN = "unknown"

_CACHE: Dict[str, Dict[str, str]] = {}


def dataset_family(corpus: Optional[str]) -> str:
    """``'heedb'`` -> ``'harvard'``, ``'tueg'`` -> ``'temple'``, None -> ``'unknown'``."""
    if not corpus:
        return UNKNOWN
    return DATASET_FAMILIES.get(str(corpus).lower(), str(corpus).lower())


def root_dataset_lookup(root: str) -> Dict[str, str]:
    """``{stem: dataset family}`` from ``<root>/manifest.json`` (cached); empty
    when the root has no npy manifest (e.g. pickle datasets)."""
    if root in _CACHE:
        return _CACHE[root]
    from labram.data.tuh_datasets import _read_manifest
    try:
        rows = _read_manifest(root)["recordings"]
    except (OSError, ValueError, KeyError, TypeError):
        rows = []
    out = {r["stem"]: dataset_family(r.get("corpus")) for r in rows if "stem" in r}
    _CACHE[root] = out
    return out


def _leaves(dataset: Any) -> List[Any]:
    from labram.data.window_selection import _leaves as leaves
    return leaves(dataset)


def dataset_lookup(dataset: Any) -> Dict[str, str]:
    """``{stem: dataset family}`` over every leaf (root) of ``dataset``."""
    out: Dict[str, str] = {}
    for leaf in _leaves(dataset):
        root = getattr(leaf, "root", None)
        if root:
            out.update(root_dataset_lookup(root))
    return out


def train_eval_subset(dataset_train: Any, per_dataset: int, windows_per_recording: int,
                      seed: int = 0) -> Optional[Any]:
    """A deterministic evaluation copy of the training data: up to
    ``per_dataset`` recordings of each dataset family (seeded draw), each with
    its first ``windows_per_recording`` listed training windows. The leaves are
    shallow copies with ``files`` replaced, so targets, channel handling and
    window ids behave as in training (random crops off). None when nothing is
    selected or the data has no file lists."""
    if dataset_train is None or per_dataset <= 0:
        return None
    rng = random.Random(seed)
    parts = []
    for leaf in _leaves(dataset_train):
        files = getattr(leaf, "files", None)
        if not files:
            continue
        by_rec: "OrderedDict[str, List[str]]" = OrderedDict()
        for f in files:
            by_rec.setdefault(recording_stem(f), []).append(f)
        lookup = root_dataset_lookup(getattr(leaf, "root", "") or "")
        by_ds: Dict[str, List[str]] = {}
        for stem in by_rec:
            by_ds.setdefault(lookup.get(stem, UNKNOWN), []).append(stem)
        chosen = []
        for ds in sorted(by_ds):
            stems = by_ds[ds]
            chosen += sorted(rng.sample(stems, min(per_dataset, len(stems))))
        keep = [f for stem in chosen for f in by_rec[stem][:max(1, windows_per_recording)]]
        if not keep:
            continue
        part = copy.copy(leaf)
        part.files = keep
        if hasattr(part, "random_crop"):
            part.random_crop = False
        if hasattr(part, "_mmaps"):
            part._mmaps = OrderedDict()            # do not share open memmaps
        parts.append(part)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    import torch
    return torch.utils.data.ConcatDataset(parts)
