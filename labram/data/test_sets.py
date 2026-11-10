"""Named extra test sets scored next to the regular test split (``evaluation.test_sets``).

Each entry of ``evaluation.test_sets`` is one of

* ``<family>`` -- the run's own test split restricted to one source dataset
  family (``temple`` = TUH, ``harvard`` = HEEDB; ``labram/data/dataset_groups.py``),
  e.g. ``temple`` to follow the TUH test recordings on their own;
* ``<name>=<path>`` -- an external, test-only cohort in the npy format
  (``dataset_maker/make_external_npy.py``, e.g. LEMON), scored zero-shot: its
  windows go through the run's own window selection and channel handling, and
  its targets are z-scored with the *training* cohort's stats so the run's
  metrics de-normalize them like the regular test split.

``$LABRAM_TEST_SETS=off`` turns them all off (the test suite does). A set
whose path does not exist, or whose family has no test recording, is skipped
with a warning, so a default list never breaks a run on a machine
without that data. The sets are evaluated wherever the test split is
(every epoch, and in eval-only runs) and logged as ``test_<name>``.
"""
import copy
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

from labram.data.dataset_groups import UNKNOWN, root_dataset_lookup
from labram.data.window_selection import _leaves

logger = logging.getLogger("labram")


def parse_test_sets(entries: Sequence[str]) -> List[Tuple[str, Optional[str]]]:
    """``["temple", "lemon=/data/..."]`` -> ``[("temple", None), ("lemon", "/data/...")]``."""
    out, seen = [], set()
    for entry in entries or []:
        name, sep, path = str(entry).partition("=")
        name = name.strip()
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"evaluation.test_sets entry {entry!r}: the name must be alphanumeric")
        if name == "test" or name in seen:
            raise ValueError(f"evaluation.test_sets: duplicate or reserved name {name!r}")
        seen.add(name)
        out.append((name, path.strip() if sep else None))
    return out


def family_subset(dataset, family: str):
    """Shallow copies of ``dataset``'s leaves keeping the windows of recordings
    from source dataset ``family`` (None when there are none)."""
    from labram.data.tuh_metadata import recording_stem
    parts = []
    for leaf in _leaves(dataset):
        files = getattr(leaf, "files", None)
        if not files:
            continue
        lookup = root_dataset_lookup(getattr(leaf, "root", "") or "")
        keep = [f for f in files if lookup.get(recording_stem(f), UNKNOWN) == family]
        if not keep:
            continue
        part = copy.copy(leaf)
        part.files = keep
        if hasattr(part, "_mmaps"):
            part._mmaps = OrderedDict()            # do not share open memmaps
        parts.append(part)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    import torch
    return torch.utils.data.ConcatDataset(parts)


def external_test(path: str, data_format: str, selection, target_stats,
                  drop_channels: Sequence[str] = ()):
    """The test split of a test-only cohort at ``path``, selected like the run's
    own test split, with targets z-scored by the run's ``target_stats``."""
    from labram.data import get_dataset_bundle
    from labram.data.window_selection import apply_window_selection
    bundle = get_dataset_bundle("TUAB_AGE", path, data_format=data_format)
    bundle = apply_window_selection(bundle, selection)
    if drop_channels:
        from labram.data.channel_selection import drop_channels as _drop
        _drop(bundle, list(drop_channels))
    if bundle.test is None:
        return None
    for leaf in _leaves(bundle.test):
        leaf.target_stats = target_stats
    return bundle.test


def build_test_sets(entries: Sequence[str], test_dataset, data_cfg, selection,
                    target_stats) -> "OrderedDict[str, object]":
    """``{name: dataset}`` for ``evaluation.test_sets`` (skipped sets left out)."""
    out: "OrderedDict[str, object]" = OrderedDict()
    for name, path in parse_test_sets(entries):
        if path is None:
            ds = family_subset(test_dataset, name) if test_dataset is not None else None
            if ds is None:
                logger.warning("test set %r: no %r recording in the test split; skipped", name, name)
                continue
        else:
            if not os.path.isdir(path):
                logger.warning("test set %r: %s does not exist; skipped", name, path)
                continue
            ds = external_test(path, data_cfg.data_format, selection, target_stats,
                               data_cfg.drop_channels)
            if ds is None or len(ds) == 0:
                logger.warning("test set %r: no test window in %s; skipped", name, path)
                continue
        logger.info("test set %r: %d sample(s)%s", name, len(ds), f" from {path}" if path else "")
        out[name] = ds
    return out
