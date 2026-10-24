#!/usr/bin/env python
"""Check that the age-target normalization never sees test ages.

A brain-age run z-scores its target with a (mean, std) of ages. If those
statistics included test recordings, the test labels would leak into training.
For each setup this script builds the data exactly as ``run_finetune.main``
does (dataset bundle -> CV fold / reused split -> window selection) twice:

* as recorded, and
* with every **test** recording's age shifted by ``--shift`` years
  (``load_age_lookup_for`` patched),

and reports the (mean, std) the run would train with in both cases, the
(mean, std) of the selected training windows' ages, and whether the shift
reached the test targets (a sanity check that the patch took effect). The
normalization is test-free when the two (mean, std) pairs are identical and
equal the training windows' own statistics.

    python scripts/check_target_norm_leak.py --out docs/data/target_norm_leak_check.csv
"""
import argparse
import os
import sys
from contextlib import contextmanager
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labram.configs.run_configs import FinetuneRunConfig  # noqa: E402
from labram.data import get_dataset_bundle  # noqa: E402
from labram.data import cross_validation as cv  # noqa: E402
from labram.data import tuh_datasets  # noqa: E402
from labram.data.window_selection import (  # noqa: E402
    WindowSelection, _leaves, apply_window_selection, train_window_ages)

TUH = os.environ.get("LABRAM_TUH_ROOT", "/data/datasets/EEG-public/TUH_EEG")
CONFIG = "labram/configs/defaults/finetune_tuab_age.json"
# (name, data path, CV settings or None); the window selection is the age config's (D's).
SETUPS = (
    ("phase 1: TUAB (D recipe)", f"{TUH}/tuab/v3.0.1/edf", None),
    ("phase 2: cohort all_both_v3 (Exp4*)", f"{TUH}/cohorts/all_both_v3", None),
    ("phase 3: CV fold 0, test = fold (cvbase/cvctrl/cvaug)", f"{TUH}/cohorts/all_both_v3_cvcompact", "fold"),
    ("phase 3: CV fold 0, test = TUAB held out (cvaugval)", f"{TUH}/cohorts/all_both_v3_cvcompact", "held_out"),
)


def _stems(dataset):
    return {tuh_datasets.recording_stem(f) for leaf in _leaves(dataset) for f in leaf.files}


def _test_targets(dataset):
    return np.array([a for leaf in _leaves(dataset) if hasattr(leaf, "ages") for a in leaf.ages()])


def build(cfg, data_path, test_set):
    bundle = get_dataset_bundle(cfg.data.dataset, data_path, data_format="npy")
    if test_set is not None:
        c = cfg.cross_validation
        c.enabled, c.n_folds, c.fold, c.split_by, c.pool, c.test_set = True, 5, 0, "subject", "train_val", test_set
        folds = cv.build_grouped_folds(bundle, c, cfg.data.dataset)
        bundle = cv.materialize_fold(folds, 0, bundle)
    return apply_window_selection(bundle, WindowSelection.from_data_config(cfg.data))


@contextmanager
def shifted_test_ages(test_stems, shift):
    original = tuh_datasets.load_age_lookup_for

    def patched(path):
        ages = dict(original(path))
        for stem in test_stems:
            if stem in ages:
                ages[stem] = ages[stem] + shift
        return ages

    # Age loaders cache the lookup per data root (_resolve_lookup); clear it so
    # the loaders that re-read it (CV fold loaders) see the shifted ages too.
    caches = [c._lookup_cache for c in vars(tuh_datasets).values()
              if isinstance(c, type) and isinstance(getattr(c, "_lookup_cache", None), dict)]
    for c in caches:
        c.clear()
    try:
        with mock.patch.object(tuh_datasets, "load_age_lookup_for", patched):
            yield
    finally:
        for c in caches:
            c.clear()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shift", type=float, default=40.0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    cfg = FinetuneRunConfig.load_config(CONFIG)
    rows = []
    for name, path, test_set in SETUPS:
        if not os.path.isdir(path):
            print(f"skip {name}: {path} not found")
            continue
        base = build(FinetuneRunConfig.load_config(CONFIG), path, test_set)
        test_stems = _stems(base.test)
        with shifted_test_ages(test_stems, args.shift):
            moved = build(FinetuneRunConfig.load_config(CONFIG), path, test_set)
        ages = np.asarray(train_window_ages(base.train), float)
        own = (float(ages.mean()), float(ages.std(ddof=1)))
        dt = _test_targets(moved.test) - _test_targets(base.test)
        rows.append({
            "setup": name,
            "train recordings": len(_stems(base.train)), "test recordings": len(test_stems),
            "train ∩ test recordings": len(_stems(base.train) & test_stems),
            "mean": base.target_stats[0], "std": base.target_stats[1],
            "mean, test ages +shift": moved.target_stats[0], "std, test ages +shift": moved.target_stats[1],
            "= train windows' own stats": bool(np.allclose(base.target_stats, own, atol=1e-9)),
            "test targets moved by": float(np.median(dt)) if len(dt) else float("nan"),
            "normalization uses test ages": "no" if np.allclose(base.target_stats, moved.target_stats,
                                                               atol=1e-12) else "YES",
        })
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    if args.out:
        d.to_csv(args.out, index=False)
    print(d.to_string(index=False))
    return 0 if (d["normalization uses test ages"] == "no").all() else 1


if __name__ == "__main__":
    sys.exit(main())
