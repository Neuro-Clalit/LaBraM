"""Tests for the TUAB age-regression dataset: split building and TUABAgeLoader.

Builds a synthetic ``processed/`` tree of tiny window pickles plus a metadata
sidecar, so nothing here needs the real 60 GB corpus.
"""
import json
import pickle

import numpy as np
import pytest
import torch

from labram.data.age_splits import (
    build_age_split,
    load_age_split,
    save_age_split,
)
from labram.data.bundles import DatasetBundle
from labram.data.cross_validation import group_id_for
from labram.data.tuh_datasets import TUABAgeLoader, prepare_TUAB_age_dataset
from labram.data.tuh_metadata import save_metadata_sidecar
from labram.data.tuh_metadata import RecordingMetadata

N_CHANNELS = 23
N_SAMPLES = 2000


def _metadata(stem, age):
    subject, session, token = stem.split("_")
    return RecordingMetadata(
        stem=stem, subject=subject, session=session, token=token,
        age=age, sex="F", year=2012, raw_age=age if age is not None else 999)


def _make_corpus(tmp_path, subjects_per_split=(3, 0, 2), windows=2, ages=None):
    """Create processed/{train,val,test} pickles + the age metadata sidecar.

    Returns ``(root, processed_dir, {stem: age})``. Subjects are named so that a
    subject never spans two directories, mirroring the real corpus.
    """
    processed = tmp_path / "processed"
    meta = {}
    expected = {}
    counter = 0
    for split, n_subjects in zip(("train", "val", "test"), subjects_per_split):
        directory = processed / split
        directory.mkdir(parents=True, exist_ok=True)
        for _ in range(n_subjects):
            subject = f"aaaaa{counter:03d}"
            stem = f"{subject}_s001_t000"
            age = (ages or {}).get(stem, 20 + counter * 5)
            meta[stem] = _metadata(stem, age)
            if age is not None:
                expected[stem] = float(age)
            for w in range(windows):
                with open(directory / f"{stem}_{w}.pkl", "wb") as fh:
                    pickle.dump(
                        {"X": np.random.randn(N_CHANNELS, N_SAMPLES), "y": 0}, fh)
            counter += 1
    save_metadata_sidecar(meta, str(processed / "age_metadata.json"))
    return tmp_path, processed, expected


# --------------------------------------------------------------- splits


def test_split_is_subject_disjoint_and_reproducible(tmp_path):
    _, processed, ages = _make_corpus(tmp_path, subjects_per_split=(8, 2, 3))

    first = build_age_split(str(processed), ages, seed=7)
    second = build_age_split(str(processed), ages, seed=7)

    assert first.files == second.files, "a fixed seed must give a fixed split"
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not first.subjects(left) & first.subjects(right)
    # Every pooled window is assigned exactly once.
    assert len(first.files["train"]) + len(first.files["val"]) == 10 * 2


def test_split_keeps_the_official_eval_set_as_test(tmp_path):
    _, processed, ages = _make_corpus(tmp_path, subjects_per_split=(6, 0, 2))
    split = build_age_split(str(processed), ages)

    assert len(split.subjects("test")) == 2
    assert all(f.startswith("test/") for f in split.files["test"])


def test_split_drops_windows_without_a_usable_age(tmp_path):
    _, processed, _ = _make_corpus(tmp_path, subjects_per_split=(4, 0, 1))
    # Withhold one subject's age, as the Age:999 redaction would.
    ages = {"aaaaa000_s001_t000": 33.0, "aaaaa001_s001_t000": 44.0,
            "aaaaa002_s001_t000": 55.0, "aaaaa004_s001_t000": 66.0}
    split = build_age_split(str(processed), ages)

    all_files = split.files["train"] + split.files["val"] + split.files["test"]
    assert not any("aaaaa003" in f for f in all_files)


def test_split_round_trips_through_json(tmp_path):
    _, processed, ages = _make_corpus(tmp_path, subjects_per_split=(6, 2, 2))
    split = build_age_split(str(processed), ages, seed=3)
    out = processed / "age_split.json"

    save_age_split(split, str(out))
    assert load_age_split(str(out)).files == split.files


def test_saving_a_leaking_split_is_refused(tmp_path):
    """The 16-subject train/val overlap in the shipped TUAB split is the bug this
    guards: a leak must raise, not pass silently onto disk."""
    _, processed, ages = _make_corpus(tmp_path, subjects_per_split=(4, 0, 1))
    split = build_age_split(str(processed), ages)
    split.files["val"].append(split.files["train"][0])

    out = processed / "bad.json"
    with pytest.raises(ValueError, match="appear in both train and val"):
        save_age_split(split, str(out))
    assert not out.exists(), "a leaking split must not be written"


def test_loading_a_leaking_split_is_refused(tmp_path):
    """Guards a hand-edited or legacy artifact, not just in-process mutation."""
    _, processed, ages = _make_corpus(tmp_path, subjects_per_split=(4, 0, 1))
    split = build_age_split(str(processed), ages)
    out = processed / "leaky.json"
    save_age_split(split, str(out))

    payload = json.loads(out.read_text())
    payload["files"]["val"].append(payload["files"]["train"][0])
    out.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="appear in both train and val"):
        load_age_split(str(out))


# --------------------------------------------------------------- loader


def test_loader_returns_float32_scalar_age(tmp_path):
    _, processed, ages = _make_corpus(tmp_path, subjects_per_split=(2, 0, 0))
    files = sorted(f.name for f in (processed / "train").glob("*.pkl"))

    loader = TUABAgeLoader(str(processed / "train"), files, age_lookup=ages)
    X, y = loader[0]

    assert X.shape == (N_CHANNELS, N_SAMPLES) and X.dtype == torch.float32
    assert isinstance(y, torch.Tensor) and y.dtype == torch.float32
    assert y.ndim == 0
    assert float(y) == ages[files[0].rsplit("_", 1)[0]]


def test_loader_collates_to_a_float32_batch(tmp_path):
    _, processed, ages = _make_corpus(tmp_path, subjects_per_split=(3, 0, 0))
    files = sorted(f.name for f in (processed / "train").glob("*.pkl"))
    loader = TUABAgeLoader(str(processed / "train"), files, age_lookup=ages)

    xb, yb = next(iter(torch.utils.data.DataLoader(loader, batch_size=4)))
    assert yb.dtype == torch.float32 and yb.shape == (4,)
    assert xb.dtype == torch.float32


def test_loader_normalizes_and_metrics_can_invert_it(tmp_path):
    _, processed, ages = _make_corpus(tmp_path, subjects_per_split=(2, 0, 0))
    files = sorted(f.name for f in (processed / "train").glob("*.pkl"))
    stats = (50.0, 10.0)

    plain = TUABAgeLoader(str(processed / "train"), files, age_lookup=ages)
    scaled = TUABAgeLoader(str(processed / "train"), files, age_lookup=ages,
                           target_stats=stats)

    raw = float(plain[0][1])
    assert float(scaled[0][1]) == pytest.approx((raw - stats[0]) / stats[1])
    # De-normalizing recovers the age in years.
    from labram.utils import denormalize
    assert float(denormalize(np.array([float(scaled[0][1])]), stats)[0]) == pytest.approx(raw)


def test_loader_filters_unlabelled_windows_at_construction(tmp_path):
    """TUHLoader.__getitem__ swallows KeyError and substitutes another window, so
    an unresolvable age must be removed up front rather than silently mislabelled."""
    _, processed, ages = _make_corpus(tmp_path, subjects_per_split=(3, 0, 0))
    files = sorted(f.name for f in (processed / "train").glob("*.pkl"))
    partial = {k: v for k, v in ages.items() if not k.startswith("aaaaa001")}

    loader = TUABAgeLoader(str(processed / "train"), files, age_lookup=partial)

    assert len(loader) < len(files)
    assert not any("aaaaa001" in f for f in loader.files)
    # Every remaining window resolves, so no substitution can occur.
    assert all(isinstance(loader[i][1], torch.Tensor) for i in range(len(loader)))


def test_loader_resolves_its_ages_from_root_for_cv_reconstruction(tmp_path):
    """cross_validation._build_split_dataset rebuilds loaders as
    ``type(src)(root, files, sampling_rate)`` -- three positional args, no
    keywords. The lookup therefore has to be recoverable from the root alone."""
    _, processed, ages = _make_corpus(tmp_path, subjects_per_split=(2, 0, 0))
    files = sorted(f.name for f in (processed / "train").glob("*.pkl"))
    src = TUABAgeLoader(str(processed / "train"), files, age_lookup=ages)

    rebuilt = type(src)(src.root, sorted(src.files), getattr(src, "sampling_rate", 200))

    assert len(rebuilt) == len(src)
    assert float(rebuilt[0][1]) == float(src[0][1])


def test_cv_fold_rebuild_preserves_target_normalization(tmp_path):
    """Regression test: ``_build_split_dataset`` rebuilds loaders positionally and
    so cannot pass ``target_stats``. When it was dropped, folds emitted raw ages
    that the eval path then de-normalized a second time, inflating MAE ~18x."""
    from labram.configs.cv_config import CrossValidationConfig
    from labram.data.cross_validation import build_grouped_folds, materialize_fold

    root, _, _ = _make_corpus(tmp_path, subjects_per_split=(9, 3, 3))
    from labram.data import get_dataset_bundle
    bundle = get_dataset_bundle("TUAB_AGE", str(root))
    assert bundle.target_stats is not None

    cfg = CrossValidationConfig(enabled=True, n_folds=3, split_by="subject", seed=1)
    folds = build_grouped_folds(bundle, cfg, "TUAB_AGE")
    fold = materialize_fold(folds, 0, bundle)

    assert fold.target_stats == bundle.target_stats
    for split in (fold.train, fold.val, fold.test):
        # A fold is a ConcatDataset when its groups span several source loaders.
        parts = (list(split.datasets)
                 if isinstance(split, torch.utils.data.ConcatDataset) else [split])
        for part in parts:
            assert part.target_stats == bundle.target_stats, (
                "a fold loader lost its normalization stats")
        # A normalized target is O(1); a raw age would be O(10).
        assert abs(float(split[0][1])) < 10.0


def test_loader_group_ids_survive_a_split_prefixed_path(tmp_path):
    """A pooled split stores files as '<subdir>/<name>.pkl' so it stays one
    loader; case ids and CV grouping must still see through to the basename."""
    _, processed, ages = _make_corpus(tmp_path, subjects_per_split=(2, 0, 0))
    files = ["train/" + f.name for f in sorted((processed / "train").glob("*.pkl"))]

    loader = TUABAgeLoader(str(processed), files, age_lookup=ages)
    loader.return_id = True

    _, _, case_id = loader[0]
    assert case_id == files[0].split("/")[-1].rsplit("_", 1)[0]
    assert group_id_for(loader, files[0], "subject") == "aaaaa000"


# --------------------------------------------------------------- bundle


def test_prepare_builds_normalized_splits_and_train_stats(tmp_path):
    root, processed, ages = _make_corpus(tmp_path, subjects_per_split=(8, 2, 3))
    train, test, val, stats = prepare_TUAB_age_dataset(str(root))

    assert stats is not None
    mean, std = stats
    train_ages = train.ages()
    assert mean == pytest.approx(float(np.mean(train_ages)))
    assert std > 0
    # Stats come from train only -- deriving them from val/test would leak.
    assert all(len(ds) > 0 for ds in (train, val, test))
    assert not (set(train.files) & set(val.files) & set(test.files))


def test_prepare_can_skip_normalization(tmp_path):
    root, _, _ = _make_corpus(tmp_path, subjects_per_split=(6, 2, 2))
    train, _, _, stats = prepare_TUAB_age_dataset(str(root), normalize_targets=False)

    assert stats is None
    assert float(train[0][1]) == train.ages()[0], "target must be the raw age"


def test_bundle_marks_the_task_as_regression(tmp_path):
    from labram.data import get_dataset_bundle
    root, _, _ = _make_corpus(tmp_path, subjects_per_split=(8, 2, 3))

    bundle = get_dataset_bundle("TUAB_AGE", str(root))

    assert isinstance(bundle, DatasetBundle)
    assert bundle.is_regression and bundle.task == "regression"
    assert bundle.nb_classes == 1, "a scalar head, same width as a binary head"
    assert "mae" in bundle.metrics
    assert bundle.target_stats is not None


def test_classification_bundle_still_defaults_to_classification():
    bundle = DatasetBundle(train=None, val=None, test=None, ch_names=[],
                           nb_classes=1, metrics=["accuracy"])
    assert bundle.task == "classification"
    assert not bundle.is_regression
    assert bundle.target_stats is None
