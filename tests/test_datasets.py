"""Tests for data_processor.dataset.{SingleShockDataset, ShockDataset}.

Builds tiny synthetic HDF5 files in tmp_path (the pytest fixture) that
match the layout the dataset expects:
    /<subject>/eeg                -> (n_channels, n_samples)
    /<subject>/eeg.attrs[chOrder] -> list of channel name strings
"""
import h5py
import numpy as np
import pytest

from data_processor.dataset import SingleShockDataset, ShockDataset


N_CHANNELS = 4
N_SAMPLES = 5000
CH_ORDER = ['C3', 'C4', 'O1', 'O2']


def _make_hdf5(path, subjects=("subj_0",), n_channels=N_CHANNELS, n_samples=N_SAMPLES):
    with h5py.File(path, "w") as f:
        for s in subjects:
            grp = f.create_group(s)
            eeg = grp.create_dataset(
                "eeg", data=np.random.randn(n_channels, n_samples).astype("f4"),
            )
            eeg.attrs["chOrder"] = CH_ORDER[:n_channels]
    return path


@pytest.fixture
def synthetic_hdf5(tmp_path):
    return _make_hdf5(tmp_path / "subj.h5")


@pytest.fixture
def two_synthetic_hdf5(tmp_path):
    return [
        _make_hdf5(tmp_path / "subj_a.h5"),
        _make_hdf5(tmp_path / "subj_b.h5"),
    ]


class TestSingleShockDataset:
    def test_constructs_and_reports_length(self, synthetic_hdf5):
        ds = SingleShockDataset(synthetic_hdf5, window_size=200, stride=100)
        # 5000-sample signal, 200-window, 100-stride -> floor((5000-200)/100)+1 = 49
        assert len(ds) == 49

    def test_feature_size_uses_window(self, synthetic_hdf5):
        ds = SingleShockDataset(synthetic_hdf5, window_size=200, stride=100)
        assert ds.feature_size == [N_CHANNELS, 200]

    def test_getitem_first_and_last(self, synthetic_hdf5):
        ds = SingleShockDataset(synthetic_hdf5, window_size=200, stride=100)
        first = ds[0]
        last = ds[len(ds) - 1]
        assert first.shape == (N_CHANNELS, 200)
        assert last.shape == (N_CHANNELS, 200)

    def test_get_ch_names(self, synthetic_hdf5):
        ds = SingleShockDataset(synthetic_hdf5, window_size=200, stride=100)
        names = list(ds.get_ch_names())
        assert names == CH_ORDER

    def test_percentage_subsetting(self, synthetic_hdf5):
        full = SingleShockDataset(synthetic_hdf5, window_size=200, stride=100)
        first_half = SingleShockDataset(
            synthetic_hdf5, window_size=200, stride=100,
            start_percentage=0.0, end_percentage=0.5,
        )
        # First half should have roughly half as many samples (within 1)
        assert abs(len(first_half) - len(full) // 2) <= 1

    def test_free_closes_handle(self, synthetic_hdf5):
        ds = SingleShockDataset(synthetic_hdf5, window_size=200, stride=100)
        assert ds._file is not None
        ds.free()
        assert ds._file is None

    def test_stride_one_yields_max_samples(self, synthetic_hdf5):
        ds = SingleShockDataset(synthetic_hdf5, window_size=200, stride=1)
        # 5000 - 200 + 1 = 4801
        assert len(ds) == 4801


class TestShockDataset:
    def test_wraps_two_files_with_summed_length(self, two_synthetic_hdf5):
        single = SingleShockDataset(two_synthetic_hdf5[0], window_size=200, stride=100)
        multi = ShockDataset(two_synthetic_hdf5, window_size=200, stride=100)
        assert len(multi) == 2 * len(single)

    def test_indexing_crosses_file_boundary(self, two_synthetic_hdf5):
        single = SingleShockDataset(two_synthetic_hdf5[0], window_size=200, stride=100)
        n_per_file = len(single)
        multi = ShockDataset(two_synthetic_hdf5, window_size=200, stride=100)
        # Items just before and after the split should both work
        assert multi[n_per_file - 1].shape == (N_CHANNELS, 200)
        assert multi[n_per_file].shape == (N_CHANNELS, 200)

    def test_get_ch_names_delegates_to_first(self, two_synthetic_hdf5):
        multi = ShockDataset(two_synthetic_hdf5, window_size=200, stride=100)
        assert list(multi.get_ch_names()) == CH_ORDER

    def test_free_closes_all(self, two_synthetic_hdf5):
        multi = ShockDataset(two_synthetic_hdf5, window_size=200, stride=100)
        multi.free()
        for ds in multi._datasets:
            assert ds._file is None

    def test_feature_size_matches_first_dataset(self, two_synthetic_hdf5):
        multi = ShockDataset(two_synthetic_hdf5, window_size=200, stride=100)
        assert multi.feature_size == [N_CHANNELS, 200]
