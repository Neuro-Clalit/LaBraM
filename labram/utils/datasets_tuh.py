# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# TUH-EEG (TUAB / TUEV) torch.utils.data.Dataset wrappers and split assembly.
# ---------------------------------------------------------

import os
import pickle
from typing import Callable

import numpy as np
import torch
import torch.utils.data
from scipy.signal import resample


class TUHLoader(torch.utils.data.Dataset):
    """Parameterised loader for TUH-EEG pickle datasets (TUAB, TUEV, etc.).

    Args:
        root: Directory containing the pickle files.
        files: List of file names within *root*.
        sampling_rate: Target sampling rate; data is resampled if it differs from
            the default 200 Hz recorded rate.
        signal_key: Key used to read the EEG array from each pickle dict.
        duration_sec: Recording duration in seconds at the default rate (used to
            compute the resample target length).
        label_fn: Callable that maps a pickle dict to the integer label.
    """

    def __init__(
        self,
        root: str,
        files: list,
        sampling_rate: int = 200,
        *,
        signal_key: str,
        duration_sec: int,
        label_fn: Callable,
    ):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self._signal_key = signal_key
        self._duration_sec = duration_sec
        self._label_fn = label_fn

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample[self._signal_key]
        if self.sampling_rate != self.default_rate:
            X = resample(X, self._duration_sec * self.sampling_rate, axis=-1)
        Y = self._label_fn(sample)
        return torch.FloatTensor(X), Y


class TUABLoader(TUHLoader):
    """Loader for the TUH Abnormal EEG Corpus (TUAB) pickle files."""

    def __init__(self, root: str, files: list, sampling_rate: int = 200):
        super().__init__(
            root, files, sampling_rate,
            signal_key="X",
            duration_sec=10,
            label_fn=lambda s: s["y"],
        )


class TUEVLoader(TUHLoader):
    """Loader for the TUH EEG Events Corpus (TUEV) pickle files."""

    def __init__(self, root: str, files: list, sampling_rate: int = 200):
        super().__init__(
            root, files, sampling_rate,
            signal_key="signal",
            duration_sec=5,
            label_fn=lambda s: int(s["label"][0] - 1),
        )


def prepare_TUEV_dataset(root):
    seed = 4523
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files = os.listdir(os.path.join(root, "processed_eval"))
    test_files = os.listdir(os.path.join(root, "processed_test"))

    train_dataset = TUEVLoader(os.path.join(root, "processed_train"), train_files)
    test_dataset = TUEVLoader(os.path.join(root, "processed_test"), test_files)
    val_dataset = TUEVLoader(os.path.join(root, "processed_eval"), val_files)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_TUAB_dataset(root):
    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    train_dataset = TUABLoader(os.path.join(root, "train"), train_files)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset
