# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Per-task dataset bundles used by the fine-tuning runner.
# ---------------------------------------------------------

from dataclasses import dataclass
from typing import List

import torch.utils.data

from labram.data.eeg_constants import TUH_EEG_CH_NAMES, normalize_ch_names
from labram.data.tuh_datasets import prepare_TUAB_dataset, prepare_TUEV_dataset


@dataclass
class DatasetBundle:
    train: torch.utils.data.Dataset
    val: torch.utils.data.Dataset
    test: torch.utils.data.Dataset
    ch_names: List[str]
    nb_classes: int
    metrics: List[str]


def get_dataset_bundle(dataset_name: str, data_path: str) -> DatasetBundle:
    if dataset_name == 'TUAB':
        root = data_path or "path/to/TUAB"
        train, test, val = prepare_TUAB_dataset(root)
        return DatasetBundle(
            train=train, val=val, test=test,
            ch_names=normalize_ch_names(TUH_EEG_CH_NAMES),
            nb_classes=1,
            metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
        )
    if dataset_name == 'TUEV':
        root = data_path or "path/to/TUEV"
        train, test, val = prepare_TUEV_dataset(root)
        return DatasetBundle(
            train=train, val=val, test=test,
            ch_names=normalize_ch_names(TUH_EEG_CH_NAMES),
            nb_classes=6,
            metrics=["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
        )
    raise ValueError(f"Unknown dataset: {dataset_name!r} (expected 'TUAB' or 'TUEV')")
