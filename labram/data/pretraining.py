# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Pre-training dataset assembly from HDF5 file lists.
# ---------------------------------------------------------

from pathlib import Path
from typing import List, Tuple

from labram.data.hdf5_datasets import ShockDataset


def build_pretraining_dataset(
    datasets: List[List[str]],
    time_window: List[int],
    stride: int = 200,
    start_percentage: float = 0,
    end_percentage: float = 1,
) -> Tuple[List[ShockDataset], List[List[str]]]:
    shock_dataset_list = []
    ch_names_list = []
    for dataset_list, window_size in zip(datasets, time_window):
        dataset = ShockDataset(
            [Path(file_path) for file_path in dataset_list],
            window_size * 200, stride, start_percentage, end_percentage,
        )
        shock_dataset_list.append(dataset)
        ch_names_list.append(dataset.get_ch_names())
    return shock_dataset_list, ch_names_list
