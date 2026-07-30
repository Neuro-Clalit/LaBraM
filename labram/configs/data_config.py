from dataclasses import dataclass, field
from typing import List

from labram.configs.base_configs import ConfigBase
from labram.configs.defaults import (
    DEFAULT_DATA_SPLIT_JSON,
    DEFAULT_DATASET_END_PERCENTAGE,
    DEFAULT_DATASET_START_PERCENTAGE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PIN_MEM,
    DEFAULT_PRETRAIN_STRIDE,
    DEFAULT_TIME_WINDOWS,
)


@dataclass
class DataConfig(ConfigBase):
    """Pre-training and VQNSP dataset spec.

    ``datasets_train`` mirrors the nested-list layout
    ``build_pretraining_dataset`` expects: outer list groups files that
    share a channel montage.
    """
    dataset: str = ""
    data_path: str = ""
    robust_test: str = ""
    # Reuse a recorded data_split.json (local or s3://) instead of the dataset's
    # default split — pins the train/val/test case assignment across runs.
    split_json: str = DEFAULT_DATA_SPLIT_JSON
    num_workers: int = DEFAULT_NUM_WORKERS
    pin_mem: bool = DEFAULT_PIN_MEM
    datasets_train: List[List[str]] = field(default_factory=list)
    datasets_val: List[List[str]] = field(default_factory=list)
    time_window: List[int] = field(default_factory=lambda: list(DEFAULT_TIME_WINDOWS))
    val_time_window: List[int] = field(default_factory=list)
    stride: int = DEFAULT_PRETRAIN_STRIDE
    start_percentage: float = DEFAULT_DATASET_START_PERCENTAGE
    end_percentage: float = DEFAULT_DATASET_END_PERCENTAGE
