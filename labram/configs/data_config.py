from dataclasses import dataclass, field
from typing import List, Union

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
    # --- Recording/window selection (TUAB-family fine-tuning; see
    # labram/data/window_selection.py). The defaults keep every window. ---
    # Which TUAB cases to train *and* evaluate on: "all", "normal" or "abnormal".
    case_filter: str = "all"
    # Seconds dropped from the start / end of every recording (rounded up to
    # whole 10 s windows), e.g. 60 to skip set-up and wind-down artifacts.
    trim_start_sec: Union[int, float] = 0.0
    trim_end_sec: Union[int, float] = 0.0
    # Model input length in seconds: consecutive 10 s windows are concatenated,
    # so this must be a multiple of 10 in [10, 60].
    window_sec: int = 10
    # Evaluate (val/test) on only the first N minutes of each recording, after
    # trimming. 0 keeps the whole recording. Training always uses everything.
    eval_minutes: Union[int, float] = 0.0
    # Storage format of the TUAB windows: "pickle" (one file per 10 s window,
    # processed/) or "npy" (one float32 file per recording, processed_npy/,
    # built by dataset_maker/make_TUAB_npy.py). Same samples either way.
    data_format: str = "pickle"
    # npy only: move each training sample to a random start within half a
    # sample length of its grid position (inside the trimmed range), redrawn
    # every epoch. Evaluation always uses the fixed grid.
    random_crop: bool = False
