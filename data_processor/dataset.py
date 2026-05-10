import h5py
import bisect
from pathlib import Path
from typing import List
from torch.utils.data import Dataset


list_path = List[Path]

class SingleShockDataset(Dataset):
    """Read single hdf5 file regardless of label, subject, and paradigm."""
    def __init__(self, file_path: Path, window_size: int=200, stride: int=1, start_percentage: float=0, end_percentage: float=1):
        '''
        Extract datasets from file_path.

        param Path file_path: the path of target data
        param int window_size: the length of a single sample
        param int stride: the step between two adjacent samples
        param float start_percentage: Index of percentage of the first sample of the dataset in the data file (inclusive)
        param float end_percentage: Index of percentage of end of dataset sample in data file (not included)
        '''
        self._file_path = file_path
        self._window_size = window_size
        self._stride = stride
        self._start_percentage = start_percentage
        self._end_percentage = end_percentage

        self._file = None
        self._length = None
        self._feature_size = None

        self._subjects = []
        self._global_idxes = []
        self._local_idxes = []

        self._init_dataset()

    def _init_dataset(self) -> None:
        self._file = h5py.File(str(self._file_path), 'r')
        self._subjects = [i for i in self._file]

        global_idx = 0
        for subject in self._subjects:
            self._global_idxes.append(global_idx)  # the start index of the subject's sample in the dataset
            subject_len = self._file[subject]['eeg'].shape[1]
            # total number of samples
            total_sample_num = (subject_len - self._window_size) // self._stride + 1
            # cut out part of samples
            start_idx = int(total_sample_num * self._start_percentage) * self._stride
            end_idx = int(total_sample_num * self._end_percentage - 1) * self._stride

            self._local_idxes.append(start_idx)
            global_idx += (end_idx - start_idx) // self._stride + 1
        self._length = global_idx

        self._feature_size = [i for i in self._file[self._subjects[0]]['eeg'].shape]
        self._feature_size[1] = self._window_size

    @property
    def feature_size(self):
        return self._feature_size

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int):
        subject_idx = bisect.bisect(self._global_idxes, idx) - 1
        item_start_idx = (idx - self._global_idxes[subject_idx]) * self._stride + self._local_idxes[subject_idx]
        return self._file[self._subjects[subject_idx]]['eeg'][:, item_start_idx:item_start_idx + self._window_size]

    def free(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def get_ch_names(self):
        return self._file[self._subjects[0]]['eeg'].attrs['chOrder']


class ShockDataset(Dataset):
    """integrate multiple hdf5 files"""
    def __init__(self, file_paths: list_path, window_size: int=200, stride: int=1, start_percentage: float=0, end_percentage: float=1):
        '''
        Arguments will be passed to SingleShockDataset. Refer to SingleShockDataset.
        '''
        self._file_paths = file_paths
        self._window_size = window_size
        self._stride = stride
        self._start_percentage = start_percentage
        self._end_percentage = end_percentage

        self._datasets = []
        self._length = None
        self._feature_size = None

        self._dataset_idxes = []

        self._init_dataset()

    def _init_dataset(self) -> None:
        self._datasets = [SingleShockDataset(file_path, self._window_size, self._stride, self._start_percentage, self._end_percentage) for file_path in self._file_paths]

        # calculate the number of samples for each subdataset to form the integral indexes
        dataset_idx = 0
        for dataset in self._datasets:
            self._dataset_idxes.append(dataset_idx)
            dataset_idx += len(dataset)
        self._length = dataset_idx

        self._feature_size = self._datasets[0].feature_size

    @property
    def feature_size(self):
        return self._feature_size

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int):
        dataset_idx = bisect.bisect(self._dataset_idxes, idx) - 1
        item_idx = (idx - self._dataset_idxes[dataset_idx])
        return self._datasets[dataset_idx][item_idx]

    def free(self) -> None:
        for dataset in self._datasets:
            dataset.free()

    def get_ch_names(self):
        return self._datasets[0].get_ch_names()
