import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import train_test_split

TIME_STEP = 1. / 7.695
X_TIME = np.arange(0, 24) * TIME_STEP
X_TIME_MAX = X_TIME[-1]

DATASET_ROOT_PATH = Path('data/dataset/dataset.pkl')
EXPANDED_DATASET_ROOT_PATH = Path('data/dataset/dataset_exp.pkl')

PlaneChannel = tuple[int, int]


def load_dataset(pwd: Path, plane: int, channel: int) -> tuple[np.ndarray, np.ndarray]:
    with open(pwd / DATASET_ROOT_PATH, 'rb') as file:
        dataset = pickle.load(file)

    all_X, all_y = dataset[(plane, channel)][0], dataset[(plane, channel)][1]
    return all_X, all_y


def load_dataset_train_test(
        pwd: Path, plane: int, channel: int, test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_X, all_y = load_dataset(pwd, plane, channel)
    x_train, x_test, y_train, y_test = train_test_split(all_X, all_y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


def load_dataset_train_val(
        pwd: Path, plane: int, channel: int, test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_base_train, _, y_base_train, _ = load_dataset_train_test(pwd, plane, channel)
    x_train, x_val, y_train, y_val = train_test_split(x_base_train, y_base_train, test_size=test_size,
                                                      random_state=random_state)
    return x_train, x_val, y_train, y_val


def load_dataset_all_channels(pwd: Path) -> tuple[dict[tuple, np.ndarray], dict[tuple, np.ndarray]]:
    with open(pwd / DATASET_ROOT_PATH, 'rb') as file:
        dataset = pickle.load(file)

    all_X, all_y = {}, {}
    for (plane, channel), (x, y) in dataset.items():
        all_X[(plane, channel)] = x
        all_y[(plane, channel)] = y
    return all_X, all_y


def load_dataset_train_test_all_channels(
        pwd: Path, test_size: float = 0.2, random_state: int = 42
) -> tuple[dict[tuple, np.ndarray], dict[tuple, np.ndarray], dict[tuple, np.ndarray], dict[tuple, np.ndarray]]:
    all_X, all_y = load_dataset_all_channels(pwd)

    x_train, x_test, y_train, y_test = {}, {}, {}, {}
    for (plane, channel), x in all_X.items():
        y = all_y[(plane, channel)]

        x_train_ch, x_test_ch, y_train_ch, y_test_ch = train_test_split(x, y, test_size=test_size,
                                                                        random_state=random_state)
        x_train[(plane, channel)] = x_train_ch
        x_test[(plane, channel)] = x_test_ch
        y_train[(plane, channel)] = y_train_ch
        y_test[(plane, channel)] = y_test_ch

    return x_train, x_test, y_train, y_test


def load_dataset_train_val_all_channels(
        pwd: Path, test_size: float = 0.2, random_state: int = 42
) -> tuple[dict[tuple, np.ndarray], dict[tuple, np.ndarray], dict[tuple, np.ndarray], dict[tuple, np.ndarray]]:
    all_X, _, all_y, _ = load_dataset_train_test_all_channels(pwd)

    x_train, x_val, y_train, y_val = {}, {}, {}, {}
    for (plane, channel), x in all_X.items():
        y = all_y[(plane, channel)]

        x_train_ch, x_val_ch, y_train_ch, y_val_ch = train_test_split(x, y, test_size=test_size,
                                                                      random_state=random_state)
        x_train[(plane, channel)] = x_train_ch
        x_val[(plane, channel)] = x_val_ch
        y_train[(plane, channel)] = y_train_ch
        y_val[(plane, channel)] = y_val_ch

    return x_train, x_val, y_train, y_val


@dataclass
class ExpandedDataset:
    t_avg: np.ndarray
    wav: dict[PlaneChannel, np.ndarray]
    t0: dict[PlaneChannel, np.ndarray]
    t_pred: dict[PlaneChannel, np.ndarray]
    _t_ref: dict[PlaneChannel, np.ndarray] | None = None
    _notnan_mask: dict[PlaneChannel, np.ndarray] | None = None

    @property
    def t_ref(self) -> dict[PlaneChannel, np.ndarray]:
        if self._t_ref is None:
            self._t_ref = self._compute_t_ref()
        return self._t_ref

    @property
    def notnan_mask(self) -> dict[PlaneChannel, np.ndarray]:
        if self._notnan_mask is None:
            self._notnan_mask = self._compute_notnan_mask()
        return self._notnan_mask

    def keys(self) -> Iterable[PlaneChannel]:
        return self.wav.keys()

    def extract_by_idx(self, idx: np.ndarray) -> 'ExpandedDataset':
        new_t_avg = self.t_avg[idx]
        new_wav, new_t0, new_t_pred = {}, {}, {}
        new_t_ref = {} if self._t_ref is not None else None
        new_notnan_mask = {} if self._notnan_mask is not None else None

        for key in self.keys():
            new_wav[key] = self.wav[key][idx]
            new_t0[key] = self.t0[key][idx]
            new_t_pred[key] = self.t_pred[key][idx]
            if self._t_ref is not None:
                new_t_ref[key] = self._t_ref[key][idx]
            if self._notnan_mask is not None:
                new_notnan_mask[key] = self._notnan_mask[key][idx]

        return ExpandedDataset(t_avg=new_t_avg, wav=new_wav, t0=new_t0, t_pred=new_t_pred, _t_ref=new_t_ref,
                               _notnan_mask=new_notnan_mask)

    def _compute_t_ref(self) -> dict[PlaneChannel, np.ndarray]:
        true_dataset_temp: dict[PlaneChannel, list] = defaultdict(list)
        for key in self.keys():
            for i, t_avg in enumerate(self.t_avg):
                if np.isnan(self.t0[key][i]):
                    true_dataset_temp[key].append(np.nan)
                else:
                    t_0 = self.t0[key][i]
                    t_pred = self.t_pred[key][i]
                    true_dataset_temp[key].append(self._compute_true_t(t_avg, t_pred, t_0))

        true_dataset = {}
        for key in self.keys():
            true_dataset[key] = np.array(true_dataset_temp[key])

        return dict(true_dataset)

    @staticmethod
    def _compute_true_t(t_avg: float, t_ch: float, t_0: float) -> float:
        # use only timestamps from other channels
        return ((3 * t_avg) - (t_ch + t_0)) / 2 - t_0

    def _compute_notnan_mask(self) -> dict[PlaneChannel, np.ndarray]:
        return {key: ~np.isnan(t0_array) for key, t0_array in self.t0.items()}

    def __len__(self):
        return len(self.t_avg)


def load_expanded_dataset(pwd: Path) -> ExpandedDataset:
    with open(pwd / EXPANDED_DATASET_ROOT_PATH, 'rb') as file:
        dataset = pickle.load(file)

    return dataset


def load_expanded_dataset_train_test(
        pwd: Path, test_size: float = 0.2, random_state: int = 42
) -> tuple[ExpandedDataset, ExpandedDataset]:
    dataset = load_expanded_dataset(pwd)
    train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=test_size, random_state=random_state)

    train_dataset = dataset.extract_by_idx(train_idx)
    test_dataset = dataset.extract_by_idx(test_idx)
    return train_dataset, test_dataset


def load_expanded_dataset_train_val(
        pwd: Path, test_size: float = 0.2, random_state: int = 42
) -> tuple[ExpandedDataset, ExpandedDataset]:
    dataset, _ = load_expanded_dataset_train_test(pwd, test_size, random_state)
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=test_size, random_state=random_state)

    train_dataset = dataset.extract_by_idx(train_idx)
    val_dataset = dataset.extract_by_idx(val_idx)
    return train_dataset, val_dataset
