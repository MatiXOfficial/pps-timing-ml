import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

DATASET_ROOT_PATH = Path('data/dataset/dataset.pkl')


def load_dataset(pwd: Path, plane: int, channel: int) -> tuple[np.ndarray, np.ndarray]:
    with open(pwd / DATASET_ROOT_PATH, 'rb') as file:
        dataset = pickle.load(file)

    all_X, all_y = dataset[(plane, channel)][0], dataset[(plane, channel)][1]
    return all_X, all_y


def load_dataset_train_test(
        pwd: Path, plane: int, channel: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_X, all_y = load_dataset(pwd, plane, channel)
    x_train, x_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def load_dataset_train_val(
        pwd: Path, plane: int, channel: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_base_train, _, y_base_train, _ = load_dataset_train_test(pwd, plane, channel)
    x_train, x_val, y_train, y_val = train_test_split(x_base_train, y_base_train, test_size=0.2, random_state=42)
    return x_train, x_val, y_train, y_val
