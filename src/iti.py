import copy
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.python.framework.errors_impl import NotFoundError

from src.dataset import ExpandedDataset, X_TIME, TIME_STEP, PlaneChannel
from src.gauss_hist import get_gauss_stats, plot_gauss_hist
from src.models import optimal_model_builder_iti as model_builder
from src.network_utils import gaussian_kernel, plot_difference_hist
from src.network_utils import train_model as _base_train_model
from src.utils import save_plt


def build_and_train_network(
        iteration: int, x_train: np.ndarray, x_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, overwrite: bool,
        pwd: str, dir_name: str, lr: float, n_epochs: int, batch_size: int, lr_patience: int, es_patience: int,
        es_min_delta: float, loss_weight: int, verbose: int = 2
) -> tuple[keras.Model, pd.DataFrame]:
    model = model_builder()
    name = f"optimal_it_{iteration}"
    if overwrite:
        history = _base_train_model(model, name, dir_name, x_train, y_train, x_val, y_val, lr, True, n_epochs,
                                    verbose, batch_size, lr_patience, es_patience, es_min_delta, loss_weight,
                                    root=pwd + '/data')
    else:
        try:
            history = _base_train_model(model, name, dir_name, x_train, y_train, x_val, y_val, lr, False, n_epochs,
                                        verbose, batch_size, lr_patience, es_patience, es_min_delta, loss_weight,
                                        root=pwd + '/data')
        except (NotFoundError, FileNotFoundError):
            history = _base_train_model(model, name, dir_name, x_train, y_train, x_val, y_val, lr, True, n_epochs,
                                        verbose, batch_size, lr_patience, es_patience, es_min_delta, loss_weight,
                                        root=pwd + '/data')

    return model, history


def build_nn_dataset(dataset: ExpandedDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = [], []
    for key in dataset.keys():
        x.extend(dataset.wav[key][dataset.notnan_mask[key]])
        y.extend(dataset.t_ref[key][dataset.notnan_mask[key]])

    x, y = np.array(x), np.array(y)

    # UNet
    y_network = np.array([gaussian_kernel(t) for t in y])

    return x, y, y_network


def _pred_model(model: keras.Model, x: np.ndarray, batch_size: int) -> np.ndarray:
    y_pred = model.predict(x, batch_size=batch_size)

    # UNet
    y_pred_t = np.empty(y_pred.shape[0])
    for i, y in enumerate(y_pred):
        _, y_pred_t[i], _, _ = get_gauss_stats(X_TIME, y, a_0=1, std_0=1. * TIME_STEP)

    return y_pred_t


def _plot_and_save_global_t_pred_hists(t_global_dict: dict, path: Path | str) -> None:
    plt.figure(figsize=(16, 6.5))
    for i, ((plane, channel), t_global) in enumerate(t_global_dict.items()):
        t_avg = np.mean(t_global)
        plt.subplot(2, 4, i + 1)
        plt.hist(t_global, bins=50, range=(t_avg - 1, t_avg + 1))
        plt.axvline(t_avg, c='red')
        plt.title(f'({plane}, {channel}); $t_{{pred}}$ + $t_0$ mean: {t_avg:0.2f}')
        plt.xlabel('t [ns]')

    plt.tight_layout()
    save_plt(path)
    plt.close()


def build_updated_dataset(
        model: keras.Model, dataset: ExpandedDataset, batch_size: int, log: bool = True,
        global_t_pred_hists_path: Path | str | None = None,
) -> ExpandedDataset:
    dataset_t0_updated, dataset_t_pred_updated = {}, {}
    dataset_avg, dataset_avg_count = np.zeros(len(dataset)), np.zeros(len(dataset))

    t_global_dict = {}
    for key in dataset.keys():
        if log:
            print(f'Processing channel {key}...')
        # compute updated t0
        t0_array = dataset.t0[key].copy()
        mask = dataset.notnan_mask[key]

        t_pred_array_masked = _pred_model(model, dataset.wav[key][mask], batch_size)
        t_pred_array = np.full(len(dataset), np.nan)
        t_pred_array[mask] = t_pred_array_masked

        t_global = t0_array[mask] + t_pred_array_masked
        t_mean = np.mean(t_global)
        t_global_dict[key] = t_global

        t0_array[mask] -= t_mean
        dataset_t0_updated[key] = t0_array
        dataset_t_pred_updated[key] = t_pred_array

        # add to avg
        dataset_avg[mask] += t0_array[mask] + t_pred_array_masked
        dataset_avg_count[mask] += 1

    # sum -> average
    dataset_avg /= dataset_avg_count

    if global_t_pred_hists_path is not None:
        _plot_and_save_global_t_pred_hists(t_global_dict, global_t_pred_hists_path)

    return ExpandedDataset(t_avg=dataset_avg, wav=dataset.wav, t0=dataset_t0_updated, t_pred=dataset_t_pred_updated)


def compute_pairwise_precisions(
        dataset: ExpandedDataset, hists_path: str | Path | None = None
) -> tuple[dict[tuple[PlaneChannel, PlaneChannel], float], dict[tuple[PlaneChannel, PlaneChannel], float]]:
    dataset_global_t = copy.deepcopy(dataset.t0)
    for key in dataset.t_pred.keys():
        mask = dataset.notnan_mask[key]
        dataset_global_t[key][mask] += dataset.t_pred[key][mask]

    pairwise_precisions_stat, pairwise_precisions_gauss = {}, {}
    for p_ch1, p_ch2 in combinations(dataset_global_t.keys(), 2):
        if p_ch1[1] == p_ch2[1]:  # Only for corresponding channels
            ch1_timestamps, ch2_timestamps = dataset_global_t[p_ch1], dataset_global_t[p_ch2]
            differences = [ch2_t - ch1_t for ch1_t, ch2_t in zip(ch1_timestamps, ch2_timestamps) if
                           not np.isnan(ch1_t) and not np.isnan(ch2_t)]

            std_stat = np.std(differences)
            _, std_gauss, _, _ = plot_gauss_hist(np.array(differences), show=False)

            pairwise_precisions_stat[(p_ch1, p_ch2)] = std_stat * 1000  # ns -> ps
            pairwise_precisions_gauss[(p_ch1, p_ch2)] = std_gauss * 1000  # ns -> ps

    plt.title("Difference histograms for validation purposes")
    if hists_path is not None:
        save_plt(hists_path)

    plt.close()

    return pairwise_precisions_stat, pairwise_precisions_gauss


def evaluate_model_wrt_cfd_average(model: keras.Model, x: np.ndarray, y_true_t: np.ndarray, batch_size: int,
                                   hist_path: str | Path | None = None) -> tuple[float, float]:
    y_pred = _pred_model(model, x, batch_size)
    mu, std, _, _ = plot_difference_hist(y_true_t, y_pred, show=False, close=False, return_mu=True)

    if hist_path is not None:
        plt.title("Difference histogram for validation purposes")
        save_plt(hist_path)

    plt.close()
    return mu * 1000, std * 1000  # ns -> ps
