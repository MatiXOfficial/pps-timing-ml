from collections import defaultdict
from typing import Callable

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from IPython.display import display, HTML
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

from src.gauss_hist import plot_diff_hist_stats


def train_model(model: tf.keras.Model, name: str, path_component: str, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, lr: float = 0.001, train: bool = True, n_epochs: int = 1000,
                verbose: int = 1, batch_size: int = 2048, lr_patience: int = None, es_patience: int = None,
                loss_weights: float = None, root: str = '.') -> pd.DataFrame:
    """
    Train a Keras model.
    :param model: Keras model
    :param name: model name used in weight paths
    :param path_component: component of the path for model weights and history. Usually the notebook name
    :param X_train:
    :param y_train:
    :param X_val:
    :param y_val:
    :param lr:
    :param train: if True: the model is trained and the weights and history are saved. If False, the weights and
    history are loaded
    :param n_epochs:
    :param verbose: verbosity during training
    :param batch_size:
    :param lr_patience: patience of ReduceLROnPlateau
    :param es_patience: patience of EarlyStopping
    :param loss_weights: loss function values can be multiplied by this weight
    :param root: root directory for model weights
    :return: history dict as a pd.DataFrame
    """
    model.compile(loss='mse', optimizer=optimizers.Adam(lr), loss_weights=loss_weights)

    model_callbacks = [
        callbacks.ModelCheckpoint(filepath=f'{root}/model_weights/{path_component}/{name}/weights', save_best_only=True,
                                  save_weights_only=True)
    ]
    if es_patience is not None:
        model_callbacks.append(callbacks.EarlyStopping(patience=es_patience))
    if lr_patience is not None:
        model_callbacks.append(callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=lr_patience))

    if train:
        history = model.fit(X_train, y_train, epochs=n_epochs, verbose=verbose, batch_size=batch_size,
                            validation_data=(X_val, y_val), callbacks=model_callbacks).history
        pd.DataFrame(history).to_csv(f'{root}/model_weights/{path_component}/{name}/loss_log.csv')

    model.load_weights(f'{root}/model_weights/{path_component}/{name}/weights')
    history = pd.read_csv(f'{root}/model_weights/{path_component}/{name}/loss_log.csv')

    return history


def plot_history(history: dict[str, np.array], title: str, ymax: float = None,
                 figsize: tuple[float, float] = (8, 6)) -> None:
    plt.figure(figsize=figsize)

    X = np.arange(1, len(history['loss']) + 1)

    plt.plot(X, history['loss'], label='train')
    plt.plot(X, history['val_loss'], label='validation')

    if ymax is not None:
        plt.ylim(0, ymax)

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f"validation loss: {min(history[f'val_loss'].values):0.4f}")
    plt.grid()
    plt.legend()

    plt.suptitle(title)
    plt.show()


def plot_difference_hist(y_true, y_pred, plane=None, channel=None, hist_range=(-0.5, 0.5), n_bins=100, show=True,
                         close=True, print_pcov=False) -> tuple[float, float, float]:
    mu, std, pcov, fwhm = plot_diff_hist_stats(y_true, y_pred, show=False, n_bins=n_bins, hist_range=hist_range,
                                               plot_gauss=True, plot_fwhm=True)

    if plane is not None and channel is not None:
        plt.title(f'Diff histogram (p: {plane}, ch: {channel}), mean={mu:0.4f}, std={std:0.4f}')

    if show:
        plt.show()
    else:
        if close:
            plt.close()

    if print_pcov:
        print('Covariance matrix of the Gaussian fit:')
        print(pcov)

    return std, pcov, fwhm


def compare_results(results, names, res_base, base_name='CFD', mult=1000, unit='ps') -> None:
    if res_base is not None:
        print(f"{base_name:>10}: {res_base * mult:0.2f} {unit}")
    for i, (res, name) in enumerate(sorted(zip(results, names))):
        print(f"{name:>10}: {res * mult:0.2f} {unit} (improvement: {(1 - res / res_base) * 100:0.2f} %)")


def count_params(model: keras.Model) -> int:
    return sum([K.count_params(weights) for weights in model.trainable_weights])


def cross_validate_top_hyperparameters(tuner: kt.Tuner, x: np.ndarray, y: np.ndarray,
                                       model_builder: Callable[[kt.HyperParameters], keras.Model], n_epochs: int = 3000,
                                       es_patience: int = 50, reduce_patience: int = 10, batch_size: int = 2048,
                                       n_top: int = 5, n_cv: int = 5, random_state: int = 42) -> dict[int, float]:
    model_scores = defaultdict(list)

    model_callbacks = [
        callbacks.EarlyStopping(patience=es_patience),
        callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=reduce_patience)
    ]

    for i, hyperparameters in enumerate(tuner.get_best_hyperparameters(n_top)):
        display(HTML(f"<h3>Model {i}</h3>"))
        print(hyperparameters.get_config()['values'])
        model_tmp = model_builder(hyperparameters)
        print('Number of parameters:', count_params(model_tmp))

        for train, test in KFold(n_splits=n_cv, shuffle=True, random_state=random_state).split(x):
            X_train, X_val = x[train], x[test]
            y_train, y_val = y[train], y[test]

            model = model_builder(hyperparameters)
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs,
                      callbacks=model_callbacks, batch_size=batch_size, verbose=0)

            score = model.evaluate(X_val, y_val, batch_size=batch_size, verbose=0)
            model_scores[i].append(score)

            print(f"Got score: {score}")

    return dict(model_scores)
