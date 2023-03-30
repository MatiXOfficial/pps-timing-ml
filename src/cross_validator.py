import pickle
from pathlib import Path
from typing import Callable

import keras_tuner as kt
import numpy as np
from IPython.display import display, HTML
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import callbacks

from src.network_utils import count_params


class KerasTunerCrossValidator:
    def __init__(self, tuner: kt.Tuner, x: np.ndarray, y: np.ndarray,
                 model_builder: Callable[[kt.HyperParameters], keras.Model], directory: Path | str,
                 project_name: Path | str, overwrite: bool = True, n_epochs: int = 3000, es_patience: int = 50,
                 reduce_patience: int = 10, batch_size: int = 2048, n_top: int = 5, n_cv: int = 5,
                 n_executions: int = 1, random_state: int = 42):
        self.tuner = tuner
        self.x = x
        self.y = y
        self.model_builder = model_builder
        self.directory = directory if isinstance(directory, Path) else Path(directory)
        self.project_name = project_name
        self.overwrite = overwrite
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_top = n_top
        self.n_cv = n_cv
        self.n_executions = n_executions
        self.random_state = random_state

        self.model_callbacks = [
            callbacks.EarlyStopping(patience=es_patience),
            callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=reduce_patience)
        ]

    def __call__(self) -> dict[int, list[float]]:
        """Return cross-val scores for each of the top-n models"""
        project_dir = self._build_project_dir()

        model_scores = {}
        for i_hp, hyperparameters in enumerate(self.tuner.get_best_hyperparameters(self.n_top)):
            self._print_model_log(hyperparameters, i_hp)

            hp_results_file = project_dir / (str(i_hp) + '.pkl')
            if self.overwrite or not hp_results_file.is_file():
                hp_scores = self._compute_hp_scores(hyperparameters)
                with open(hp_results_file, 'wb') as file:
                    pickle.dump(hp_scores, file)
            else:
                with open(hp_results_file, 'rb') as file:
                    hp_scores = pickle.load(file)
                for split_scores in hp_scores:
                    self._print_split_scores_log(split_scores)

            model_scores[i_hp] = [np.average(split_scores) for split_scores in hp_scores]

        return model_scores

    def _build_project_dir(self) -> Path:
        project_dir = self.directory / self.project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir

    def _compute_hp_scores(self, hp: kt.HyperParameters) -> list[list[float]]:
        hp_scores = []
        for train, test in KFold(n_splits=self.n_cv, shuffle=True, random_state=self.random_state).split(self.x):
            X_train, X_val = self.x[train], self.x[test]
            y_train, y_val = self.y[train], self.y[test]

            split_scores = []
            for _ in range(self.n_executions):
                score = self._compute_single_score(hp, X_train, y_train, X_val, y_val)
                split_scores.append(score)

            self._print_split_scores_log(split_scores)
            hp_scores.append(split_scores)

        return hp_scores

    def _compute_single_score(self, hp: kt.HyperParameters, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                              y_val: np.ndarray) -> float:
        model = self.model_builder(hp)
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=self.n_epochs,
                  callbacks=self.model_callbacks, batch_size=self.batch_size, verbose=0)

        score = model.evaluate(x_val, y_val, batch_size=self.batch_size, verbose=0)
        return score

    def _print_model_log(self, hp: kt.HyperParameters, i_hp: int) -> None:
        display(HTML(f"<h3>Model {i_hp}</h3>"))
        print(hp.get_config()['values'])
        model_tmp = self.model_builder(hp)
        print('Number of parameters:', count_params(model_tmp))

    @staticmethod
    def _print_split_scores_log(split_scores: list[float]) -> None:
        if len(split_scores) == 1:
            print(f"Got score: {split_scores[0]:0.4f}")
        else:
            avg_score = np.average(split_scores)
            scores_str = ', '.join([f"{score:0.4f}" for score in split_scores])
            print(f"Got score: {avg_score:0.4f} ({scores_str})")
