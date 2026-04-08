from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from config import GP_LENGTH_SCALE, GP_NU


class SurrogateModel(ABC):
    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class GPSurrogate(SurrogateModel):
    def __init__(self) -> None:
        self._model: GaussianProcessRegressor | None = None
        self._input_dim: int | None = None

    def _build_model(self, input_dim: int) -> GaussianProcessRegressor:
        kernel = (
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * Matern(
                length_scale=np.full(input_dim, GP_LENGTH_SCALE),
                length_scale_bounds="fixed",
                nu=GP_NU,
            )
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            optimizer=None,
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        x_train = np.asarray(x_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float).ravel()
        input_dim = int(x_train.shape[1])
        if self._model is None or self._input_dim != input_dim:
            self._model = self._build_model(input_dim)
            self._input_dim = input_dim
        self._model.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("GPSurrogate must be fitted before predict().")
        x_test = np.asarray(x_test, dtype=float)
        mean, std = self._model.predict(x_test, return_std=True)
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
