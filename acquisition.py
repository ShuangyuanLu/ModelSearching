from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AcquisitionFunction(ABC):
    @abstractmethod
    def score(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class UCBAcquisition(AcquisitionFunction):
    def __init__(self, beta: float) -> None:
        self._beta = float(beta)

    def score(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        mean = np.asarray(mean, dtype=float)
        std = np.asarray(std, dtype=float)
        return mean + self._beta * np.maximum(std, 0.0)
