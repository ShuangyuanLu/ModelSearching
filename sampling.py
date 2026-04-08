from __future__ import annotations

from typing import Callable

import numpy as np


def sample_uniform(
    bounds: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=float)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    return rng.uniform(lower, upper, size=(n_samples, bounds.shape[0]))


def generate_initial_data(
    objective: Callable[[np.ndarray], np.ndarray],
    bounds: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x_init = sample_uniform(bounds=bounds, n_samples=n_samples, rng=rng)
    y_init = np.asarray(objective(x_init), dtype=float)
    return x_init, y_init
