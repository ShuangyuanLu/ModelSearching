from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from acquisition import AcquisitionFunction
from sampling import sample_uniform
from selection import build_candidate_grid, select_next_point
from surrogate import SurrogateModel


@dataclass
class BayesianOptimizationResult:
    x_samples: np.ndarray
    y_samples: np.ndarray
    selected_points: np.ndarray
    best_values: np.ndarray
    last_candidates: np.ndarray
    last_mean: np.ndarray
    last_std: np.ndarray
    last_scores: np.ndarray

    @property
    def best_index(self) -> int:
        return int(np.argmax(self.y_samples))

    @property
    def best_point(self) -> np.ndarray:
        return self.x_samples[self.best_index]

    @property
    def best_value(self) -> float:
        return float(self.y_samples[self.best_index])


class BayesianOptimizer:
    def __init__(
        self,
        objective: Callable[[np.ndarray], np.ndarray],
        surrogate: SurrogateModel,
        acquisition: AcquisitionFunction,
        bounds: np.ndarray,
        grid_resolution: int | None = None,
        candidate_pool_size: int | None = None,
        candidate_rng: np.random.Generator | None = None,
    ) -> None:
        if (grid_resolution is None) == (candidate_pool_size is None):
            raise ValueError("Specify exactly one of grid_resolution or candidate_pool_size.")

        self._objective = objective
        self._surrogate = surrogate
        self._acquisition = acquisition
        self._bounds = np.asarray(bounds, dtype=float)
        self._grid_resolution = grid_resolution
        self._candidate_pool_size = candidate_pool_size
        self._candidate_rng = candidate_rng or np.random.default_rng()
        self._fixed_candidates = (
            build_candidate_grid(bounds=self._bounds, resolution=grid_resolution)
            if grid_resolution is not None
            else None
        )

    @property
    def candidates(self) -> np.ndarray:
        if self._fixed_candidates is None:
            raise RuntimeError("Candidates are sampled dynamically when using candidate_pool_size.")
        return self._fixed_candidates

    def _build_candidates(self) -> np.ndarray:
        if self._fixed_candidates is not None:
            return self._fixed_candidates
        return sample_uniform(
            bounds=self._bounds,
            n_samples=int(self._candidate_pool_size),
            rng=self._candidate_rng,
        )

    def run(
        self,
        x_init: np.ndarray,
        y_init: np.ndarray,
        n_iterations: int,
    ) -> BayesianOptimizationResult:
        x_samples = np.asarray(x_init, dtype=float).copy()
        y_samples = np.asarray(y_init, dtype=float).ravel().copy()
        selected_points: list[np.ndarray] = []
        best_values = [float(np.max(y_samples))]

        initial_candidates = self._build_candidates()
        last_candidates = initial_candidates
        last_mean = np.empty(initial_candidates.shape[0], dtype=float)
        last_std = np.empty(initial_candidates.shape[0], dtype=float)
        last_scores = np.empty(initial_candidates.shape[0], dtype=float)

        for _ in range(n_iterations):
            candidates = self._build_candidates()
            self._surrogate.fit(x_samples, y_samples)
            next_point, mean, std, scores, _ = select_next_point(
                surrogate=self._surrogate,
                acquisition=self._acquisition,
                candidates=candidates,
                observed_points=x_samples,
            )
            next_value = float(np.asarray(self._objective(next_point), dtype=float))

            x_samples = np.vstack([x_samples, next_point])
            y_samples = np.append(y_samples, next_value)
            selected_points.append(next_point.copy())
            best_values.append(float(np.max(y_samples)))

            last_candidates = candidates
            last_mean = mean
            last_std = std
            last_scores = scores

        return BayesianOptimizationResult(
            x_samples=x_samples,
            y_samples=y_samples,
            selected_points=np.asarray(selected_points, dtype=float),
            best_values=np.asarray(best_values, dtype=float),
            last_candidates=last_candidates,
            last_mean=last_mean,
            last_std=last_std,
            last_scores=last_scores,
        )
