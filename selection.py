from __future__ import annotations

import numpy as np

from acquisition import AcquisitionFunction
from surrogate import SurrogateModel


def build_candidate_grid(bounds: np.ndarray, resolution: int) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=float)
    axes = [
        np.linspace(lower, upper, resolution)
        for lower, upper in bounds
    ]
    mesh = np.meshgrid(*axes, indexing="xy")
    return np.column_stack([axis.ravel() for axis in mesh])


def select_next_point(
    surrogate: SurrogateModel,
    acquisition: AcquisitionFunction,
    candidates: np.ndarray,
    observed_points: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    candidates = np.asarray(candidates, dtype=float)
    mean, std = surrogate.predict(candidates)
    scores = acquisition.score(mean, std)

    if observed_points is not None:
        observed_points = np.asarray(observed_points, dtype=float)
        already_sampled = np.all(
            np.isclose(candidates[:, None, :], observed_points[None, :, :]),
            axis=2,
        ).any(axis=1)
        scores = np.asarray(scores, dtype=float).copy()
        scores[already_sampled] = -np.inf

    if np.all(np.isneginf(scores)):
        raise ValueError("No unsampled candidate points remain.")

    best_index = int(np.argmax(scores))
    next_point = candidates[best_index]
    best_score = float(scores[best_index])
    return next_point, mean, std, scores, best_score
