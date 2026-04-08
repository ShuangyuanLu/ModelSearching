from __future__ import annotations

import numpy as np


class ToyBlackBoxFunction:
    """A hidden 2D objective with several local peaks."""

    def __init__(self, bounds: np.ndarray) -> None:
        self._bounds = np.asarray(bounds, dtype=float)

    @property
    def bounds(self) -> np.ndarray:
        return self._bounds

    def __call__(self, x: np.ndarray) -> np.ndarray:
        points = np.asarray(x, dtype=float)
        x1 = points[..., 0]
        x2 = points[..., 1]

        peak_a = 1.7 * np.exp(-((x1 - 1.6) ** 2 + (x2 + 1.2) ** 2) / 0.7)
        peak_b = 1.2 * np.exp(-((x1 + 2.0) ** 2) / 1.8 - ((x2 - 1.5) ** 2) / 0.9)
        peak_c = 0.95 * np.exp(-((x1 - 0.3) ** 2) / 3.2 - ((x2 - 2.4) ** 2) / 1.4)
        peak_d = 0.8 * np.exp(-((x1 + 2.6) ** 2 + (x2 + 2.2) ** 2) / 1.1)
        ripple = 0.08 * np.sin(2.2 * x1) * np.cos(1.7 * x2)

        return peak_a + peak_b + peak_c + peak_d + ripple
