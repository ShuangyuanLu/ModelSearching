from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine


@dataclass
class BotorchOptimizationResult:
    x_samples: np.ndarray
    y_samples: np.ndarray
    selected_points: np.ndarray
    best_values: np.ndarray
    predicted_means: np.ndarray
    predicted_stds: np.ndarray
    observed_values: np.ndarray
    acquisition_values: np.ndarray
    last_candidate: np.ndarray

    @property
    def best_index(self) -> int:
        return int(np.argmax(self.y_samples))

    @property
    def best_point(self) -> np.ndarray:
        return self.x_samples[self.best_index]

    @property
    def best_value(self) -> float:
        return float(self.y_samples[self.best_index])

    @property
    def prediction_errors(self) -> np.ndarray:
        return self.observed_values - self.predicted_means

    @property
    def prediction_mae(self) -> float:
        if self.observed_values.size == 0:
            return float("nan")
        return float(np.mean(np.abs(self.prediction_errors)))


class BotorchOptimizer:
    def __init__(
        self,
        objective: Callable[[np.ndarray], np.ndarray],
        bounds: np.ndarray,
        seed: int,
        num_restarts: int = 10,
        raw_samples: int = 256,
        dtype: torch.dtype = torch.double,
    ) -> None:
        self._objective = objective
        self._bounds = torch.as_tensor(bounds, dtype=dtype).T.contiguous()
        self._dimension = int(self._bounds.shape[1])
        self._dtype = dtype
        self._num_restarts = int(num_restarts)
        self._raw_samples = int(raw_samples)
        self._sobol = SobolEngine(dimension=self._dimension, scramble=True, seed=seed)

    def _sample_initial_points(self, n_samples: int) -> torch.Tensor:
        draws = self._sobol.draw(n_samples).to(dtype=self._dtype)
        lower = self._bounds[0]
        upper = self._bounds[1]
        return lower + (upper - lower) * draws

    def _evaluate_points(self, points: torch.Tensor) -> torch.Tensor:
        values = self._objective(points.detach().cpu().numpy())
        return torch.as_tensor(np.asarray(values, dtype=float), dtype=self._dtype).view(-1, 1)

    def _fit_model(self, train_x: torch.Tensor, train_y: torch.Tensor) -> SingleTaskGP:
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            input_transform=Normalize(d=self._dimension),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def _next_candidate(self, model: SingleTaskGP, best_f: torch.Tensor) -> tuple[torch.Tensor, float]:
        acquisition = qLogExpectedImprovement(model=model, best_f=best_f)
        candidate, _ = optimize_acqf(
            acq_function=acquisition,
            bounds=self._bounds,
            q=1,
            num_restarts=self._num_restarts,
            raw_samples=self._raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
        )
        acquisition_value = float(acquisition(candidate).view(-1)[0].item())
        return candidate.detach(), acquisition_value

    def run(self, n_initial_samples: int, n_iterations: int) -> BotorchOptimizationResult:
        train_x = self._sample_initial_points(n_initial_samples)
        train_y = self._evaluate_points(train_x)
        selected_points: list[np.ndarray] = []
        best_values = [float(torch.max(train_y).item())]
        predicted_means: list[float] = []
        predicted_stds: list[float] = []
        observed_values: list[float] = []
        acquisition_values: list[float] = []
        last_candidate = train_x[-1].detach().cpu().numpy().copy()

        for _ in range(n_iterations):
            model = self._fit_model(train_x, train_y)
            candidate, acquisition_value = self._next_candidate(model=model, best_f=train_y.max())
            posterior = model.posterior(candidate)
            predicted_mean = float(posterior.mean.view(-1)[0].item())
            predicted_std = float(torch.sqrt(posterior.variance.clamp_min(0.0)).view(-1)[0].item())
            candidate_y = self._evaluate_points(candidate)
            observed_value = float(candidate_y.view(-1)[0].item())

            train_x = torch.cat([train_x, candidate], dim=0)
            train_y = torch.cat([train_y, candidate_y], dim=0)
            selected_points.append(candidate.squeeze(0).detach().cpu().numpy().copy())
            best_values.append(float(torch.max(train_y).item()))
            predicted_means.append(predicted_mean)
            predicted_stds.append(predicted_std)
            observed_values.append(observed_value)
            acquisition_values.append(acquisition_value)
            last_candidate = candidate.squeeze(0).detach().cpu().numpy().copy()

        return BotorchOptimizationResult(
            x_samples=train_x.detach().cpu().numpy(),
            y_samples=train_y.squeeze(-1).detach().cpu().numpy(),
            selected_points=np.asarray(selected_points, dtype=float),
            best_values=np.asarray(best_values, dtype=float),
            predicted_means=np.asarray(predicted_means, dtype=float),
            predicted_stds=np.asarray(predicted_stds, dtype=float),
            observed_values=np.asarray(observed_values, dtype=float),
            acquisition_values=np.asarray(acquisition_values, dtype=float),
            last_candidate=last_candidate,
        )
