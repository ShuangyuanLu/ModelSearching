from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class SearchParameter:
    block_name: str
    theta_index: int
    lower: float
    upper: float


def stable_tail_mean_score(
    result: dict,
    observable: str = "m_2",
    instability_penalty: float = 0.25,
    stderr_penalty: float = 0.05,
) -> float:
    summary_value = result["summary"][observable]["tail_mean"]
    if summary_value is None:
        raise ValueError(f"Observable {observable!r} has no tail mean in backend result.")

    analysis = result["analysis"]["observables"][observable]
    score = float(summary_value)

    if not analysis["stable"]:
        z_score = analysis.get("stability_z_score")
        score -= instability_penalty * (float(z_score) if z_score is not None else 1.0)

    stderr = analysis.get("stderr")
    if stderr is not None:
        score -= stderr_penalty * float(stderr)

    return score


class ClassicalSteadyStateObjective:
    def __init__(
        self,
        backend_root: Path,
        backend_python: str,
        simulation: dict,
        period_template: Sequence[dict],
        parameters: Sequence[SearchParameter],
        analysis: dict | None = None,
        observer: dict | None = None,
        score_getter: Callable[[dict], float] | None = None,
        workdir: Path | None = None,
    ) -> None:
        self._backend_root = Path(backend_root)
        self._backend_python = str(backend_python)
        self._simulation = copy.deepcopy(simulation)
        self._period_template = copy.deepcopy(list(period_template))
        self._parameters = tuple(parameters)
        self._analysis = copy.deepcopy(analysis) if analysis is not None else None
        self._observer = copy.deepcopy(observer) if observer is not None else {"kind": "ising_magnetization"}
        self._score_getter = score_getter or stable_tail_mean_score
        self._workdir = Path(workdir) if workdir is not None else Path("/tmp/model-search-backend")
        self._result_cache: dict[str, dict] = {}

        if not self._parameters:
            raise ValueError("At least one search parameter is required.")

        self._validate_parameters()

    @property
    def bounds(self) -> np.ndarray:
        return np.asarray(
            [[parameter.lower, parameter.upper] for parameter in self._parameters],
            dtype=float,
        )

    def _validate_parameters(self) -> None:
        blocks_by_name = {block["name"]: block for block in self._period_template}
        for parameter in self._parameters:
            if parameter.block_name not in blocks_by_name:
                raise KeyError(f"Unknown period block {parameter.block_name!r}.")

            theta = blocks_by_name[parameter.block_name]["params"]["theta"]
            if parameter.theta_index < 0 or parameter.theta_index >= len(theta):
                raise IndexError(
                    f"Theta index {parameter.theta_index} is out of range for block "
                    f"{parameter.block_name!r} with length {len(theta)}."
                )

    def vector_to_spec(self, x: np.ndarray) -> dict:
        vector = np.asarray(x, dtype=float).ravel()
        if vector.shape != (len(self._parameters),):
            raise ValueError(
                f"Expected parameter vector of shape ({len(self._parameters)},), got {vector.shape}."
            )

        spec = {
            "simulation": copy.deepcopy(self._simulation),
            "observer": copy.deepcopy(self._observer),
            "period": copy.deepcopy(self._period_template),
        }
        if self._analysis is not None:
            spec["analysis"] = copy.deepcopy(self._analysis)

        blocks_by_name = {block["name"]: block for block in spec["period"]}
        for value, parameter in zip(vector, self._parameters, strict=True):
            block = blocks_by_name[parameter.block_name]
            theta = list(block["params"]["theta"])
            theta[parameter.theta_index] = float(value)
            block["params"]["theta"] = theta

        return spec

    def _run_backend(self, spec: dict) -> dict:
        payload = json.dumps(spec, sort_keys=True, separators=(",", ":"))
        cache_key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if cache_key in self._result_cache:
            return self._result_cache[cache_key]

        self._workdir.mkdir(parents=True, exist_ok=True)
        spec_path = self._workdir / f"{cache_key}.spec.json"
        result_path = self._workdir / f"{cache_key}.result.json"
        spec_path.write_text(payload, encoding="utf-8")

        subprocess.run(
            [
                self._backend_python,
                str(self._backend_root / "evaluate_spec_cli.py"),
                str(spec_path),
                "-o",
                str(result_path),
                "--no-measurements",
            ],
            check=True,
            cwd=self._backend_root,
        )

        result = json.loads(result_path.read_text(encoding="utf-8"))
        self._result_cache[cache_key] = result
        return result

    def evaluate_result(self, x: np.ndarray) -> dict:
        return self._run_backend(self.vector_to_spec(x))

    def evaluate_score(self, x: np.ndarray) -> float:
        return float(self._score_getter(self.evaluate_result(x)))

    def __call__(self, x: np.ndarray) -> np.ndarray | float:
        points = np.asarray(x, dtype=float)
        if points.ndim == 1:
            return self.evaluate_score(points)

        if points.ndim != 2:
            raise ValueError(f"Expected a 1D or 2D array of points, got ndim={points.ndim}.")

        return np.asarray([self.evaluate_score(point) for point in points], dtype=float)
