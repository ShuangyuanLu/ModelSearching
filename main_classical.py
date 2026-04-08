from __future__ import annotations

from pathlib import Path

import numpy as np

from acquisition import UCBAcquisition
from classical_objective import ClassicalSteadyStateObjective, SearchParameter, stable_tail_mean_score
from optimizer import BayesianOptimizer
from sampling import generate_initial_data
from surrogate import GPSurrogate


BACKEND_ROOT = Path("/mnt/c/Users/shuan/OneDrive/Documents/PycharmProjects/QCphasetransition")
BACKEND_PYTHON = "/home/shuan/cupy-env/bin/python"

RANDOM_SEED = 7
N_INITIAL_SAMPLES = 48
N_BO_ITERATIONS = 80
N_CANDIDATES_PER_STEP = 2048
UCB_BETA = 3.0

SIMULATION = {
    "L": 8,
    "N_sample": 128,
    "n_periods": 60,
    "measure_every_periods": 5,
    "random_seed": 0,
}

ANALYSIS = {
    "min_tail": 6,
    "stability_z": 2.0,
    "target_observable": "m_2",
    "target_stderr": 0.02,
}

PERIOD_TEMPLATE = [
    {
        "name": "tile_11",
        "family": "symmetric_kernel",
        "support_shape": [1, 1],
        "shifts": [[0, 0]],
        "params": {
            "theta": [0.0] * 2,
        },
    },
    {
        "name": "tile_12",
        "family": "symmetric_kernel",
        "support_shape": [1, 2],
        "shifts": [[0, 0], [0, 1]],
        "params": {
            "theta": [0.0] * 6,
        },
    },
    {
        "name": "tile_21",
        "family": "symmetric_kernel",
        "support_shape": [2, 1],
        "shifts": [[0, 0], [1, 0]],
        "params": {
            "theta": [0.0] * 6,
        },
    },
    {
        "name": "tile_22",
        "family": "symmetric_kernel",
        "support_shape": [2, 2],
        "shifts": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "params": {
            "theta": [0.0] * 31,
        },
    },
]

THETA_BOUNDS_BY_BLOCK = {
    "tile_11": (-3.0, 3.0),
    "tile_12": (-3.0, 3.0),
    "tile_21": (-3.0, 3.0),
    "tile_22": (-3.0, 3.0),
}


def build_search_parameters(period_template: list[dict]) -> list[SearchParameter]:
    parameters: list[SearchParameter] = []
    for block in period_template:
        lower, upper = THETA_BOUNDS_BY_BLOCK[block["name"]]
        for theta_index in range(len(block["params"]["theta"])):
            parameters.append(
                SearchParameter(
                    block_name=block["name"],
                    theta_index=theta_index,
                    lower=lower,
                    upper=upper,
                )
            )
    return parameters


SEARCH_PARAMETERS = build_search_parameters(PERIOD_TEMPLATE)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    objective = ClassicalSteadyStateObjective(
        backend_root=BACKEND_ROOT,
        backend_python=BACKEND_PYTHON,
        simulation=SIMULATION,
        period_template=PERIOD_TEMPLATE,
        parameters=SEARCH_PARAMETERS,
        analysis=ANALYSIS,
        score_getter=lambda result: stable_tail_mean_score(result, observable="m_2"),
        workdir=Path("/tmp/model-search-backend"),
    )

    x_init, y_init = generate_initial_data(
        objective=objective,
        bounds=objective.bounds,
        n_samples=N_INITIAL_SAMPLES,
        rng=rng,
    )

    optimizer = BayesianOptimizer(
        objective=objective,
        surrogate=GPSurrogate(),
        acquisition=UCBAcquisition(beta=UCB_BETA),
        bounds=objective.bounds,
        candidate_pool_size=N_CANDIDATES_PER_STEP,
        candidate_rng=rng,
    )
    result = optimizer.run(
        x_init=x_init,
        y_init=y_init,
        n_iterations=N_BO_ITERATIONS,
    )

    best_backend_result = objective.evaluate_result(result.best_point)
    primary_observable = best_backend_result["analysis"]["primary_observable"]
    primary_analysis = best_backend_result["analysis"]["primary_analysis"]

    print(f"searched_parameters={len(SEARCH_PARAMETERS)}")
    print(f"initial_samples={len(x_init)}, total_samples={len(result.x_samples)}")
    print(f"best_point={result.best_point}")
    print(f"best_score={result.best_value:.6f}")
    print(f"{primary_observable}_tail_mean={best_backend_result['summary'][primary_observable]['tail_mean']:.6f}")
    print(f"{primary_observable}_stable={primary_analysis['stable']}")
    print(f"{primary_observable}_stderr={primary_analysis['stderr']}")


if __name__ == "__main__":
    main()
