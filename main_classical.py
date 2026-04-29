from __future__ import annotations

import numpy as np

from acquisition import UCBAcquisition
from classical_problem import RANDOM_SEED, SEARCH_PARAMETERS, create_classical_objective
from optimizer import BayesianOptimizer
from sampling import generate_initial_data
from surrogate import GPSurrogate


N_INITIAL_SAMPLES = 48
N_BO_ITERATIONS = 80
N_CANDIDATES_PER_STEP = 2048
UCB_BETA = 3.0


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    objective = create_classical_objective()

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
