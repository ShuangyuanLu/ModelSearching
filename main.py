from __future__ import annotations

from acquisition import UCBAcquisition
from config import BOUNDS, GRID_RESOLUTION, N_BO_ITERATIONS, N_INITIAL_SAMPLES, UCB_BETA, rng
from optimizer import BayesianOptimizer
from objective import ToyBlackBoxFunction
from sampling import generate_initial_data
from surrogate import GPSurrogate


def main() -> None:
    objective = ToyBlackBoxFunction(BOUNDS)
    acquisition = UCBAcquisition(beta=UCB_BETA)
    x_init, y_init = generate_initial_data(
        objective=objective,
        bounds=BOUNDS,
        n_samples=N_INITIAL_SAMPLES,
        rng=rng,
    )
    optimizer = BayesianOptimizer(
        objective=objective,
        surrogate=GPSurrogate(),
        acquisition=acquisition,
        bounds=BOUNDS,
        grid_resolution=GRID_RESOLUTION,
    )
    result = optimizer.run(
        x_init=x_init,
        y_init=y_init,
        n_iterations=N_BO_ITERATIONS,
    )
    print(f"initial_samples={len(x_init)}, total_samples={len(result.x_samples)}")
    print(f"best_point={result.best_point}, best_value={result.best_value:.4f}")
    print(f"best_values.shape={result.best_values.shape}")


if __name__ == "__main__":
    main()


