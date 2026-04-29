from __future__ import annotations

import csv
from pathlib import Path

from classical_problem import RANDOM_SEED, SEARCH_PARAMETERS, create_classical_objective
from botorch_optimizer import BotorchOptimizer


N_INITIAL_SAMPLES = 64
N_BO_ITERATIONS = 100
NUM_RESTARTS = 10
RAW_SAMPLES = 256
TRACE_OUTPUT_PATH = Path("botorch_trace.csv")


def write_trace_csv(result, n_initial_samples: int, output_path: Path) -> Path:
    output_path = output_path.resolve()
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "bo_iteration",
                "sample_index",
                "predicted_mean",
                "predicted_std",
                "observed_score",
                "prediction_error",
                "absolute_error",
                "acquisition_value",
                "best_score_so_far",
            ]
        )
        for index in range(len(result.observed_values)):
            prediction_error = float(result.prediction_errors[index])
            writer.writerow(
                [
                    index + 1,
                    n_initial_samples + index + 1,
                    float(result.predicted_means[index]),
                    float(result.predicted_stds[index]),
                    float(result.observed_values[index]),
                    prediction_error,
                    abs(prediction_error),
                    float(result.acquisition_values[index]),
                    float(result.best_values[index + 1]),
                ]
            )
    return output_path


def main() -> None:
    objective = create_classical_objective()
    optimizer = BotorchOptimizer(
        objective=objective,
        bounds=objective.bounds,
        seed=RANDOM_SEED,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    result = optimizer.run(
        n_initial_samples=N_INITIAL_SAMPLES,
        n_iterations=N_BO_ITERATIONS,
    )

    best_backend_result = objective.evaluate_result(result.best_point)
    primary_observable = best_backend_result["analysis"]["primary_observable"]
    primary_analysis = best_backend_result["analysis"]["primary_analysis"]
    trace_path = write_trace_csv(
        result=result,
        n_initial_samples=N_INITIAL_SAMPLES,
        output_path=TRACE_OUTPUT_PATH,
    )

    print("optimizer=botorch_single_task_gp_qlogei")
    print(f"searched_parameters={len(SEARCH_PARAMETERS)}")
    print(f"initial_samples={N_INITIAL_SAMPLES}, total_samples={len(result.x_samples)}")
    print(f"best_point={result.best_point}")
    print(f"best_score={result.best_value:.6f}")
    print(f"bo_prediction_mae={result.prediction_mae:.6f}")
    print(f"trace_csv={trace_path}")
    print(f"{primary_observable}_tail_mean={best_backend_result['summary'][primary_observable]['tail_mean']:.6f}")
    print(f"{primary_observable}_stable={primary_analysis['stable']}")
    print(f"{primary_observable}_stderr={primary_analysis['stderr']}")


if __name__ == "__main__":
    main()
