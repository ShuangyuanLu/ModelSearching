# ModelSearching

Bayesian optimization experiments for searching classical Monte Carlo model parameters. The repository contains a small toy optimizer example and a higher-cost classical steady-state search that calls an external simulation backend.

## Repository Layout

| File | Purpose |
| --- | --- |
| `main.py` | Runs a 2D toy black-box Bayesian optimization example. |
| `main_classical.py` | Runs the classical-model search with the local scikit-learn Gaussian-process optimizer and UCB acquisition. |
| `main_botorch.py` | Runs the classical-model search with BoTorch `SingleTaskGP` and `qLogExpectedImprovement`. Writes `botorch_trace.csv`. |
| `classical_problem.py` | Defines the classical simulation, analysis settings, period template, searchable theta parameters, bounds, and backend path. |
| `classical_objective.py` | Converts optimizer vectors into backend JSON specs, runs the backend CLI, caches results, and computes scalar objective scores. |
| `botorch_optimizer.py` | BoTorch-based optimizer implementation. |
| `optimizer.py` | Lightweight Bayesian optimizer using a candidate grid or random candidate pool. |
| `surrogate.py` | scikit-learn Gaussian-process surrogate. |
| `acquisition.py` | Acquisition functions, currently upper confidence bound. |
| `sampling.py` | Uniform sampling utilities for initial data and candidate pools. |
| `selection.py` | Candidate-grid construction and acquisition-based point selection. |
| `objective.py` | Toy 2D objective used by `main.py`. |
| `config.py` | Toy example constants and GP settings. |

Generated trace files such as `botorch_trace.csv` and `plaquette_botorch_trace.csv` record optimization diagnostics.

## Requirements

Python dependencies used by the current code:

```bash
numpy
scikit-learn
torch
botorch
gpytorch
```

The classical search also requires the external backend configured in `classical_problem.py`:

```python
BACKEND_ROOT = Path("/mnt/c/Users/shuan/OneDrive/Documents/PycharmProjects/QCphasetransition")
BACKEND_PYTHON = "/home/shuan/cupy-env/bin/python"
```

That backend is expected to provide:

```bash
evaluate_spec_cli.py
```

with an interface compatible with:

```bash
python evaluate_spec_cli.py SPEC_PATH -o RESULT_PATH --no-measurements
```

## Running

Run the toy 2D optimization:

```bash
python main.py
```

Run the classical search with the local optimizer:

```bash
python main_classical.py
```

Run the classical search with BoTorch:

```bash
python main_botorch.py
```

The BoTorch run prints the best point, best score, prediction error statistics, and the backend analysis of the best model. It also writes:

```bash
botorch_trace.csv
```

## Classical Search Setup

The classical model is defined in `classical_problem.py`.

Important constants:

```python
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
```

The optimizer searches over theta values in four period blocks:

```text
tile_11  one-site block
tile_12  1x2 block
tile_21  2x1 block
tile_22  2x2 block
```

Each block has one reference theta fixed to zero because transition probabilities are invariant under adding the same constant to every theta in that block. The remaining theta entries become `SEARCH_PARAMETERS`.

## Objective Score

The scalar objective is computed by `stable_tail_mean_score()` in `classical_objective.py`.

By default, it maximizes the tail mean of `m_2`, then subtracts penalties for:

- unstable tail behavior
- larger standard error

In simplified form:

```text
score = m_2_tail_mean - instability_penalty - stderr_penalty
```

The objective is expensive because every evaluated point runs the external backend. Results are cached in memory by a hash of the generated JSON spec.

## Model Simplicity

The current optimizer maximizes only the backend score. If the scientific goal is to find simple classical models, simplicity should be added explicitly.

A practical definition is sparsity:

```text
complexity = number of active theta terms
```

where a theta term is active when:

```text
abs(theta) > epsilon
```

For this codebase, a weighted complexity is usually better:

```text
complexity =
    1.0 * active tile_11 terms
  + 1.5 * active tile_12/tile_21 terms
  + 3.0 * active tile_22 terms
```

Then select models using either a penalized score:

```text
simple_score = physics_score - lambda_complexity * complexity
```

or a tolerance rule:

```text
choose the simplest model with score >= best_score - tolerance
```

The tolerance rule is often easier to interpret: it asks for the simplest model that performs nearly as well as the best observed model.

## Trace Files

`main_botorch.py` writes one row per BoTorch iteration:

```text
bo_iteration
sample_index
predicted_mean
predicted_std
observed_score
prediction_error
absolute_error
acquisition_value
best_score_so_far
```

Use this file to inspect surrogate quality, optimization progress, and whether new evaluations are still improving the best score.

## Common Edits

Change the number of BoTorch samples in `main_botorch.py`:

```python
N_INITIAL_SAMPLES = 64
N_BO_ITERATIONS = 100
NUM_RESTARTS = 10
RAW_SAMPLES = 256
```

Change the classical simulation or search space in `classical_problem.py`.

Change the score definition in `classical_objective.py` or pass a different `score_getter` from `create_classical_objective()`.

## Notes

- Higher `N_INITIAL_SAMPLES` and `N_BO_ITERATIONS` improve search coverage but increase backend runtime.
- `tile_22` terms are usually less simple than one-site or nearest-neighbor terms.
- If the backend path or Python environment changes, update `BACKEND_ROOT` and `BACKEND_PYTHON` before running classical searches.
