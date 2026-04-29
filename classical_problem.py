from __future__ import annotations

import copy
from pathlib import Path

from classical_objective import ClassicalSteadyStateObjective, SearchParameter, stable_tail_mean_score


BACKEND_ROOT = Path("/mnt/c/Users/shuan/OneDrive/Documents/PycharmProjects/QCphasetransition")
BACKEND_PYTHON = "/home/shuan/cupy-env/bin/python"
RANDOM_SEED = 7

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
    "tile_11": (-5.0, 5.0),
    "tile_12": (-5.0, 5.0),
    "tile_21": (-5.0, 5.0),
    "tile_22": (-5.0, 5.0),
}

REFERENCE_THETA_INDEX_BY_BLOCK = {
    "tile_11": 1,
    "tile_12": 5,
    "tile_21": 5,
    "tile_22": 30,
}


def build_search_parameters(period_template: list[dict]) -> list[SearchParameter]:
    parameters: list[SearchParameter] = []
    for block in period_template:
        lower, upper = THETA_BOUNDS_BY_BLOCK[block["name"]]
        theta_length = len(block["params"]["theta"])
        reference_theta_index = REFERENCE_THETA_INDEX_BY_BLOCK[block["name"]]
        if not 0 <= reference_theta_index < theta_length:
            raise IndexError(
                f"Reference theta index {reference_theta_index} is out of range for "
                f"block {block['name']!r} with length {theta_length}."
            )

        # The transition probabilities are invariant under adding the same
        # constant to every theta in a block. Fix one reference theta to zero so
        # the optimizer only searches independent directions.
        for theta_index in range(theta_length):
            if theta_index == reference_theta_index:
                continue
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


def create_classical_objective(workdir: Path | None = None) -> ClassicalSteadyStateObjective:
    return ClassicalSteadyStateObjective(
        backend_root=BACKEND_ROOT,
        backend_python=BACKEND_PYTHON,
        simulation=copy.deepcopy(SIMULATION),
        period_template=copy.deepcopy(PERIOD_TEMPLATE),
        parameters=SEARCH_PARAMETERS,
        analysis=copy.deepcopy(ANALYSIS),
        score_getter=lambda result: stable_tail_mean_score(result, observable="m_2"),
        workdir=workdir or Path("/tmp/model-search-backend"),
    )
