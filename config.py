from __future__ import annotations

import numpy as np


RANDOM_SEED = 7
BOUNDS = np.array([[-4.0, 4.0], [-4.0, 4.0]])
N_INITIAL_SAMPLES = 20
N_BO_ITERATIONS = 50
GRID_RESOLUTION = 120
UCB_BETA = 5.0
GP_LENGTH_SCALE = 0.8
GP_NU = 1.5


rng = np.random.default_rng(RANDOM_SEED)
