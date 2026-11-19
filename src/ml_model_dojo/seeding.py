"""Seeding helpers. Never use the global numpy random state."""
from __future__ import annotations

import numpy as np

SEEDS = (1, 2, 3, 4, 5)

# Salts keep the random streams for different purposes independent.
SALT_DATA = 101
SALT_MODEL = 202
SALT_SPLIT = 303


def make_rng(seed: int, salt: int = 0) -> np.random.Generator:
    """Return a generator that depends only on the seed and the salt."""
    return np.random.default_rng([int(seed), int(salt)])


def model_seed(seed: int) -> int:
    """Integer seed for estimators (scikit-learn, XGBoost) derived from the app seed."""
    return int(make_rng(seed, SALT_MODEL).integers(0, 2**31 - 1))
