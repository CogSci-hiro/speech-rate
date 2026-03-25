"""Residualisation helpers for the lmeEEG approximation branch."""

from __future__ import annotations

import numpy as np


def residualise_random_effects(target_values: np.ndarray, random_effect_contribution: np.ndarray) -> np.ndarray:
    """Subtract fitted random-effects contribution from the original target."""
    y = np.asarray(target_values, dtype=float)
    contribution = np.asarray(random_effect_contribution, dtype=float)
    if y.shape != contribution.shape:
        raise ValueError("Random-effect contribution must have the same shape as the target values.")
    return y - contribution
