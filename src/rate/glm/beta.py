"""Helpers for extracting named predictor beta maps from GLM fits."""

from __future__ import annotations

import numpy as np

from rate.design import get_predictor_column_indices
from rate.glm.fit import GLMFitResult


def get_predictor_column_names(result: GLMFitResult, predictor: str) -> list[str]:
    """Resolve a tested predictor or term into concrete design-column names."""
    column_indices = get_predictor_column_indices(
        predictor,
        column_names=result.column_names,
        term_name_slices=result.design_result.term_name_slices,
    )
    return [result.column_names[index] for index in column_indices]


def extract_predictor_betas(result: GLMFitResult, predictor: str) -> dict[str, np.ndarray]:
    """Extract beta maps for one named predictor or formula term.

    Parameters
    ----------
    result
        Fitted subject-level GLM result.
    predictor
        Predictor or term name as declared in ``test_predictors``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping from concrete design-column name to beta map with shape
        ``(n_channels, n_times)``.
    """
    column_indices = get_predictor_column_indices(
        predictor,
        column_names=result.column_names,
        term_name_slices=result.design_result.term_name_slices,
    )
    return {
        result.column_names[index]: np.asarray(result.betas[index], dtype=float)
        for index in column_indices
    }


def extract_beta_maps(result: GLMFitResult, predictors: list[str] | None = None) -> dict[str, np.ndarray]:
    """Extract concrete beta maps for tested predictors.

    Returns a flat mapping keyed by actual design-column names so downstream
    stacking stays unambiguous when terms expand into multiple columns.
    """
    selected_predictors = predictors or result.test_predictors
    extracted: dict[str, np.ndarray] = {}
    for predictor in selected_predictors:
        extracted.update(extract_predictor_betas(result, predictor))
    return extracted


def build_predictor_column_map(result: GLMFitResult) -> dict[str, list[str]]:
    """Return tested predictors mapped to concrete design-column names."""
    return {predictor: get_predictor_column_names(result, predictor) for predictor in result.test_predictors}
