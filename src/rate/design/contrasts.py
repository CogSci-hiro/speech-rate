"""Minimal contrast helpers for predictor-to-column mapping."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from rate.design.matrices import DesignMatrixResult


def get_predictor_column_indices(
    predictor: str,
    *,
    column_names: Sequence[str],
    term_name_slices: Mapping[str, slice] | None = None,
) -> list[int]:
    """Return design-column indices corresponding to a tested predictor or term."""
    if term_name_slices and predictor in term_name_slices:
        term_slice = term_name_slices[predictor]
        return list(range(term_slice.start, term_slice.stop))

    matches = [index for index, column_name in enumerate(column_names) if column_name == predictor]
    if matches:
        return matches

    raise KeyError(f"Predictor `{predictor}` is absent from design columns.")


def build_contrast_vector(
    predictor: str,
    *,
    column_names: Sequence[str],
    term_name_slices: Mapping[str, slice] | None = None,
    weight: float = 1.0,
) -> np.ndarray:
    """Build a simple predictor contrast vector aligned to design columns.

    Terms spanning multiple columns, such as categorical expansions, receive the
    same non-zero weight in each matched column.
    """
    indices = get_predictor_column_indices(
        predictor,
        column_names=column_names,
        term_name_slices=term_name_slices,
    )
    contrast = np.zeros((len(column_names),), dtype=float)
    contrast[indices] = weight
    return contrast


def get_test_predictor_contrasts(result: DesignMatrixResult) -> dict[str, np.ndarray]:
    """Build one simple contrast vector per tested predictor."""
    return {
        predictor: build_contrast_vector(
            predictor,
            column_names=result.column_names,
            term_name_slices=result.term_name_slices,
        )
        for predictor in result.test_predictors
    }


def build_contrasts(result: DesignMatrixResult) -> dict[str, np.ndarray]:
    """Backward-compatible wrapper returning simple tested-predictor contrasts."""
    return get_test_predictor_contrasts(result)
