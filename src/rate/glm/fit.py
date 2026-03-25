"""Subject-level mass-univariate GLM fitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rate.design import DesignMatrixResult, build_design_matrix


@dataclass(frozen=True)
class GLMFitResult:
    """Container holding subject-level mass-univariate GLM outputs."""

    betas: np.ndarray
    standard_errors: np.ndarray | None
    t_values: np.ndarray | None
    residual_variance: np.ndarray | None
    degrees_of_freedom: int
    rank: int
    channel_names: list[str]
    times: np.ndarray
    column_names: list[str]
    test_predictors: list[str]
    row_index: np.ndarray
    n_epochs_input: int
    n_epochs_used: int
    design_matrix_shape: tuple[int, int]
    target_matrix_shape: tuple[int, int]
    design_result: DesignMatrixResult


def prepare_glm_design(
    metadata_df: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    model_name: str | None = None,
) -> DesignMatrixResult:
    """Build a subject-level GLM design matrix from epoch metadata."""
    return build_design_matrix(metadata_df, config=config, section="glm", model_name=model_name)


def reshape_targets(eeg_data: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Reshape epochs data into a 2D target matrix for vectorized OLS.

    Parameters
    ----------
    eeg_data
        EEG data with shape ``(n_epochs, n_channels, n_times)``.

    Returns
    -------
    target_matrix, original_shape
        Two-dimensional targets with shape ``(n_epochs, n_channels * n_times)``
        plus the original 3D shape for later reconstruction.
    """
    if eeg_data.ndim != 3:
        raise ValueError("Expected EEG data with shape (n_epochs, n_channels, n_times).")
    n_epochs, n_channels, n_times = eeg_data.shape
    return eeg_data.reshape(n_epochs, n_channels * n_times), (n_epochs, n_channels, n_times)


def fit_mass_univariate_ols(
    design_matrix: np.ndarray,
    target_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, int, int]:
    """Fit one OLS design matrix against many targets at once.

    Parameters
    ----------
    design_matrix
        Matrix ``X`` with shape ``(n_epochs, n_predictors)``.
    target_matrix
        Matrix ``Y`` with shape ``(n_epochs, n_targets)``.

    Returns
    -------
    tuple
        ``(betas, standard_errors, t_values, residual_variance, dof, rank)``
        where betas have shape ``(n_predictors, n_targets)``.
    """
    X = np.asarray(design_matrix, dtype=float)
    Y = np.asarray(target_matrix, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Design and target matrices must both be 2D.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Design and target matrices must have the same number of rows.")
    if X.shape[0] == 0:
        raise ValueError("Cannot fit GLM with zero rows.")
    if np.isnan(X).any():
        raise ValueError("Design matrix contains NaN values after design construction.")
    if np.isnan(Y).any():
        raise ValueError("EEG target matrix contains NaN values.")

    betas, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    fitted = X @ betas
    residuals = Y - fitted
    degrees_of_freedom = int(X.shape[0] - rank)

    if degrees_of_freedom <= 0:
        return betas, None, None, None, degrees_of_freedom, int(rank)

    residual_sum_squares = np.sum(residuals * residuals, axis=0)
    residual_variance = residual_sum_squares / float(degrees_of_freedom)
    xtx_pinv = np.linalg.pinv(X.T @ X)
    beta_variance_scale = np.clip(np.diag(xtx_pinv), a_min=0.0, a_max=None)
    standard_errors = np.sqrt(beta_variance_scale[:, np.newaxis] * residual_variance[np.newaxis, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.divide(
            betas,
            standard_errors,
            out=np.zeros_like(betas),
            where=standard_errors > 0,
        )
    return betas, standard_errors, t_values, residual_variance, degrees_of_freedom, int(rank)


def fit_subject_glm(epochs, design_result: DesignMatrixResult) -> GLMFitResult:
    """Fit a mass-univariate OLS model for one subject/run epochs object.

    Parameters
    ----------
    epochs
        MNE ``Epochs`` object aligned to the metadata passed to the design layer.
    design_result
        Shared design-layer result containing the final design matrix and row
        indices into the original metadata table.

    Returns
    -------
    GLMFitResult
        Subject-level beta maps and optional inferential summaries.
    """
    n_epochs_input = len(epochs)
    row_index = _coerce_row_positions(design_result.row_index, n_epochs_input=n_epochs_input)

    eeg_data = epochs.get_data(copy=True)
    if len(design_result.row_index) != len(design_result.filtered_metadata):
        raise ValueError("Design result row index and filtered metadata length must match.")

    selected_eeg = eeg_data[row_index, :, :]
    target_matrix, (_, n_channels, n_times) = reshape_targets(selected_eeg)
    design_matrix = np.asarray(design_result.design_matrix, dtype=float)

    if target_matrix.shape[0] != design_matrix.shape[0]:
        raise ValueError(
            "EEG rows and design rows do not align after subsetting: "
            f"{target_matrix.shape[0]} != {design_matrix.shape[0]}"
        )
    if design_matrix.shape[0] == 0:
        raise ValueError("No epochs remain after design filtering.")

    betas_2d, standard_errors_2d, t_values_2d, residual_variance, dof, rank = fit_mass_univariate_ols(
        design_matrix,
        target_matrix,
    )
    betas = betas_2d.reshape(design_matrix.shape[1], n_channels, n_times)
    standard_errors = None
    if standard_errors_2d is not None:
        standard_errors = standard_errors_2d.reshape(design_matrix.shape[1], n_channels, n_times)
    t_values = None
    if t_values_2d is not None:
        t_values = t_values_2d.reshape(design_matrix.shape[1], n_channels, n_times)
    residual_variance_map = None
    if residual_variance is not None:
        residual_variance_map = residual_variance.reshape(n_channels, n_times)

    return GLMFitResult(
        betas=betas,
        standard_errors=standard_errors,
        t_values=t_values,
        residual_variance=residual_variance_map,
        degrees_of_freedom=dof,
        rank=rank,
        channel_names=list(epochs.ch_names),
        times=np.asarray(epochs.times, dtype=float),
        column_names=list(design_result.column_names),
        test_predictors=list(design_result.test_predictors),
        row_index=row_index,
        n_epochs_input=n_epochs_input,
        n_epochs_used=int(design_matrix.shape[0]),
        design_matrix_shape=tuple(design_matrix.shape),
        target_matrix_shape=tuple(target_matrix.shape),
        design_result=design_result,
    )


def _coerce_row_positions(row_index: pd.Index, *, n_epochs_input: int) -> np.ndarray:
    try:
        positions = row_index.to_numpy(dtype=int, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Design row indices must be integer row positions for EEG alignment.") from exc
    if np.any(positions < 0) or np.any(positions >= n_epochs_input):
        raise ValueError("Design row indices fall outside the available epochs range.")
    return positions
