"""Design integration and GLM helpers for FOOOF-derived features."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rate.design import DesignMatrixResult, build_design_matrix, load_model_spec


@dataclass(frozen=True)
class FooofMassGLMResult:
    """Subject-level GLM outputs for one FOOOF-derived outcome."""

    outcome_name: str
    betas: np.ndarray
    standard_errors: np.ndarray
    t_values: np.ndarray
    valid_counts: np.ndarray
    channel_names: list[str]
    window_centers_s: np.ndarray
    column_names: list[str]
    test_predictors: list[str]
    design_result: DesignMatrixResult
    target_shape: tuple[int, int]


def prepare_fooof_mass_design(
    metadata_df: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    model_name: str | None = None,
) -> DesignMatrixResult:
    """Build the predictor design matrix for a FOOOF-derived outcome model."""
    model_spec = load_model_spec(config, section="fooof", model_name=model_name)
    design_spec = _make_rhs_only_model_spec(model_spec)
    return build_design_matrix(metadata_df, model_spec=design_spec)


def fit_fooof_mass_glm(
    feature_tensor: np.ndarray,
    metadata_df: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    model_name: str,
    channel_names: list[str],
    window_centers_s: np.ndarray,
) -> FooofMassGLMResult:
    """Fit OLS for one FOOOF-derived feature outcome.

    Missing-data strategy
    ---------------------
    The design layer determines the base set of eligible trials once. For each
    channel-window target, the GLM then uses the subset of those same trials
    whose feature value is finite. This keeps the predictor matrix definition
    fixed while handling target-specific FOOOF failures explicitly.
    """
    if feature_tensor.ndim != 3:
        raise ValueError("Expected feature tensor with shape (n_epochs, n_channels, n_windows).")

    model_spec = load_model_spec(config, section="fooof", model_name=model_name)
    design_result = prepare_fooof_mass_design(metadata_df, config, model_name=model_name)
    row_positions = design_result.row_index.to_numpy(dtype=int, copy=True)
    selected_features = np.asarray(feature_tensor[row_positions, :, :], dtype=float)
    if selected_features.shape[0] != design_result.design_matrix.shape[0]:
        raise ValueError("FOOOF feature rows and design rows do not align after design filtering.")

    X = np.asarray(design_result.design_matrix, dtype=float)
    n_epochs_used, n_predictors = X.shape
    n_channels = selected_features.shape[1]
    n_windows = selected_features.shape[2]
    Y = selected_features.reshape(n_epochs_used, n_channels * n_windows)

    betas = np.full((n_predictors, Y.shape[1]), np.nan, dtype=float)
    standard_errors = np.full((n_predictors, Y.shape[1]), np.nan, dtype=float)
    t_values = np.full((n_predictors, Y.shape[1]), np.nan, dtype=float)
    valid_counts = np.zeros((Y.shape[1],), dtype=int)

    for target_index in range(Y.shape[1]):
        outcome_values = Y[:, target_index]
        valid_mask = np.isfinite(outcome_values)
        valid_counts[target_index] = int(valid_mask.sum())
        if valid_counts[target_index] <= n_predictors:
            continue

        X_valid = X[valid_mask]
        y_valid = outcome_values[valid_mask]
        beta_vector, _, rank, _ = np.linalg.lstsq(X_valid, y_valid[:, np.newaxis], rcond=None)
        beta_vector = beta_vector[:, 0]
        betas[:, target_index] = beta_vector

        dof = int(len(y_valid) - rank)
        if dof <= 0:
            continue

        residuals = y_valid - (X_valid @ beta_vector)
        sigma2 = float(np.sum(residuals * residuals) / dof)
        xtx_pinv = np.linalg.pinv(X_valid.T @ X_valid)
        se_vector = np.sqrt(np.clip(np.diag(xtx_pinv), a_min=0.0, a_max=None) * sigma2)
        standard_errors[:, target_index] = se_vector
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values[:, target_index] = np.divide(
                beta_vector,
                se_vector,
                out=np.full_like(beta_vector, np.nan, dtype=float),
                where=se_vector > 0,
            )

    return FooofMassGLMResult(
        outcome_name=str(model_spec.outcome or model_name),
        betas=betas.reshape(n_predictors, n_channels, n_windows),
        standard_errors=standard_errors.reshape(n_predictors, n_channels, n_windows),
        t_values=t_values.reshape(n_predictors, n_channels, n_windows),
        valid_counts=valid_counts.reshape(n_channels, n_windows),
        channel_names=list(channel_names),
        window_centers_s=np.asarray(window_centers_s, dtype=float),
        column_names=list(design_result.column_names),
        test_predictors=list(design_result.test_predictors),
        design_result=design_result,
        target_shape=(n_channels, n_windows),
    )


def _make_rhs_only_model_spec(model_spec):
    if "~" not in model_spec.formula:
        return model_spec
    lhs, rhs = model_spec.formula.split("~", maxsplit=1)
    if not lhs.strip():
        return model_spec
    return replace(model_spec, formula=f"~ {rhs.strip()}", outcome=None)
