"""Mass-univariate mixed-model fitting for the lmeEEG approximation branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from lmeeeg import fit_lmm_mass_univariate

from rate.lmeeeg.trial_data import LMEEEGTrialData, resolve_random_effect_group, resolve_random_effect_structure


@dataclass(frozen=True)
class LMEEEGResult:
    """Residualised-data and OLS outputs for one lmeEEG subject run."""

    residualised_data: np.ndarray
    random_effects_contribution: np.ndarray
    betas: np.ndarray
    standard_errors: np.ndarray | None
    t_values: np.ndarray | None
    residual_variance: np.ndarray | None
    column_names: list[str]
    test_predictors: list[str]
    row_index: np.ndarray
    channel_names: list[str]
    times: np.ndarray
    backend: str
    group_column: str
    random_effect_structure: str
    n_trials_input: int
    n_trials_used: int
    n_targets: int
    n_lmm_attempted: int
    n_lmm_failed: int
    failed_target_indices: list[int]
    lmm_diagnostics: dict[str, Any]
    design_result: Any
    ols_degrees_of_freedom: int
    ols_rank: int


def fit_lmeeeg_mass_model(
    trial_data: LMEEEGTrialData,
    config: Mapping[str, Any],
    *,
    config_section: str = "lmeeeg",
    model_name: str | None = None,
) -> LMEEEGResult:
    """Run the lmeEEG approximation through the external `lmeeeg` package."""
    backend = str((config.get(config_section) or {}).get("backend", "lmeeeg")).lower()
    if backend not in {"python", "lmeeeg"}:
        raise ValueError(
            f"Unsupported lmeEEG backend `{backend}`. Supported values are `lmeeeg` and legacy alias `python`."
        )

    group_column = resolve_random_effect_group(config, config_section=config_section, model_name=model_name)
    random_effect_structure = resolve_random_effect_structure(config, config_section=config_section, model_name=model_name)

    eeg_data = np.asarray(trial_data.eeg_data, dtype=float)
    n_trials_used, n_channels, n_times = eeg_data.shape
    fit_result = fit_lmeeeg_backend_result(trial_data, config, config_section=config_section, model_name=model_name)

    converged_mask = _converged_feature_mask(
        fit_result.feature_diagnostics,
        n_channels=n_channels,
        n_times=n_times,
    )
    failed_target_indices = np.flatnonzero(~converged_mask.reshape(-1)).tolist()

    column_names = list(trial_data.design_result.column_names)
    betas = _stack_feature_maps(fit_result.ols_betas, column_names)
    t_values = _stack_feature_maps(fit_result.ols_t_values, column_names)
    residual_variance = np.asarray(fit_result.ols_residual_variance, dtype=float)
    ols_rank = int(trial_data.design_result.design_matrix.shape[1])
    ols_dof = int(n_trials_used - ols_rank)
    diagnostics = _build_lmm_diagnostics(fit_result.feature_diagnostics)

    return LMEEEGResult(
        residualised_data=np.asarray(fit_result.marginal_eeg, dtype=float),
        random_effects_contribution=np.asarray(fit_result.fitted_random_effects, dtype=float),
        betas=betas,
        standard_errors=None,
        t_values=t_values,
        residual_variance=residual_variance,
        column_names=column_names,
        test_predictors=list(trial_data.design_result.test_predictors),
        row_index=np.asarray(trial_data.row_index, dtype=int),
        channel_names=list(trial_data.channel_names),
        times=np.asarray(trial_data.times, dtype=float),
        backend="lmeeeg",
        group_column=group_column,
        random_effect_structure=random_effect_structure,
        n_trials_input=int(len(trial_data.design_result.row_mask)),
        n_trials_used=n_trials_used,
        n_targets=int(n_channels * n_times),
        n_lmm_attempted=int(n_channels * n_times),
        n_lmm_failed=len(failed_target_indices),
        failed_target_indices=failed_target_indices,
        lmm_diagnostics=diagnostics,
        design_result=trial_data.design_result,
        ols_degrees_of_freedom=int(ols_dof),
        ols_rank=int(ols_rank),
    )


def fit_lmeeeg_backend_result(
    trial_data: LMEEEGTrialData,
    config: Mapping[str, Any],
    *,
    config_section: str = "lmeeeg",
    model_name: str | None = None,
):
    """Return the package-native `lmeeeg` fit result for a prepared trial table."""
    group_column = resolve_random_effect_group(config, config_section=config_section, model_name=model_name)
    metadata = trial_data.trial_metadata.reset_index(drop=True).copy()
    return fit_lmm_mass_univariate(
        eeg=np.asarray(trial_data.eeg_data, dtype=float),
        metadata=metadata,
        formula=_build_lmeeeg_formula(trial_data.design_result.model_spec.formula, group_column),
        variable_types=_infer_variable_types(metadata, group_column=group_column),
        fit_intercept="Intercept" in trial_data.design_result.column_names,
    )


def _build_lmeeeg_formula(base_formula: str, group_column: str) -> str:
    if "~" not in base_formula:
        raise ValueError(f"Invalid lmeEEG formula `{base_formula}`.")
    _lhs, rhs = base_formula.split("~", maxsplit=1)
    rhs_text = rhs.strip()
    if not rhs_text:
        raise ValueError(f"Invalid lmeEEG formula `{base_formula}`.")
    return f"y ~ {rhs_text} + (1|{group_column})"


def _infer_variable_types(metadata: pd.DataFrame, *, group_column: str) -> dict[str, str]:
    variable_types: dict[str, str] = {}
    for column_name in metadata.columns:
        if column_name == group_column:
            variable_types[column_name] = "group"
        elif pd.api.types.is_numeric_dtype(metadata[column_name]):
            variable_types[column_name] = "numeric"
        else:
            variable_types[column_name] = "categorical"
    if group_column not in variable_types:
        raise ValueError(f"Configured lmeEEG random-effect group column `{group_column}` is missing from metadata.")
    return variable_types


def _stack_feature_maps(feature_maps: Mapping[str, np.ndarray], column_names: list[str]) -> np.ndarray:
    missing = [column_name for column_name in column_names if column_name not in feature_maps]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"External lmeeeg result is missing expected fixed-effect maps: {missing_text}")
    return np.stack([np.asarray(feature_maps[column_name], dtype=float) for column_name in column_names], axis=0)


def _converged_feature_mask(feature_diagnostics: pd.DataFrame, *, n_channels: int, n_times: int) -> np.ndarray:
    if len(feature_diagnostics) != n_channels * n_times:
        raise ValueError(
            "External lmeeeg diagnostics rows do not match EEG targets: "
            f"{len(feature_diagnostics)} != {n_channels * n_times}"
        )
    return feature_diagnostics["converged"].to_numpy(dtype=bool).reshape(n_channels, n_times)


def _build_lmm_diagnostics(feature_diagnostics: pd.DataFrame) -> dict[str, Any]:
    converged_targets = int(feature_diagnostics["converged"].sum())
    failed_targets = int(len(feature_diagnostics) - converged_targets)
    boundary_targets = int(feature_diagnostics["boundary_warning"].sum())
    return {
        "converged_targets": converged_targets,
        "failed_targets": failed_targets,
        "boundary_warning_targets": boundary_targets,
        "valid_targets_for_ols": converged_targets,
    }
