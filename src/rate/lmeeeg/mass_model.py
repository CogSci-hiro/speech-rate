"""Mass-univariate mixed-model fitting for the lmeEEG approximation branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM

from rate.glm.fit import GLMFitResult, fit_mass_univariate_ols
from rate.lmeeeg.residualise import residualise_random_effects
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
    model_name: str | None = None,
) -> LMEEEGResult:
    """Run the lmeEEG approximation: random-intercept removal plus residualised OLS."""
    backend = str((config.get("lmeeeg") or {}).get("backend", "python")).lower()
    if backend != "python":
        raise ValueError(f"Unsupported lmeEEG backend `{backend}`. Only `python` is implemented.")

    group_column = resolve_random_effect_group(config, model_name=model_name)
    random_effect_structure = resolve_random_effect_structure(config, model_name=model_name)

    eeg_data = np.asarray(trial_data.eeg_data, dtype=float)
    n_trials_used, n_channels, n_times = eeg_data.shape
    Y = eeg_data.reshape(n_trials_used, n_channels * n_times)
    X = np.asarray(trial_data.design_result.design_matrix, dtype=float)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("lmeEEG design rows and EEG trial rows do not align.")

    residualised_targets = np.full_like(Y, np.nan, dtype=float)
    random_effects_targets = np.full_like(Y, np.nan, dtype=float)
    failed_target_indices: list[int] = []
    converged_count = 0
    singular_count = 0

    for target_index in range(Y.shape[1]):
        fit_result = _fit_single_target_mixedlm(
            X,
            y=Y[:, target_index],
            groups=trial_data.trial_metadata[group_column],
        )
        if not fit_result["success"]:
            failed_target_indices.append(target_index)
            if fit_result["singular"]:
                singular_count += 1
            continue

        random_contribution = fit_result["random_effect_contribution"]
        random_effects_targets[:, target_index] = random_contribution
        residualised_targets[:, target_index] = residualise_random_effects(Y[:, target_index], random_contribution)
        converged_count += 1

    valid_target_mask = np.all(np.isfinite(residualised_targets), axis=0)
    betas = np.full((X.shape[1], Y.shape[1]), np.nan, dtype=float)
    standard_errors = np.full((X.shape[1], Y.shape[1]), np.nan, dtype=float)
    t_values = np.full((X.shape[1], Y.shape[1]), np.nan, dtype=float)
    residual_variance = np.full((Y.shape[1],), np.nan, dtype=float)
    ols_dof = 0
    ols_rank = 0
    if np.any(valid_target_mask):
        (
            beta_valid,
            se_valid,
            t_valid,
            sigma_valid,
            ols_dof,
            ols_rank,
        ) = fit_mass_univariate_ols(X, residualised_targets[:, valid_target_mask])
        betas[:, valid_target_mask] = beta_valid
        if se_valid is not None:
            standard_errors[:, valid_target_mask] = se_valid
        if t_valid is not None:
            t_values[:, valid_target_mask] = t_valid
        if sigma_valid is not None:
            residual_variance[valid_target_mask] = sigma_valid

    return LMEEEGResult(
        residualised_data=residualised_targets.reshape(n_trials_used, n_channels, n_times),
        random_effects_contribution=random_effects_targets.reshape(n_trials_used, n_channels, n_times),
        betas=betas.reshape(X.shape[1], n_channels, n_times),
        standard_errors=None if not np.any(np.isfinite(standard_errors)) else standard_errors.reshape(X.shape[1], n_channels, n_times),
        t_values=None if not np.any(np.isfinite(t_values)) else t_values.reshape(X.shape[1], n_channels, n_times),
        residual_variance=None if not np.any(np.isfinite(residual_variance)) else residual_variance.reshape(n_channels, n_times),
        column_names=list(trial_data.design_result.column_names),
        test_predictors=list(trial_data.design_result.test_predictors),
        row_index=np.asarray(trial_data.row_index, dtype=int),
        channel_names=list(trial_data.channel_names),
        times=np.asarray(trial_data.times, dtype=float),
        backend=backend,
        group_column=group_column,
        random_effect_structure=random_effect_structure,
        n_trials_input=int(len(trial_data.design_result.row_mask)),
        n_trials_used=n_trials_used,
        n_targets=int(Y.shape[1]),
        n_lmm_attempted=int(Y.shape[1]),
        n_lmm_failed=len(failed_target_indices),
        failed_target_indices=failed_target_indices,
        lmm_diagnostics={
            "converged_targets": converged_count,
            "failed_targets": len(failed_target_indices),
            "singular_targets": singular_count,
            "valid_targets_for_ols": int(valid_target_mask.sum()),
        },
        design_result=trial_data.design_result,
        ols_degrees_of_freedom=int(ols_dof),
        ols_rank=int(ols_rank),
    )


def _fit_single_target_mixedlm(
    design_matrix: np.ndarray,
    *,
    y: np.ndarray,
    groups,
) -> dict[str, Any]:
    y_array = np.asarray(y, dtype=float)
    X = np.asarray(design_matrix, dtype=float)
    group_values = np.asarray(groups)
    if np.isnan(y_array).any():
        return {"success": False, "singular": False}
    if np.unique(group_values).size < 2:
        return {"success": False, "singular": True}

    try:
        model = MixedLM(
            endog=y_array,
            exog=X,
            groups=group_values,
            exog_re=np.ones((len(y_array), 1), dtype=float),
        )
        fitted = model.fit(reml=False, method="lbfgs", disp=False)
    except Exception:
        return {"success": False, "singular": False}

    converged = bool(getattr(fitted, "converged", False))
    random_effect_contribution = _extract_random_intercept_contribution(fitted, group_values)
    singular = bool(np.allclose(random_effect_contribution, 0.0) and not np.isclose(float(np.var(y_array)), 0.0))
    if not converged:
        return {"success": False, "singular": singular}

    return {
        "success": True,
        "singular": singular,
        "random_effect_contribution": random_effect_contribution,
    }


def _extract_random_intercept_contribution(fitted, groups: np.ndarray) -> np.ndarray:
    contributions = np.zeros((len(groups),), dtype=float)
    random_effects = getattr(fitted, "random_effects", {})
    for row_index, group_value in enumerate(groups):
        effect = random_effects.get(group_value)
        if effect is None:
            contributions[row_index] = 0.0
            continue
        if hasattr(effect, "iloc"):
            contributions[row_index] = float(effect.iloc[0])
        elif isinstance(effect, Mapping):
            contributions[row_index] = float(next(iter(effect.values())))
        else:
            contributions[row_index] = float(np.asarray(effect).reshape(-1)[0])
    return contributions
