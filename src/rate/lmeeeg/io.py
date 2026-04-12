"""lmeEEG IO helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rate.epochs.io import build_epoch_output_paths, read_epochs
from rate.legacy.fooof_mass.io import build_fooof_epoch_feature_paths
from rate.lmeeeg.mass_model import LMEEEGResult
from rate.paths import ensure_parent_dir, resolve_out_dir
from rate.util import write_parquet_dataframe

TRIAL_REGRESSOR_REQUIRED_COLUMNS = (
    "predicted_self_rate",
    "residual_signed",
    "residual_abs",
)
TRIAL_REGRESSOR_OPTIONAL_COLUMNS = ("residual_prediction_mode",)
TRIAL_REGRESSOR_MERGE_KEY_PRIORITY = ("event_id", "subject_id", "task", "run")


@dataclass(frozen=True)
class LMEEEGRunPaths:
    """Resolved input and output paths for one subject-level lmeEEG run."""

    epochs_input_path: Path
    metadata_input_path: Path
    residualised_trials_output_path: Path
    random_effects_output_path: Path
    betas_output_path: Path
    standard_errors_output_path: Path
    t_values_output_path: Path
    residual_variance_output_path: Path
    row_index_output_path: Path
    times_output_path: Path
    channel_names_output_path: Path
    column_names_output_path: Path
    test_predictors_output_path: Path
    selection_output_path: Path
    lmm_diagnostics_output_path: Path
    summary_output_path: Path


@dataclass(frozen=True)
class LMEEEGModelPaths:
    """Resolved output paths for one pooled lmeEEG model fit."""

    residualised_trials_output_path: Path
    random_effects_output_path: Path
    betas_output_path: Path
    standard_errors_output_path: Path
    t_values_output_path: Path
    residual_variance_output_path: Path
    row_index_output_path: Path
    times_output_path: Path
    channel_names_output_path: Path
    column_names_output_path: Path
    test_predictors_output_path: Path
    selection_output_path: Path
    lmm_diagnostics_output_path: Path
    summary_output_path: Path


def _sanitize_model_name(model_name: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(model_name).strip()).strip("._-")
    return token or "model"


def _analysis_root_name(config_section: str) -> str:
    return "lmeeeg" if config_section == "lmeeeg" else str(config_section)


def build_lmeeeg_output_paths(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    config_section: str = "lmeeeg",
    target_name: str | None = None,
    model_name: str,
    epochs_path: Path | None = None,
    metadata_path: Path | None = None,
) -> LMEEEGRunPaths:
    """Build canonical input and output paths for one lmeEEG run."""
    out_dir = resolve_out_dir(config)
    if config_section == "lmeeeg_fooof":
        if target_name is None:
            raise ValueError("`target_name` is required when building target-specific lmeEEG paths.")
        epoch_paths = build_fooof_epoch_feature_paths(config, row, feature_name=target_name)
        model_dir = (
            out_dir
            / _analysis_root_name(config_section)
            / _sanitize_model_name(target_name)
            / str(row["subject_id"])
            / f"task-{row['task']}"
            / f"run-{row['run']}"
            / _sanitize_model_name(model_name)
        )
    else:
        epoch_paths = build_epoch_output_paths(config, row)
        model_dir = (
            out_dir
            / _analysis_root_name(config_section)
            / str(row["subject_id"])
            / f"task-{row['task']}"
            / f"run-{row['run']}"
            / _sanitize_model_name(model_name)
        )
    return LMEEEGRunPaths(
        epochs_input_path=epochs_path or epoch_paths.epochs_output_path,
        metadata_input_path=metadata_path or epoch_paths.metadata_output_path,
        residualised_trials_output_path=model_dir / "residualised_trials.npy",
        random_effects_output_path=model_dir / "random_effects.npy",
        betas_output_path=model_dir / "betas.npy",
        standard_errors_output_path=model_dir / "standard_errors.npy",
        t_values_output_path=model_dir / "t_values.npy",
        residual_variance_output_path=model_dir / "residual_variance.npy",
        row_index_output_path=model_dir / "row_index.npy",
        times_output_path=model_dir / "times.npy",
        channel_names_output_path=model_dir / "channel_names.json",
        column_names_output_path=model_dir / "column_names.json",
        test_predictors_output_path=model_dir / "test_predictors.json",
        selection_output_path=model_dir / "selection.parquet",
        lmm_diagnostics_output_path=model_dir / "lmm_diagnostics.json",
        summary_output_path=model_dir / "summary.json",
    )


def build_lmeeeg_model_output_paths(
    config: Mapping[str, Any],
    *,
    config_section: str = "lmeeeg",
    target_name: str | None = None,
    model_name: str,
) -> LMEEEGModelPaths:
    """Build canonical output paths for one pooled lmeEEG model fit."""
    out_dir = resolve_out_dir(config)
    if config_section == "lmeeeg_fooof":
        if target_name is None:
            raise ValueError("`target_name` is required when building target-specific pooled lmeEEG paths.")
        model_dir = out_dir / _analysis_root_name(config_section) / _sanitize_model_name(target_name) / _sanitize_model_name(model_name)
    else:
        model_dir = out_dir / _analysis_root_name(config_section) / _sanitize_model_name(model_name)
    return LMEEEGModelPaths(
        residualised_trials_output_path=model_dir / "residualised_trials.npy",
        random_effects_output_path=model_dir / "random_effects.npy",
        betas_output_path=model_dir / "betas.npy",
        standard_errors_output_path=model_dir / "standard_errors.npy",
        t_values_output_path=model_dir / "t_values.npy",
        residual_variance_output_path=model_dir / "residual_variance.npy",
        row_index_output_path=model_dir / "row_index.npy",
        times_output_path=model_dir / "times.npy",
        channel_names_output_path=model_dir / "channel_names.json",
        column_names_output_path=model_dir / "column_names.json",
        test_predictors_output_path=model_dir / "test_predictors.json",
        selection_output_path=model_dir / "selection.parquet",
        lmm_diagnostics_output_path=model_dir / "lmm_diagnostics.json",
        summary_output_path=model_dir / "summary.json",
    )


def load_lmeeeg_inputs(epochs_path: Path, metadata_path: Path):
    """Load epochs and metadata for one lmeEEG run."""
    epochs = read_epochs(epochs_path, preload=True)
    metadata_df = pd.read_csv(metadata_path)
    if len(epochs) != len(metadata_df):
        raise ValueError(
            "Epochs length and metadata length must match before lmeEEG filtering: "
            f"{len(epochs)} != {len(metadata_df)}"
        )
    return epochs, metadata_df.reset_index(drop=True)


def load_lmeeeg_inputs_with_trial_regressors(
    epochs_path: Path,
    metadata_path: Path,
    *,
    trial_regressors_path: Path | None = None,
):
    """Load epochs/metadata and optionally augment metadata with trialwise regressors."""
    epochs, metadata_df = load_lmeeeg_inputs(epochs_path, metadata_path)
    if trial_regressors_path is None:
        metadata_df.attrs["metadata_provenance"] = {"trial_regressors": None}
        return epochs, metadata_df

    trial_regressors_df = pd.read_csv(trial_regressors_path)
    merged_metadata_df, merge_info = merge_trial_regressors_into_metadata(
        metadata_df,
        trial_regressors_df,
        source_path=trial_regressors_path,
    )
    merged_metadata_df.attrs["metadata_provenance"] = {"trial_regressors": merge_info}
    return epochs, merged_metadata_df


def merge_trial_regressors_into_metadata(
    metadata_df: pd.DataFrame,
    trial_regressors_df: pd.DataFrame,
    *,
    source_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Left-merge trialwise regressors into epoch metadata without changing row order."""
    merge_keys = _resolve_trial_regressor_merge_keys(metadata_df, trial_regressors_df)
    missing_required = [column for column in TRIAL_REGRESSOR_REQUIRED_COLUMNS if column not in trial_regressors_df.columns]
    if missing_required:
        missing_text = ", ".join(missing_required)
        raise ValueError(f"Trial regressor table is missing required columns: {missing_text}")

    duplicate_mask = trial_regressors_df.duplicated(subset=list(merge_keys), keep=False)
    if duplicate_mask.any():
        raise ValueError(
            "Trial regressor table contains duplicate merge keys for "
            f"{list(merge_keys)}."
        )

    regressor_columns = [
        column_name
        for column_name in [*TRIAL_REGRESSOR_REQUIRED_COLUMNS, *TRIAL_REGRESSOR_OPTIONAL_COLUMNS]
        if column_name in trial_regressors_df.columns
    ]
    conflicting_columns = [column_name for column_name in regressor_columns if column_name in metadata_df.columns]
    if conflicting_columns:
        conflict_text = ", ".join(conflicting_columns)
        raise ValueError(f"Metadata already contains trial regressor columns: {conflict_text}")

    merge_payload = trial_regressors_df.loc[:, [*merge_keys, *regressor_columns]].copy()
    metadata_with_order = metadata_df.reset_index(drop=True).copy()
    metadata_with_order.insert(0, "_metadata_row_order", np.arange(len(metadata_with_order), dtype=int))
    merged_df = metadata_with_order.merge(
        merge_payload,
        how="left",
        on=list(merge_keys),
        sort=False,
        validate="many_to_one",
    )

    if len(merged_df) != len(metadata_df):
        raise ValueError("Trial regressor merge changed the number of metadata rows.")
    if not np.array_equal(
        merged_df["_metadata_row_order"].to_numpy(dtype=int),
        np.arange(len(metadata_df), dtype=int),
    ):
        raise ValueError("Trial regressor merge changed metadata row order.")

    matched_mask = merged_df.loc[:, list(TRIAL_REGRESSOR_REQUIRED_COLUMNS)].notna().any(axis=1)
    merge_info = {
        "source_path": str(source_path),
        "merge_keys": list(merge_keys),
        "matched_rows": int(matched_mask.sum()),
        "unmatched_rows": int((~matched_mask).sum()),
        "merged_columns": regressor_columns,
    }
    merged_df = merged_df.drop(columns="_metadata_row_order").reset_index(drop=True)
    return merged_df, merge_info


def _resolve_trial_regressor_merge_keys(
    metadata_df: pd.DataFrame,
    trial_regressors_df: pd.DataFrame,
) -> tuple[str, ...]:
    shared_keys = [
        column_name
        for column_name in TRIAL_REGRESSOR_MERGE_KEY_PRIORITY
        if column_name in metadata_df.columns and column_name in trial_regressors_df.columns
    ]
    if "event_id" in shared_keys:
        return ("event_id",)
    if len(shared_keys) >= 2:
        return tuple(shared_keys)
    raise ValueError(
        "Unable to resolve merge keys for trial regressors. Expected shared identifiers such as "
        "`event_id` or a combination of `subject_id`, `task`, and `run`."
    )


def save_lmeeeg_result(result: LMEEEGResult, output_paths: LMEEEGRunPaths | LMEEEGModelPaths) -> dict[str, Any]:
    """Write primary lmeEEG artifacts to disk."""
    ensure_parent_dir(output_paths.summary_output_path)
    np.save(output_paths.residualised_trials_output_path, result.residualised_data)
    np.save(output_paths.random_effects_output_path, result.random_effects_contribution)
    np.save(output_paths.betas_output_path, result.betas)
    np.save(output_paths.row_index_output_path, result.row_index)
    np.save(output_paths.times_output_path, result.times)
    np.save(output_paths.standard_errors_output_path, result.standard_errors if result.standard_errors is not None else np.asarray([], dtype=float))
    np.save(output_paths.t_values_output_path, result.t_values if result.t_values is not None else np.asarray([], dtype=float))
    np.save(output_paths.residual_variance_output_path, result.residual_variance if result.residual_variance is not None else np.asarray([], dtype=float))

    output_paths.channel_names_output_path.write_text(json.dumps(result.channel_names, indent=2) + "\n", encoding="utf-8")
    output_paths.column_names_output_path.write_text(json.dumps(result.column_names, indent=2) + "\n", encoding="utf-8")
    output_paths.test_predictors_output_path.write_text(json.dumps(result.test_predictors, indent=2) + "\n", encoding="utf-8")
    output_paths.lmm_diagnostics_output_path.write_text(json.dumps(result.lmm_diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    selection_df = result.design_result.filtered_metadata.copy()
    selection_df.insert(0, "row_index", result.row_index.tolist())
    write_parquet_dataframe(selection_df.reset_index(drop=True), output_paths.selection_output_path)

    summary = {
        "status": "ok",
        "backend": result.backend,
        "random_effect_structure": result.random_effect_structure,
        "group_column": result.group_column,
        "model_formula": result.design_result.model_spec.formula,
        "predictor_of_interest": result.test_predictors[0] if len(result.test_predictors) == 1 else list(result.test_predictors),
        "predictor_variant": _classify_predictor_variant(result.test_predictors),
        "control_terms": _extract_control_terms(result),
        "preprocessing_source": result.metadata_provenance,
        "n_trials_input": result.n_trials_input,
        "n_trials_used": result.n_trials_used,
        "n_channels": len(result.channel_names),
        "n_times": int(result.times.shape[0]),
        "n_targets": result.n_targets,
        "n_lmm_attempted": result.n_lmm_attempted,
        "n_lmm_failed": result.n_lmm_failed,
        "failed_target_indices": result.failed_target_indices,
        "design_matrix_shape": list(result.design_result.design_matrix.shape),
        "betas_shape": list(result.betas.shape),
        "drop_report": dict(result.design_result.drop_report),
        "method_note": (
            "This lmeEEG branch estimates and subtracts fitted random-intercept contributions per target, "
            "then runs OLS on residualised trial-level EEG. It is an approximation workflow, not full mixed-model inference."
        ),
    }
    output_paths.summary_output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def read_lmeeeg_outputs(summary_path: Path) -> dict[str, Any]:
    """Read a saved lmeEEG summary payload."""
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _extract_control_terms(result: LMEEEGResult) -> list[str]:
    term_names = [term_name for term_name in result.design_result.term_name_slices if term_name != "Intercept"]
    return [term_name for term_name in term_names if term_name not in set(result.test_predictors)]


def _classify_predictor_variant(test_predictors: list[str]) -> str:
    if len(test_predictors) != 1:
        return "custom"
    predictor_name = str(test_predictors[0])
    if predictor_name == "residual_abs":
        return "absolute_residual"
    if predictor_name == "residual_signed":
        return "signed_residual"
    if predictor_name == "other_rate":
        return "other_rate"
    return "custom"
