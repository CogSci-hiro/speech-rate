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
from rate.lmeeeg.mass_model import LMEEEGResult
from rate.paths import ensure_parent_dir, resolve_out_dir
from rate.util import write_parquet_dataframe


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


def build_lmeeeg_output_paths(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    model_name: str,
    epochs_path: Path | None = None,
    metadata_path: Path | None = None,
) -> LMEEEGRunPaths:
    """Build canonical input and output paths for one lmeEEG run."""
    out_dir = resolve_out_dir(config)
    epoch_paths = build_epoch_output_paths(config, row)
    model_dir = (
        out_dir
        / "lmeeeg"
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
    model_name: str,
) -> LMEEEGModelPaths:
    """Build canonical output paths for one pooled lmeEEG model fit."""
    out_dir = resolve_out_dir(config)
    model_dir = out_dir / "lmeeeg" / _sanitize_model_name(model_name)
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
