"""IO helpers for subject-level GLM artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rate.design import get_predictor_column_indices
from rate.epochs.io import build_epoch_output_paths, read_epochs
from rate.glm.beta import build_predictor_column_map, extract_beta_maps
from rate.glm.fit import GLMFitResult
from rate.paths import ensure_parent_dir, resolve_out_dir
from rate.util import write_parquet_dataframe


@dataclass(frozen=True)
class GLMRunPaths:
    """Resolved input and output paths for one subject-level GLM fit."""

    epochs_input_path: Path
    metadata_input_path: Path
    betas_output_path: Path
    standard_errors_output_path: Path
    t_values_output_path: Path
    residual_variance_output_path: Path
    design_matrix_output_path: Path
    row_index_output_path: Path
    times_output_path: Path
    channel_names_output_path: Path
    column_names_output_path: Path
    test_predictors_output_path: Path
    predictor_columns_output_path: Path
    predictor_betas_output_path: Path
    selection_output_path: Path
    summary_output_path: Path


def _sanitize_model_name(model_name: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(model_name).strip()).strip("._-")
    return token or "model"


def build_glm_output_paths(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    model_name: str,
    epochs_path: Path | None = None,
    metadata_path: Path | None = None,
) -> GLMRunPaths:
    """Build canonical input and output paths for one subject-level GLM fit."""
    out_dir = resolve_out_dir(config)
    epoch_paths = build_epoch_output_paths(config, row)
    model_dir = out_dir / "glm" / str(row["subject_id"]) / _sanitize_model_name(model_name)
    return GLMRunPaths(
        epochs_input_path=epochs_path or epoch_paths.epochs_output_path,
        metadata_input_path=metadata_path or epoch_paths.metadata_output_path,
        betas_output_path=model_dir / "betas.npy",
        standard_errors_output_path=model_dir / "standard_errors.npy",
        t_values_output_path=model_dir / "t_values.npy",
        residual_variance_output_path=model_dir / "residual_variance.npy",
        design_matrix_output_path=model_dir / "design_matrix.npy",
        row_index_output_path=model_dir / "row_index.npy",
        times_output_path=model_dir / "times.npy",
        channel_names_output_path=model_dir / "channel_names.json",
        column_names_output_path=model_dir / "column_names.json",
        test_predictors_output_path=model_dir / "test_predictors.json",
        predictor_columns_output_path=model_dir / "predictor_columns.json",
        predictor_betas_output_path=model_dir / "predictor_betas.npz",
        selection_output_path=model_dir / "selection.parquet",
        summary_output_path=model_dir / "summary.json",
    )


def load_glm_inputs(epochs_path: Path, metadata_path: Path):
    """Load epoched EEG data and metadata for one GLM run."""
    epochs = read_epochs(epochs_path, preload=True)
    metadata_df = pd.read_csv(metadata_path)
    if len(epochs) != len(metadata_df):
        raise ValueError(
            "Epochs length and metadata length must match before design filtering: "
            f"{len(epochs)} != {len(metadata_df)}"
        )
    if epochs.metadata is not None and len(epochs.metadata) == len(metadata_df):
        _validate_epoch_metadata_alignment(metadata_df, epochs.metadata.reset_index(drop=True))
    return epochs, metadata_df.reset_index(drop=True)


def save_glm_result(result: GLMFitResult, output_paths: GLMRunPaths) -> dict[str, Any]:
    """Write all primary subject-level GLM outputs to disk."""
    ensure_parent_dir(output_paths.betas_output_path)
    np.save(output_paths.betas_output_path, result.betas)
    np.save(output_paths.design_matrix_output_path, result.design_result.design_matrix)
    np.save(output_paths.row_index_output_path, result.row_index)
    np.save(output_paths.times_output_path, result.times)

    np.save(
        output_paths.standard_errors_output_path,
        result.standard_errors if result.standard_errors is not None else np.asarray([], dtype=float),
    )
    np.save(
        output_paths.t_values_output_path,
        result.t_values if result.t_values is not None else np.asarray([], dtype=float),
    )
    np.save(
        output_paths.residual_variance_output_path,
        result.residual_variance if result.residual_variance is not None else np.asarray([], dtype=float),
    )

    output_paths.channel_names_output_path.write_text(
        json.dumps(result.channel_names, indent=2) + "\n",
        encoding="utf-8",
    )
    output_paths.column_names_output_path.write_text(
        json.dumps(result.column_names, indent=2) + "\n",
        encoding="utf-8",
    )
    output_paths.test_predictors_output_path.write_text(
        json.dumps(result.test_predictors, indent=2) + "\n",
        encoding="utf-8",
    )

    predictor_columns = build_predictor_column_map(result)
    output_paths.predictor_columns_output_path.write_text(
        json.dumps(predictor_columns, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    predictor_betas = extract_beta_maps(result)
    np.savez(output_paths.predictor_betas_output_path, **predictor_betas)

    selection_df = result.design_result.filtered_metadata.copy()
    selection_df.insert(0, "row_index", result.row_index.tolist())
    write_parquet_dataframe(selection_df.reset_index(drop=True), output_paths.selection_output_path)

    summary = {
        "status": "ok",
        "n_epochs_input": result.n_epochs_input,
        "n_epochs_used": result.n_epochs_used,
        "n_channels": len(result.channel_names),
        "n_times": int(result.times.shape[0]),
        "design_matrix_shape": list(result.design_matrix_shape),
        "target_matrix_shape": list(result.target_matrix_shape),
        "column_names": list(result.column_names),
        "test_predictors": list(result.test_predictors),
        "predictor_columns": predictor_columns,
        "degrees_of_freedom": result.degrees_of_freedom,
        "rank": result.rank,
        "drop_report": dict(result.design_result.drop_report),
        "paths": {
            "betas": str(output_paths.betas_output_path),
            "standard_errors": str(output_paths.standard_errors_output_path),
            "t_values": str(output_paths.t_values_output_path),
            "residual_variance": str(output_paths.residual_variance_output_path),
            "design_matrix": str(output_paths.design_matrix_output_path),
            "selection": str(output_paths.selection_output_path),
            "predictor_betas": str(output_paths.predictor_betas_output_path),
        },
    }
    output_paths.summary_output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def save_glm_skip_result(
    epochs,
    design_result,
    output_paths: GLMRunPaths,
    *,
    reason: str,
) -> dict[str, Any]:
    """Write placeholder GLM outputs for a subject with zero usable epochs."""
    ensure_parent_dir(output_paths.betas_output_path)

    channel_names = list(epochs.ch_names)
    times = np.asarray(epochs.times, dtype=float)
    column_names = list(design_result.column_names)
    n_columns = len(column_names)
    n_channels = len(channel_names)
    n_times = int(times.shape[0])

    empty_betas = np.full((n_columns, n_channels, n_times), np.nan, dtype=float)
    empty_residuals = np.full((n_channels, n_times), np.nan, dtype=float)
    np.save(output_paths.betas_output_path, empty_betas)
    np.save(output_paths.standard_errors_output_path, empty_betas)
    np.save(output_paths.t_values_output_path, empty_betas)
    np.save(output_paths.residual_variance_output_path, empty_residuals)
    np.save(output_paths.design_matrix_output_path, np.asarray(design_result.design_matrix, dtype=float))
    np.save(output_paths.row_index_output_path, np.asarray([], dtype=int))
    np.save(output_paths.times_output_path, times)

    output_paths.channel_names_output_path.write_text(json.dumps(channel_names, indent=2) + "\n", encoding="utf-8")
    output_paths.column_names_output_path.write_text(json.dumps(column_names, indent=2) + "\n", encoding="utf-8")
    output_paths.test_predictors_output_path.write_text(
        json.dumps(list(design_result.test_predictors), indent=2) + "\n",
        encoding="utf-8",
    )

    predictor_columns = {
        predictor: [
            column_names[index]
            for index in get_predictor_column_indices(
                predictor,
                column_names=column_names,
                term_name_slices=design_result.term_name_slices,
            )
        ]
        for predictor in design_result.test_predictors
    }
    output_paths.predictor_columns_output_path.write_text(
        json.dumps(predictor_columns, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    predictor_betas = {
        column_name: np.full((n_channels, n_times), np.nan, dtype=float)
        for columns in predictor_columns.values()
        for column_name in columns
    }
    np.savez(output_paths.predictor_betas_output_path, **predictor_betas)

    selection_df = design_result.filtered_metadata.copy()
    selection_df.insert(0, "row_index", [])
    write_parquet_dataframe(selection_df.reset_index(drop=True), output_paths.selection_output_path)

    summary = {
        "status": "skipped",
        "reason": reason,
        "n_epochs_input": int(len(epochs)),
        "n_epochs_used": 0,
        "n_channels": n_channels,
        "n_times": n_times,
        "design_matrix_shape": list(np.asarray(design_result.design_matrix).shape),
        "target_matrix_shape": [0, int(n_channels * n_times)],
        "column_names": column_names,
        "test_predictors": list(design_result.test_predictors),
        "predictor_columns": predictor_columns,
        "degrees_of_freedom": 0,
        "rank": 0,
        "drop_report": dict(design_result.drop_report),
        "paths": {
            "betas": str(output_paths.betas_output_path),
            "standard_errors": str(output_paths.standard_errors_output_path),
            "t_values": str(output_paths.t_values_output_path),
            "residual_variance": str(output_paths.residual_variance_output_path),
            "design_matrix": str(output_paths.design_matrix_output_path),
            "selection": str(output_paths.selection_output_path),
            "predictor_betas": str(output_paths.predictor_betas_output_path),
        },
    }
    output_paths.summary_output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def read_glm_results(summary_path: Path) -> dict[str, Any]:
    """Read a saved GLM summary payload from disk."""
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _validate_epoch_metadata_alignment(metadata_df: pd.DataFrame, epochs_metadata_df: pd.DataFrame) -> None:
    common_columns = [column for column in ("epoch_sample", "subject_id", "task", "run") if column in metadata_df.columns and column in epochs_metadata_df.columns]
    if not common_columns:
        return
    left = metadata_df.loc[:, common_columns].reset_index(drop=True)
    right = epochs_metadata_df.loc[:, common_columns].reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Metadata CSV and Epochs.metadata are not aligned for the shared columns.")
