"""FOOOF mass IO helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rate.epochs.io import build_epoch_output_paths, read_epochs
from rate.fooof_mass.features import FooofFeatureResult
from rate.fooof_mass.fit import FooofTensorResult
from rate.fooof_mass.glm import FooofMassGLMResult
from rate.fooof_mass.windows import WindowDefinition, windows_to_records
from rate.paths import ensure_parent_dir, resolve_out_dir
from rate.util import write_parquet_dataframe


@dataclass(frozen=True)
class FooofMassRunPaths:
    """Resolved input and output paths for one subject-level FOOOF mass run."""

    epochs_input_path: Path
    metadata_input_path: Path
    windows_output_path: Path
    frequencies_output_path: Path
    fit_success_output_path: Path
    feature_validity_output_path: Path
    feature_tensor_output_path: Path
    betas_output_path: Path
    standard_errors_output_path: Path
    t_values_output_path: Path
    valid_counts_output_path: Path
    column_names_output_path: Path
    channel_names_output_path: Path
    test_predictors_output_path: Path
    selection_output_path: Path
    summary_output_path: Path


def _sanitize_name(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip()).strip("._-")
    return token or "item"


def build_fooof_mass_output_paths(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    model_name: str,
    outcome_name: str,
    epochs_path: Path | None = None,
    metadata_path: Path | None = None,
) -> FooofMassRunPaths:
    """Build canonical input/output paths for one FOOOF-mass model run."""
    out_dir = resolve_out_dir(config)
    epoch_paths = build_epoch_output_paths(config, row)
    model_dir = (
        out_dir
        / "fooof_mass"
        / str(row["subject_id"])
        / f"task-{row['task']}"
        / f"run-{row['run']}"
        / _sanitize_name(model_name)
    )
    outcome_token = _sanitize_name(outcome_name)
    return FooofMassRunPaths(
        epochs_input_path=epochs_path or epoch_paths.epochs_output_path,
        metadata_input_path=metadata_path or epoch_paths.metadata_output_path,
        windows_output_path=model_dir / "windows.json",
        frequencies_output_path=model_dir / "frequencies.npy",
        fit_success_output_path=model_dir / "fit_success.npy",
        feature_validity_output_path=model_dir / "feature_validity.json",
        feature_tensor_output_path=model_dir / f"feature_{outcome_token}.npy",
        betas_output_path=model_dir / f"feature_{outcome_token}_betas.npy",
        standard_errors_output_path=model_dir / f"feature_{outcome_token}_standard_errors.npy",
        t_values_output_path=model_dir / f"feature_{outcome_token}_t_values.npy",
        valid_counts_output_path=model_dir / f"feature_{outcome_token}_valid_counts.npy",
        column_names_output_path=model_dir / "column_names.json",
        channel_names_output_path=model_dir / "channel_names.json",
        test_predictors_output_path=model_dir / "test_predictors.json",
        selection_output_path=model_dir / "selection.parquet",
        summary_output_path=model_dir / "summary.json",
    )


def load_fooof_mass_inputs(epochs_path: Path, metadata_path: Path):
    """Load epochs and metadata for one FOOOF-mass run."""
    epochs = read_epochs(epochs_path, preload=True)
    metadata_df = pd.read_csv(metadata_path)
    if len(epochs) != len(metadata_df):
        raise ValueError(
            "Epochs length and metadata length must match before design filtering: "
            f"{len(epochs)} != {len(metadata_df)}"
        )
    return epochs, metadata_df.reset_index(drop=True)


def save_fooof_mass_result(
    *,
    output_paths: FooofMassRunPaths,
    windows: list[WindowDefinition],
    fit_result: FooofTensorResult,
    feature_result: FooofFeatureResult,
    glm_result: FooofMassGLMResult,
    feature_tensor: np.ndarray,
) -> dict[str, Any]:
    """Write primary subject-level FOOOF-mass outputs to disk."""
    ensure_parent_dir(output_paths.summary_output_path)
    output_paths.windows_output_path.write_text(
        json.dumps(windows_to_records(windows), indent=2) + "\n",
        encoding="utf-8",
    )
    np.save(output_paths.frequencies_output_path, fit_result.frequencies)
    np.save(output_paths.fit_success_output_path, fit_result.success_mask)
    np.save(output_paths.feature_tensor_output_path, feature_tensor)
    np.save(output_paths.betas_output_path, glm_result.betas)
    np.save(output_paths.standard_errors_output_path, glm_result.standard_errors)
    np.save(output_paths.t_values_output_path, glm_result.t_values)
    np.save(output_paths.valid_counts_output_path, glm_result.valid_counts)

    output_paths.feature_validity_output_path.write_text(
        json.dumps(feature_result.valid_fraction_by_feature, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_paths.column_names_output_path.write_text(
        json.dumps(glm_result.column_names, indent=2) + "\n",
        encoding="utf-8",
    )
    output_paths.channel_names_output_path.write_text(
        json.dumps(glm_result.channel_names, indent=2) + "\n",
        encoding="utf-8",
    )
    output_paths.test_predictors_output_path.write_text(
        json.dumps(glm_result.test_predictors, indent=2) + "\n",
        encoding="utf-8",
    )

    selection_df = glm_result.design_result.filtered_metadata.copy()
    selection_df.insert(0, "row_index", glm_result.design_result.row_index.to_numpy(dtype=int, copy=True))
    write_parquet_dataframe(selection_df.reset_index(drop=True), output_paths.selection_output_path)

    summary = {
        "status": "ok",
        "outcome_name": glm_result.outcome_name,
        "n_windows": len(windows),
        "window_centers_s": glm_result.window_centers_s.tolist(),
        "n_channels": len(glm_result.channel_names),
        "n_epochs_input": int(feature_tensor.shape[0]),
        "n_epochs_used": int(glm_result.design_result.n_rows_final),
        "n_fooof_attempted": int(fit_result.n_attempted),
        "n_fooof_failed": int(fit_result.n_failed),
        "feature_valid_fraction": float(feature_result.valid_fraction_by_feature.get(glm_result.outcome_name, 0.0)),
        "design_shape": list(glm_result.design_result.design_matrix.shape),
        "beta_shape": list(glm_result.betas.shape),
        "target_shape": list(glm_result.target_shape),
        "drop_report": dict(glm_result.design_result.drop_report),
        "missing_data_strategy": (
            "Trials are selected once by the design layer, then each channel-window target is fit "
            "using the finite subset of those trials for the chosen feature."
        ),
    }
    output_paths.summary_output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def read_fooof_mass_results(summary_path: Path) -> dict[str, Any]:
    """Read a saved FOOOF-mass summary payload."""
    return json.loads(summary_path.read_text(encoding="utf-8"))
