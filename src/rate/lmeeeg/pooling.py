"""Helpers for pooled lmeEEG model fitting across many rows."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rate.epochs.io import build_epoch_output_paths
from rate.lmeeeg.io import load_lmeeeg_inputs_with_trial_regressors
from rate.lmeeeg.mass_model import LMEEEGResult, fit_lmeeeg_mass_model
from rate.lmeeeg.targets import section_supports_targets
from rate.lmeeeg.trial_data import build_lmeeeg_trial_data_from_arrays
from rate.paths import resolve_out_dir
from rate.util import read_parquet_dataframe

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PooledLMEEEGInputs:
    """Pooled EEG arrays and metadata before model-specific design filtering."""

    eeg_data: np.ndarray
    metadata_df: pd.DataFrame
    channel_names: list[str]
    times: np.ndarray
    metadata_provenance: dict[str, Any] | None = None


def resolve_dataset_index_path(config: Mapping[str, Any], *, dataset_index: str | None = None) -> Path:
    """Resolve the dataset-index parquet path."""
    if dataset_index:
        return Path(dataset_index)
    return resolve_out_dir(config) / "dataset" / "index.parquet"


def select_lmeeeg_rows(
    config: Mapping[str, Any],
    *,
    dataset_index: str | None = None,
    subject_id: str | None = None,
    task: str | None = None,
    run: str | None = None,
) -> list[Mapping[str, Any]]:
    """Return runnable dataset rows matching optional filters."""
    dataset_index_path = resolve_dataset_index_path(config, dataset_index=dataset_index)
    if not dataset_index_path.exists():
        raise FileNotFoundError(f"Dataset index not found: {dataset_index_path}")

    LOGGER.info("Reading lmeEEG dataset index: %s", dataset_index_path)
    index_df = read_parquet_dataframe(dataset_index_path)
    if "status" in index_df.columns:
        index_df = index_df.loc[index_df["status"].astype(str).str.lower() == "ok"].copy()

    for field_name, field_value in (("subject_id", subject_id), ("task", task), ("run", run)):
        if field_value is None:
            continue
        index_df = index_df.loc[index_df[field_name].astype(str) == str(field_value)].copy()

    required = ["subject_id", "task", "run"]
    index_df = index_df.dropna(subset=required)
    rows = [row._asdict() for row in index_df.itertuples(index=False, name="Row")]
    if not rows:
        raise ValueError("No runnable dataset-index rows matched the requested lmeEEG selection.")
    LOGGER.info("Selected %d dataset rows for pooled lmeEEG fitting", len(rows))
    return rows


def load_pooled_lmeeeg_inputs(
    config: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    *,
    config_section: str = "lmeeeg",
    target_name: str | None = None,
    model_name: str | None = None,
    trial_regressors_path: Path | None = None,
):
    """Load and concatenate lmeEEG-ready arrays across many dataset rows."""
    pooled_eeg: list[np.ndarray] = []
    pooled_metadata: list[pd.DataFrame] = []
    reference_channel_names: list[str] | None = None
    reference_times: np.ndarray | None = None
    unique_rows_by_metadata_path: dict[str, Mapping[str, Any]] = {}
    merge_summaries: list[dict[str, Any]] = []

    for row in rows:
        if section_supports_targets(config, config_section=config_section):
            if target_name is None:
                raise ValueError("`target_name` is required when loading pooled target-specific lmeEEG trial data.")
            metadata_path = str(build_epoch_output_paths(config, row).metadata_output_path)
        else:
            metadata_path = str(build_epoch_output_paths(config, row).metadata_output_path)
        unique_rows_by_metadata_path.setdefault(metadata_path, row)

    LOGGER.info(
        "Loading pooled lmeEEG inputs for model=%s from %d dataset rows (%d unique metadata files)",
        model_name or "<default>",
        len(rows),
        len(unique_rows_by_metadata_path),
    )

    for row_index, (metadata_path, row) in enumerate(unique_rows_by_metadata_path.items(), start=1):
        epoch_paths = build_epoch_output_paths(config, row)
        LOGGER.info(
            "Loading lmeEEG source %d/%d: subject=%s task=%s run=%s file=%s",
            row_index,
            len(unique_rows_by_metadata_path),
            row.get("subject_id"),
            row.get("task"),
            row.get("run"),
            metadata_path,
        )
        epochs, metadata_df = load_lmeeeg_inputs_with_trial_regressors(
            epoch_paths.epochs_output_path,
            epoch_paths.metadata_output_path,
            trial_regressors_path=trial_regressors_path,
        )
        for field_name in ("dyad_id", "subject_id", "task", "run"):
            if field_name in row and field_name not in metadata_df.columns:
                metadata_df.loc[:, field_name] = row[field_name]
        merge_summaries.append(dict(getattr(metadata_df, "attrs", {}).get("metadata_provenance") or {}))

        eeg_data = epochs.get_data(copy=True)
        times = np.asarray(epochs.times, dtype=float)
        channel_names = list(epochs.ch_names)
        if reference_channel_names is None:
            reference_channel_names = channel_names
            reference_times = times
        else:
            if channel_names != reference_channel_names:
                raise ValueError("All pooled lmeEEG rows must share the same channel order.")
            if not np.allclose(times, reference_times):
                raise ValueError("All pooled lmeEEG rows must share the same time axis.")

        pooled_eeg.append(np.asarray(eeg_data, dtype=float))
        pooled_metadata.append(metadata_df.reset_index(drop=True))

    assert reference_channel_names is not None
    assert reference_times is not None
    total_observations = int(sum(chunk.shape[0] for chunk in pooled_eeg))
    LOGGER.info(
        "Loaded pooled lmeEEG data for model=%s: %d observations, %d channels, %d time points",
        model_name or "<default>",
        total_observations,
        len(reference_channel_names),
        int(reference_times.shape[0]),
    )
    return PooledLMEEEGInputs(
        eeg_data=np.concatenate(pooled_eeg, axis=0),
        metadata_df=pd.concat(pooled_metadata, axis=0, ignore_index=True),
        channel_names=list(reference_channel_names),
        times=np.asarray(reference_times, dtype=float),
        metadata_provenance={"sources": merge_summaries},
    )


def load_pooled_lmeeeg_trial_data(
    config: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    *,
    config_section: str = "lmeeeg",
    target_name: str | None = None,
    model_name: str | None = None,
    trial_regressors_path: Path | None = None,
):
    """Load pooled arrays and prepare one model-specific trial table."""
    pooled_inputs = load_pooled_lmeeeg_inputs(
        config,
        rows,
        config_section=config_section,
        target_name=target_name,
        model_name=model_name,
        trial_regressors_path=trial_regressors_path,
    )
    return build_lmeeeg_trial_data_from_arrays(
        eeg_data=pooled_inputs.eeg_data,
        metadata_df=pooled_inputs.metadata_df,
        channel_names=pooled_inputs.channel_names,
        times=pooled_inputs.times,
        config=config,
        config_section=config_section,
        model_name=model_name,
        metadata_provenance=pooled_inputs.metadata_provenance,
    )


def fit_pooled_lmeeeg_model(
    config: Mapping[str, Any],
    *,
    pooled_inputs: PooledLMEEEGInputs,
    config_section: str = "lmeeeg",
    model_name: str | None = None,
    metadata_df: pd.DataFrame | None = None,
    metadata_provenance: dict[str, Any] | None = None,
) -> LMEEEGResult:
    """Fit one pooled lmeEEG model from cached pooled inputs and metadata."""
    resolved_metadata = pooled_inputs.metadata_df if metadata_df is None else metadata_df
    resolved_provenance = pooled_inputs.metadata_provenance if metadata_provenance is None else metadata_provenance
    trial_data = build_lmeeeg_trial_data_from_arrays(
        eeg_data=pooled_inputs.eeg_data,
        metadata_df=resolved_metadata,
        channel_names=pooled_inputs.channel_names,
        times=pooled_inputs.times,
        config=config,
        config_section=config_section,
        model_name=model_name,
        metadata_provenance=resolved_provenance,
    )
    return fit_lmeeeg_mass_model(
        trial_data,
        config,
        config_section=config_section,
        model_name=model_name,
    )
