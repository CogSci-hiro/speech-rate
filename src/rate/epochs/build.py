"""Epoch building helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rate.eeg.io import read_raw_eeg
from rate.epochs.metadata import build_epoch_metadata
from rate.epochs.io import read_events_table

LOGGER = logging.getLogger(__name__)

EVENT_SELECTION_COLUMNS = ("dyad_id", "subject_id", "task", "run")


class EpochBuildError(RuntimeError):
    """Raised when epoch construction fails."""


@dataclass(frozen=True)
class EpochBuildResult:
    """Container for the final epoching artifacts before writing."""

    epochs: Any
    metadata: pd.DataFrame
    events_array: np.ndarray
    selected_events: pd.DataFrame
    summary: dict[str, Any]


def build_epoch_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a minimal description of the intended epoching plan."""
    epochs_cfg = dict(config.get("epochs") or {})
    return {
        "stage": "epochs",
        "tmin_s": epochs_cfg.get("tmin_s"),
        "tmax_s": epochs_cfg.get("tmax_s"),
        "event_column": epochs_cfg.get("event_column"),
        "time_origin": epochs_cfg.get("time_origin"),
    }


def select_run_events(events_df: pd.DataFrame, row: Mapping[str, Any]) -> pd.DataFrame:
    """Select the event rows matching one subject/task/run entry."""
    missing_columns = [column for column in EVENT_SELECTION_COLUMNS if column not in events_df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise EpochBuildError(f"Events table is missing required selection columns: {missing_text}")

    mask = np.ones((len(events_df),), dtype=bool)
    for column in EVENT_SELECTION_COLUMNS:
        mask &= events_df[column].astype(str) == str(row[column])

    selected = events_df.loc[mask].copy().reset_index(drop=True)
    if selected.empty:
        raise EpochBuildError(
            "No events found for "
            f"dyad_id={row['dyad_id']}, subject_id={row['subject_id']}, task={row['task']}, run={row['run']}"
        )
    return selected


def convert_event_rows_to_samples(
    event_rows: pd.DataFrame,
    raw,
    *,
    event_column: str,
    time_origin: str,
) -> np.ndarray:
    """Convert event-table timing values into absolute MNE sample indices.

    Parameters
    ----------
    event_rows
        Event rows selected for a single raw file.
    raw
        Loaded MNE Raw object for the same run.
    event_column
        Column used for event locking. This may be a time column or the
        absolute ``sample`` column itself.
    time_origin
        Timing convention for ``event_column``. Supported values are:
        ``"absolute_sample"``, ``"recording_start"``, and ``"raw_start"``.

    Returns
    -------
    numpy.ndarray
        Absolute integer sample indices suitable for the first column of an
        MNE events array.

    Notes
    -----
    ``raw.first_samp`` is handled explicitly here. For ``recording_start``
    times, the times already include the recording sample origin, so sample
    indices are ``round(time_s * sfreq)``. For ``raw_start`` times, the times
    are relative to ``raw.times`` and must be shifted by ``raw.first_samp``.
    """
    if event_column not in event_rows.columns:
        raise EpochBuildError(f"Configured epoch event column is not present in events table: {event_column}")

    sfreq = float(raw.info["sfreq"])

    if time_origin == "absolute_sample" or event_column == "sample":
        source = pd.to_numeric(event_rows[event_column], errors="raise")
        samples = np.rint(source.to_numpy(dtype=float)).astype(int)
    elif time_origin == "recording_start":
        event_times = pd.to_numeric(event_rows[event_column], errors="raise").to_numpy(dtype=float)
        samples = np.rint(event_times * sfreq).astype(int)
    elif time_origin == "raw_start":
        event_times = pd.to_numeric(event_rows[event_column], errors="raise").to_numpy(dtype=float)
        samples = raw.first_samp + np.rint(event_times * sfreq).astype(int)
    else:
        raise EpochBuildError(f"Unsupported epochs.time_origin value: {time_origin}")

    if "sample" in event_rows.columns and event_column != "sample" and event_column in event_rows.columns:
        reference_samples = pd.to_numeric(event_rows["sample"], errors="raise").to_numpy(dtype=float)
        event_values = pd.to_numeric(event_rows[event_column], errors="raise").to_numpy(dtype=float)
        finite_mask = np.isfinite(reference_samples) & np.isfinite(event_values) & (event_values != 0.0)
        if np.any(finite_mask):
            inferred_source_sfreq = float(np.nanmedian(reference_samples[finite_mask] / event_values[finite_mask]))
            if np.isclose(inferred_source_sfreq, sfreq, atol=1e-6, rtol=1e-3):
                rounded_reference = np.rint(reference_samples).astype(int)
                if not np.all(np.abs(samples - rounded_reference) <= 1):
                    mismatch_count = int(np.sum(np.abs(samples - rounded_reference) > 1))
                    raise EpochBuildError(
                        f"Converted epoch samples disagree with events.sample for {mismatch_count} rows. "
                        "Check epochs.time_origin and event_column."
                    )
            else:
                LOGGER.info(
                    "Skipping strict sample cross-check because events appear to use source sfreq=%0.6f but raw uses sfreq=%0.6f.",
                    inferred_source_sfreq,
                    sfreq,
                )

    first_valid_sample = int(raw.first_samp)
    last_valid_sample = int(raw.last_samp)
    if np.any(samples < first_valid_sample) or np.any(samples > last_valid_sample):
        bad_samples = samples[(samples < first_valid_sample) | (samples > last_valid_sample)]
        preview = ", ".join(str(value) for value in bad_samples[:5])
        raise EpochBuildError(
            f"Converted event samples fall outside raw bounds [{first_valid_sample}, {last_valid_sample}]: {preview}"
        )

    return samples


def build_mne_events_array(
    event_rows: pd.DataFrame,
    event_samples: np.ndarray,
    *,
    event_id_column: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build the integer MNE events array and label mapping for epoching."""
    if event_id_column not in event_rows.columns:
        raise EpochBuildError(f"Configured event_id_column is not present in events table: {event_id_column}")

    raw_labels = event_rows[event_id_column].astype(str).reset_index(drop=True)
    numeric_codes = pd.to_numeric(raw_labels, errors="coerce")
    if numeric_codes.notna().all():
        event_codes = numeric_codes.to_numpy(dtype=int)
        event_id_map = {
            label: int(code)
            for label, code in zip(raw_labels.tolist(), event_codes.tolist())
        }
    else:
        unique_labels = list(dict.fromkeys(raw_labels.tolist()))
        event_id_map = {label: index + 1 for index, label in enumerate(unique_labels)}
        event_codes = np.asarray([event_id_map[label] for label in raw_labels.tolist()], dtype=int)

    previous_values = np.zeros_like(event_codes, dtype=int)
    events_array = np.column_stack([event_samples.astype(int), previous_values, event_codes.astype(int)])
    return events_array.astype(int, copy=False), event_id_map


def _bounds_mask_for_epochs(raw, event_samples: np.ndarray, *, tmin_s: float, tmax_s: float) -> np.ndarray:
    sfreq = float(raw.info["sfreq"])
    first_valid_sample = int(raw.first_samp)
    last_valid_sample = int(raw.last_samp)
    start_offsets = np.rint(tmin_s * sfreq).astype(int)
    end_offsets = np.rint(tmax_s * sfreq).astype(int)
    epoch_starts = event_samples + start_offsets
    epoch_ends = event_samples + end_offsets
    return (epoch_starts >= first_valid_sample) & (epoch_ends <= last_valid_sample)


def _resolve_baseline(config: Mapping[str, Any]) -> tuple[float | None, float | None] | None:
    baseline_cfg = dict((config.get("epochs") or {}).get("baseline") or {})
    if not baseline_cfg:
        return None

    start_s = baseline_cfg.get("start_s")
    end_s = baseline_cfg.get("end_s")
    if start_s is None and end_s is None:
        return None
    return (
        None if start_s is None else float(start_s),
        None if end_s is None else float(end_s),
    )


def build_epochs_for_run(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    raw_path: Path,
    events_table_path: Path,
):
    """Build epochs for one preprocessed raw file and one events table subset."""
    import mne

    raw = read_raw_eeg(raw_path, preload=bool((config.get("epochs") or {}).get("preload", True)), verbose="ERROR")
    events_df = read_events_table(events_table_path)
    selected_events = select_run_events(events_df, row)

    epochs_cfg = dict(config.get("epochs") or {})
    event_column = str(epochs_cfg.get("event_column", "time_s"))
    time_origin = str(epochs_cfg.get("time_origin", "recording_start"))
    event_id_column = str(epochs_cfg.get("event_id_column", "event_id"))
    tmin_s = float(epochs_cfg.get("tmin_s", -0.5))
    tmax_s = float(epochs_cfg.get("tmax_s", 1.0))
    on_out_of_bounds = str(epochs_cfg.get("on_out_of_bounds", "drop")).lower()

    event_samples = convert_event_rows_to_samples(
        selected_events,
        raw,
        event_column=event_column,
        time_origin=time_origin,
    )
    in_bounds_mask = _bounds_mask_for_epochs(raw, event_samples, tmin_s=tmin_s, tmax_s=tmax_s)
    dropped_count = int((~in_bounds_mask).sum())
    if dropped_count and on_out_of_bounds == "raise":
        raise EpochBuildError(
            f"{dropped_count} events would create out-of-bounds epochs for "
            f"subject_id={row['subject_id']}, task={row['task']}, run={row['run']}"
        )

    kept_events = selected_events.loc[in_bounds_mask].reset_index(drop=True)
    kept_samples = event_samples[in_bounds_mask]
    if kept_events.empty:
        raise EpochBuildError(
            "No valid in-bounds events remain for "
            f"subject_id={row['subject_id']}, task={row['task']}, run={row['run']}"
        )

    events_array, event_id_map = build_mne_events_array(kept_events, kept_samples, event_id_column=event_id_column)
    metadata_df = build_epoch_metadata(kept_events, config, epoch_samples=events_array[:, 0])
    epochs = mne.Epochs(
        raw,
        events_array,
        event_id=event_id_map,
        tmin=tmin_s,
        tmax=tmax_s,
        baseline=_resolve_baseline(config),
        preload=bool(epochs_cfg.get("preload", True)),
        metadata=metadata_df,
        verbose="ERROR",
    )

    if len(epochs) != len(metadata_df):
        raise EpochBuildError("Epochs length and metadata length diverged during construction.")

    summary = {
        "status": "ok",
        "dyad_id": str(row["dyad_id"]),
        "subject_id": str(row["subject_id"]),
        "task": str(row["task"]),
        "run": str(row["run"]),
        "raw_path": str(raw_path),
        "events_table_path": str(events_table_path),
        "event_column": event_column,
        "time_origin": time_origin,
        "raw_first_samp": int(raw.first_samp),
        "raw_last_samp": int(raw.last_samp),
        "sfreq_hz": float(raw.info["sfreq"]),
        "selected_event_count": int(len(selected_events)),
        "dropped_out_of_bounds_count": dropped_count,
        "kept_event_count": int(len(kept_events)),
        "epochs_count": int(len(epochs)),
    }
    if dropped_count:
        LOGGER.warning(
            "Dropped %d out-of-bounds events for %s task=%s run=%s",
            dropped_count,
            row["subject_id"],
            row["task"],
            row["run"],
        )

    return EpochBuildResult(
        epochs=epochs,
        metadata=metadata_df,
        events_array=events_array,
        selected_events=kept_events,
        summary=summary,
    )
