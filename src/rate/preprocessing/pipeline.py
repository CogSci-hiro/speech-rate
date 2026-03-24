"""End-to-end preprocessing pipeline orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from rate.eeg.io import read_raw_eeg
from rate.preprocessing.bad_channels import interpolate_bad_channels, load_bad_channels_from_bids_tsv
from rate.preprocessing.channels import apply_montage, split_eeg_and_emg_channels
from rate.preprocessing.downsample import downsample_raw
from rate.preprocessing.events import extract_events_table
from rate.preprocessing.filter import apply_bandpass_filter
from rate.preprocessing.ica import apply_precomputed_ica
from rate.preprocessing.io import (
    PreprocessingRunPaths,
    build_preprocessing_output_paths,
    write_emg_artifact,
    write_events_table,
    write_preprocessing_summary,
)
from rate.preprocessing.rereference import apply_average_reference

LOGGER = logging.getLogger(__name__)


def _row_mapping_from_series(row: pd.Series | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        return row.to_dict()
    return dict(row)


def preprocess_run(
    config: Mapping[str, Any],
    row: pd.Series | Mapping[str, Any],
    *,
    raw_eeg_path: Path | None = None,
    channels_tsv_path: Path | None = None,
    ica_path: Path | None = None,
    eeg_output_path: Path | None = None,
    emg_output_path: Path | None = None,
    events_output_path: Path | None = None,
    summary_output_path: Path | None = None,
) -> PreprocessingRunPaths:
    """Run the preprocessing pipeline in the required order for one dataset row.

    Parameters
    ----------
    config
        Resolved repository config.
    row
        Dataset-index row with at least ``dyad_id``, ``subject_id``, ``task``,
        ``run``, and ``eeg_path``.

    Returns
    -------
    PreprocessingRunPaths
        Paths to the generated preprocessing artifacts.
    """
    row_mapping = _row_mapping_from_series(row)
    resolved_paths: PreprocessingRunPaths | None = None

    def _get_resolved_paths() -> PreprocessingRunPaths:
        nonlocal resolved_paths
        if resolved_paths is None:
            resolved_paths = build_preprocessing_output_paths(config, row_mapping)
        return resolved_paths

    run_paths = PreprocessingRunPaths(
        raw_eeg_path=raw_eeg_path or _get_resolved_paths().raw_eeg_path,
        channels_tsv_path=channels_tsv_path or _get_resolved_paths().channels_tsv_path,
        ica_path=ica_path or _get_resolved_paths().ica_path,
        eeg_output_path=eeg_output_path or _get_resolved_paths().eeg_output_path,
        emg_output_path=emg_output_path or _get_resolved_paths().emg_output_path,
        events_output_path=events_output_path or _get_resolved_paths().events_output_path,
        summary_output_path=summary_output_path or _get_resolved_paths().summary_output_path,
    )

    raw = read_raw_eeg(run_paths.raw_eeg_path, preload=True, verbose="ERROR")

    # 1. Separate/drop channels, extract events, and add montage.
    events_df, event_id_map = extract_events_table(raw, config)
    channel_split = split_eeg_and_emg_channels(raw, config)
    eeg_raw = raw.copy().pick(channel_split.eeg_names)
    emg_raw = raw.copy().pick(channel_split.emg_names) if channel_split.emg_names else None
    apply_montage(eeg_raw, config)
    load_bad_channels_from_bids_tsv(run_paths.channels_tsv_path, raw=eeg_raw)

    # 2. Downsample.
    downsample_raw(eeg_raw, config)

    # 3. Bandpass.
    apply_bandpass_filter(eeg_raw, config)

    # 4. Apply precomputed ICA.
    apply_precomputed_ica(eeg_raw, run_paths.ica_path)

    # 5. Interpolate bad channels.
    interpolate_bad_channels(eeg_raw)

    # 6. Rereference.
    apply_average_reference(eeg_raw)

    run_paths.eeg_output_path.parent.mkdir(parents=True, exist_ok=True)
    eeg_raw.save(run_paths.eeg_output_path, overwrite=True, verbose="ERROR")
    write_emg_artifact(run_paths.emg_output_path, emg_raw)
    write_events_table(run_paths.events_output_path, events_df)

    summary = {
        "dyad_id": str(row_mapping["dyad_id"]),
        "subject_id": str(row_mapping["subject_id"]),
        "task": str(row_mapping["task"]),
        "run": str(row_mapping["run"]),
        "raw_eeg_path": str(run_paths.raw_eeg_path),
        "channels_tsv_path": str(run_paths.channels_tsv_path),
        "ica_path": str(run_paths.ica_path),
        "eeg_output_path": str(run_paths.eeg_output_path),
        "emg_output_path": str(run_paths.emg_output_path),
        "events_output_path": str(run_paths.events_output_path),
        "sfreq_before_hz": float(raw.info["sfreq"]),
        "sfreq_after_hz": float(eeg_raw.info["sfreq"]),
        "event_count": int(len(events_df)),
        "event_id_map": dict(event_id_map),
        "bad_channels": list(eeg_raw.info["bads"]),
        "eeg_channels": list(eeg_raw.ch_names),
        "emg_channels": list(emg_raw.ch_names) if emg_raw is not None else [],
        "dropped_channels": channel_split.dropped_names,
        "status": "ok",
    }
    write_preprocessing_summary(run_paths.summary_output_path, summary)
    LOGGER.info(
        "Preprocessed %s task=%s run=%s -> %s",
        row_mapping["subject_id"],
        row_mapping["task"],
        row_mapping["run"],
        run_paths.eeg_output_path,
    )
    return run_paths
