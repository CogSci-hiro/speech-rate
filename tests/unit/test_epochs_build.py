"""Unit tests for epoch building and timing conversion."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import mne
import numpy as np
import pandas as pd
import pytest

from rate.epochs.build import build_epochs_for_run, convert_event_rows_to_samples, select_run_events


def _config(out_dir: Path) -> dict:
    return {
        "paths": {"out_dir": str(out_dir)},
        "epochs": {
            "event_column": "time_s",
            "time_origin": "recording_start",
            "event_id_column": "event_id",
            "tmin_s": -0.1,
            "tmax_s": 0.2,
            "baseline": {"start_s": -0.1, "end_s": 0.0},
            "preload": True,
            "on_out_of_bounds": "drop",
        },
    }


def _make_raw(*, sfreq: float = 100.0, first_samp: int = 500, n_samples: int = 1000) -> mne.io.RawArray:
    info = mne.create_info(["Fz", "Cz"], sfreq=sfreq, ch_types=["eeg", "eeg"])
    data = np.random.default_rng(42).normal(scale=1e-6, size=(2, n_samples))
    return mne.io.RawArray(data, info, first_samp=first_samp, verbose="ERROR")


def test_select_run_events_filters_one_run() -> None:
    events_df = pd.DataFrame(
        [
            {"dyad_id": "dyad-001", "subject_id": "sub-001", "task": "conversation", "run": "1", "time_s": 5.5, "event_id": 1},
            {"dyad_id": "dyad-001", "subject_id": "sub-001", "task": "conversation", "run": "2", "time_s": 6.5, "event_id": 2},
        ]
    )
    row = {"dyad_id": "dyad-001", "subject_id": "sub-001", "task": "conversation", "run": "1"}

    selected = select_run_events(events_df, row)

    assert len(selected) == 1
    assert selected.iloc[0]["time_s"] == pytest.approx(5.5)


def test_convert_event_rows_to_samples_handles_first_samp_for_both_time_origins() -> None:
    raw = _make_raw(sfreq=100.0, first_samp=500)
    event_rows = pd.DataFrame(
        [
            {"time_s": 5.5, "sample": 550, "event_id": 1},
            {"time_s": 0.5, "sample": 550, "event_id": 1},
        ]
    )

    recording_start_samples = convert_event_rows_to_samples(
        event_rows.iloc[[0]],
        raw,
        event_column="time_s",
        time_origin="recording_start",
    )
    raw_start_samples = convert_event_rows_to_samples(
        event_rows.iloc[[1]],
        raw,
        event_column="time_s",
        time_origin="raw_start",
    )

    assert recording_start_samples.tolist() == [550]
    assert raw_start_samples.tolist() == [550]


def test_build_epochs_for_run_uses_time_s_on_downsampled_raw_and_drops_out_of_bounds(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = _config(out_dir)
    row = {"dyad_id": "dyad-001", "subject_id": "sub-001", "role": "odd", "task": "conversation", "run": "1"}

    raw = _make_raw(sfreq=100.0, first_samp=250, n_samples=400)
    raw_path = tmp_path / "sub-001_raw.fif"
    raw.save(raw_path, overwrite=True, verbose="ERROR")

    events_df = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-001",
                "role": "odd",
                "task": "conversation",
                "run": "1",
                "sample": 505,
                "time_s": 2.525,
                "previous": 0,
                "event_id": 1,
            },
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-001",
                "role": "odd",
                "task": "conversation",
                "run": "1",
                "sample": 550,
                "time_s": 2.75,
                "previous": 0,
                "event_id": 2,
            },
        ]
    )
    events_table_path = tmp_path / "events.csv"
    events_df.to_csv(events_table_path, index=False)

    result = build_epochs_for_run(
        config,
        row,
        raw_path=raw_path,
        events_table_path=events_table_path,
    )

    assert len(result.epochs) == 1
    assert result.events_array[:, 0].tolist() == [275]
    assert result.events_array[:, 2].tolist() == [2]
    assert result.metadata["event_id"].tolist() == [2]
    assert result.metadata["epoch_sample"].tolist() == [275]
    assert result.summary["raw_first_samp"] == 250
    assert result.summary["selected_event_count"] == 2
    assert result.summary["dropped_out_of_bounds_count"] == 1
    assert result.summary["epochs_count"] == 1
