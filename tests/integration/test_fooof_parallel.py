from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _fooof_config(out_dir: Path, n_jobs: int) -> dict:
    return {
        "paths": {"out_dir": str(out_dir)},
        "dataset": {},
        "events": {},
        "evoked_trf": {},
        "induced_trf": {},
        "fooof": {
            "n_jobs": int(n_jobs),
            "prespeech_window_s": {"tmin": -1.0, "tmax": 0.0},
            "rois": {
                "definition": "inline",
                "values": {
                    "frontocentral": ["Fz", "Cz"],
                },
            },
            "spectrum": {
                "fmin_hz": 2.0,
                "fmax_hz": 40.0,
                "method": "welch",
                "welch_nperseg_s": 1.0,
                "welch_overlap": 0.5,
            },
            "model": {
                "knee": False,
                "max_n_peaks": 4,
                "peak_width_limits_hz": [1.0, 8.0],
                "min_peak_height": 0.0,
                "peak_threshold": 2.0,
            },
            "features": {
                "periodic_bands_hz": {
                    "theta": [4.0, 7.0],
                    "alpha": [8.0, 12.0],
                    "beta": [13.0, 30.0],
                }
            },
        },
        "ddc": {},
        "reporting": {},
        "random_seed": 123,
    }


def _make_raw(path: Path, sfreq: float, f1: float, f2: float) -> None:
    import mne

    t = np.arange(0.0, 4.0, 1.0 / sfreq)
    data = np.vstack(
        [
            np.sin(2 * np.pi * f1 * t),
            0.5 * np.sin(2 * np.pi * f2 * t),
        ]
    )
    info = mne.create_info(ch_names=["Fz", "Cz"], sfreq=sfreq, ch_types=["eeg", "eeg"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.save(path, overwrite=True, verbose="ERROR")


def _build_inputs(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sfreq = 100.0
    raw_a = tmp_path / "sub-001_run-1_raw.fif"
    raw_b = tmp_path / "sub-002_run-1_raw.fif"
    _make_raw(raw_a, sfreq, 10.0, 6.0)
    _make_raw(raw_b, sfreq, 12.0, 8.0)

    dataset_index = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-001",
                "run": 1,
                "task": "conversation",
                "eeg_path": str(raw_a),
                "status": "ok",
            },
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-002",
                "run": 1,
                "task": "conversation",
                "eeg_path": str(raw_b),
                "status": "ok",
            },
        ]
    )

    events_df = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-001",
                "run": 1,
                "task": "conversation",
                "event_id": "dyad-001_1_conversation_ev00001",
                "t_onset": 1.5,
            },
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-001",
                "run": 1,
                "task": "conversation",
                "event_id": "dyad-001_1_conversation_ev00002",
                "t_onset": 2.5,
            },
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-002",
                "run": 1,
                "task": "conversation",
                "event_id": "dyad-001_1_conversation_ev00003",
                "t_onset": 1.6,
            },
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-002",
                "run": 1,
                "task": "conversation",
                "event_id": "dyad-001_1_conversation_ev00004",
                "t_onset": 2.6,
            },
        ]
    )
    return dataset_index, events_df


def test_fooof_parallel_matches_serial(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    monkeypatch.setenv("MNE_DONTWRITE_HOME", "true")
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    orig_sysconf = os.sysconf

    def _sysconf_patched(name: str):
        if name == "SC_SEM_NSEMS_MAX":
            return 256
        return orig_sysconf(name)

    monkeypatch.setattr(os, "sysconf", _sysconf_patched)

    from rate.fooof.pipeline import run_fooof_pipeline
    from rate.util import read_parquet_dataframe

    dataset_index, events_df = _build_inputs(tmp_path)

    out_serial = tmp_path / "out_serial"
    out_parallel = tmp_path / "out_parallel"

    run_fooof_pipeline(
        config=_fooof_config(out_serial, n_jobs=1),
        dataset_index=dataset_index.copy(),
        events_df=events_df.copy(),
        out_dir=out_serial,
    )
    run_fooof_pipeline(
        config=_fooof_config(out_parallel, n_jobs=2),
        dataset_index=dataset_index.copy(),
        events_df=events_df.copy(),
        out_dir=out_parallel,
    )

    serial_df = read_parquet_dataframe(out_serial / "fooof" / "trial_level" / "fooof_trial_channels.parquet")
    parallel_df = read_parquet_dataframe(out_parallel / "fooof" / "trial_level" / "fooof_trial_channels.parquet")

    pd.testing.assert_frame_equal(serial_df, parallel_df)
