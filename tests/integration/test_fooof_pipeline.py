from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _make_config(out_dir: Path) -> dict:
    return {
        "paths": {
            "out_dir": str(out_dir),
        },
        "dataset": {},
        "events": {},
        "evoked_trf": {},
        "induced_trf": {},
        "fooof": {
            "n_jobs": 1,
            "prespeech_window_s": {"tmin": -1.0, "tmax": 0.0},
            "rois": {
                "definition": "inline",
                "values": {
                    "frontal": ["Fz", "Cz"],
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


def test_fooof_cli_smoke_writes_canonical_channel_table(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    monkeypatch.setenv("MNE_DONTWRITE_HOME", "true")
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")

    import mne
    from rate.cli.main import main
    from rate.util import read_parquet_dataframe, write_parquet_dataframe

    out_dir = tmp_path / "outputs"
    (out_dir / "dataset").mkdir(parents=True, exist_ok=True)
    (out_dir / "events").mkdir(parents=True, exist_ok=True)

    sfreq = 100.0
    t = np.arange(0.0, 4.0, 1.0 / sfreq)
    data = np.vstack(
        [
            np.sin(2 * np.pi * 10.0 * t),
            0.5 * np.sin(2 * np.pi * 6.0 * t),
        ]
    )
    info = mne.create_info(ch_names=["Fz", "Cz"], sfreq=sfreq, ch_types=["eeg", "eeg"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    fif_path = tmp_path / "sub-001_run-1_raw.fif"
    raw.save(fif_path, overwrite=True, verbose="ERROR")

    dataset_index = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-001",
                "run": 1,
                "task": "conversation",
                "eeg_path": str(fif_path),
                "status": "ok",
            }
        ]
    )
    write_parquet_dataframe(dataset_index, out_dir / "dataset" / "index.parquet")

    events_df = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-001",
                "run": 1,
                "task": "conversation",
                "event_id": "dyad-001_1_conversation_ev00001",
                "t_onset": 2.0,
            }
        ]
    )
    write_parquet_dataframe(events_df, out_dir / "events" / "events.parquet")

    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _make_config(out_dir))

    exit_code = main(["fooof", "--config", str(cfg_path), "--out-dir", str(out_dir)])
    assert exit_code == 0

    channels_path = out_dir / "fooof" / "trial_level" / "fooof_trial_channels.parquet"
    assert channels_path.exists()

    channels_df = read_parquet_dataframe(channels_path)
    assert len(channels_df) > 0
    required_cols = {
        "dyad_id",
        "subject_id",
        "run",
        "event_id",
        "channel",
        "window_start",
        "window_end",
        "window_length",
        "offset",
        "exponent",
        "knee",
        "theta_power",
        "alpha_power",
        "beta_power",
        "theta_cf",
        "alpha_cf",
        "beta_cf",
        "r_squared",
        "error",
        "fit_success",
    }
    assert required_cols.issubset(set(channels_df.columns))
    assert {"fooof_aperiodic_mode", "fooof_peak_params_json"}.issubset(set(channels_df.columns))
    assert set(channels_df["fooof_aperiodic_mode"].dropna().astype(str).unique()) <= {"fixed", "knee"}
    for payload in channels_df["fooof_peak_params_json"].dropna().astype(str):
        parsed = json.loads(payload)
        assert isinstance(parsed, list)


def test_fooof_cli_rejects_resampled_dataset_index(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    monkeypatch.setenv("MNE_DONTWRITE_HOME", "true")
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")

    import mne
    from rate.cli.main import main
    from rate.util import write_parquet_dataframe

    out_dir = tmp_path / "outputs"
    (out_dir / "dataset").mkdir(parents=True, exist_ok=True)
    (out_dir / "events").mkdir(parents=True, exist_ok=True)

    sfreq = 100.0
    t = np.arange(0.0, 4.0, 1.0 / sfreq)
    data = np.vstack([np.sin(2 * np.pi * 10.0 * t), np.sin(2 * np.pi * 8.0 * t)])
    info = mne.create_info(ch_names=["Fz", "Cz"], sfreq=sfreq, ch_types=["eeg", "eeg"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    resampled_dir = out_dir / "eeg_resampled"
    resampled_dir.mkdir(parents=True, exist_ok=True)
    fif_path = resampled_dir / "sub-001_run-1_32Hz_raw.fif"
    raw.save(fif_path, overwrite=True, verbose="ERROR")

    dataset_index = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-001",
                "run": 1,
                "task": "conversation",
                "eeg_path": str(fif_path),
                "status": "ok",
                "eeg_is_resampled": True,
                "eeg_downsample_hz": 32.0,
            }
        ]
    )
    write_parquet_dataframe(dataset_index, out_dir / "dataset" / "index_resampled.parquet")

    events_df = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-001",
                "run": 1,
                "task": "conversation",
                "event_id": "dyad-001_1_conversation_ev00001",
                "t_onset": 2.0,
            }
        ]
    )
    write_parquet_dataframe(events_df, out_dir / "events" / "events.parquet")

    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _make_config(out_dir))

    with pytest.raises((FileNotFoundError, ValueError), match="FOOOF.*non-downsampled|index_resampled"):
        main(["fooof", "--config", str(cfg_path), "--out-dir", str(out_dir)])
