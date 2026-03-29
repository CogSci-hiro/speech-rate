"""Unit tests for the subject-level FOOOF-mass workflow."""

from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import mne
import numpy as np
import pandas as pd

from rate.cli.commands import fooof_mass_subject
from rate.cli.commands import fooof_epochs_subject
from rate.fooof_mass.features import FooofFeatureResult, extract_fooof_features, list_fooof_epoch_feature_names
from rate.fooof_mass.fit import FooofTensorResult
from rate.fooof_mass.glm import fit_fooof_mass_glm
from rate.fooof_mass.io import build_fooof_epoch_feature_paths, build_fooof_epochs_subject_summary_path, build_fooof_mass_output_paths
from rate.fooof_mass.windows import make_sliding_windows
from rate.util import write_parquet_dataframe


def _make_epochs(data: np.ndarray, metadata: pd.DataFrame, *, sfreq: float = 100.0, tmin: float = -1.0) -> mne.EpochsArray:
    info = mne.create_info(["Fz", "Cz"], sfreq=sfreq, ch_types=["eeg", "eeg"])
    events = np.column_stack([np.arange(len(data), dtype=int), np.zeros(len(data), dtype=int), np.ones(len(data), dtype=int)])
    return mne.EpochsArray(data, info, events=events, tmin=tmin, metadata=metadata, verbose="ERROR")


def test_make_sliding_windows_is_deterministic() -> None:
    times = np.linspace(-1.0, 1.0, 201)
    config = {"fooof": {"windows": {"start_s": -0.5, "stop_s": 0.5, "length_s": 0.2, "sfreq_hz": 10.0}}}

    windows = make_sliding_windows(times, 100.0, config)

    assert len(windows) == 9
    assert windows[0].start_sample == 50
    assert windows[0].stop_sample == 70
    assert windows[0].center_s == 0.5 * (windows[0].start_s + windows[0].stop_s)
    assert np.isclose(windows[-1].start_s, 0.3)


def test_extract_fooof_features_selects_highest_power_peak_within_band() -> None:
    fit_result = FooofTensorResult(
        frequencies=np.array([4.0, 8.0, 12.0]),
        success_mask=np.ones((1, 1, 1), dtype=bool),
        aperiodic_params=np.array([[[[1.5, 0.75, np.nan]]]]),
        peak_params=np.array([[[[[5.5, 0.2, 1.0], [6.0, 0.4, 1.2], [10.0, 0.3, 1.5]]]]]),
        error_values=np.zeros((1, 1, 1), dtype=float),
        r_squared_values=np.ones((1, 1, 1), dtype=float),
        n_attempted=1,
        n_failed=0,
        max_n_peaks=3,
    )
    config = {"fooof": {"bands": {"theta": [4.0, 7.0], "alpha": [8.0, 12.0]}}}

    result = extract_fooof_features(fit_result, config)

    assert result.feature_tensors["offset"][0, 0, 0] == 1.5
    assert result.feature_tensors["exponent"][0, 0, 0] == 0.75
    assert result.feature_tensors["theta"][0, 0, 0] == 0.4
    assert result.feature_tensors["theta_cf"][0, 0, 0] == 6.0
    assert result.feature_tensors["theta_pw"][0, 0, 0] == 0.4
    assert result.feature_tensors["alpha_cf"][0, 0, 0] == 10.0


def test_extract_fooof_features_returns_zero_when_no_periodic_peak_is_found() -> None:
    fit_result = FooofTensorResult(
        frequencies=np.array([4.0, 8.0, 12.0]),
        success_mask=np.ones((1, 1, 1), dtype=bool),
        aperiodic_params=np.array([[[[1.5, 0.75, np.nan]]]]),
        peak_params=np.full((1, 1, 1, 1, 3), np.nan, dtype=float),
        error_values=np.zeros((1, 1, 1), dtype=float),
        r_squared_values=np.ones((1, 1, 1), dtype=float),
        n_attempted=1,
        n_failed=0,
        max_n_peaks=1,
    )
    config = {"fooof": {"bands": {"theta": [4.0, 7.0], "alpha": [8.0, 12.0]}}}

    result = extract_fooof_features(fit_result, config)

    assert result.feature_tensors["theta"][0, 0, 0] == 0.0
    assert result.feature_tensors["theta_cf"][0, 0, 0] == 0.0
    assert result.feature_tensors["theta_bw"][0, 0, 0] == 0.0
    assert result.feature_tensors["alpha"][0, 0, 0] == 0.0


def test_list_fooof_epoch_feature_names_prefers_configured_list() -> None:
    config = {"fooof": {"bands": {"theta": [4.0, 7.0]}, "features": ["offset", "theta"]}}

    assert list_fooof_epoch_feature_names(config) == ["offset", "theta"]


def test_fit_fooof_mass_glm_handles_targetwise_missing_values() -> None:
    speech_rate = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    metadata = pd.DataFrame({"speech_rate": speech_rate})
    feature_tensor = np.zeros((4, 1, 2), dtype=float)
    feature_tensor[:, 0, 0] = 1.0 + (2.0 * speech_rate)
    feature_tensor[:, 0, 1] = 5.0 + (3.0 * speech_rate)
    feature_tensor[2, 0, 1] = np.nan

    config = {
        "fooof": {
            "models": {
                "theta_cf_other_rate": {
                    "outcome": "theta_cf",
                    "formula": "theta_cf ~ speech_rate",
                    "test_predictors": ["speech_rate"],
                }
            }
        }
    }

    result = fit_fooof_mass_glm(
        feature_tensor,
        metadata,
        config,
        model_name="theta_cf_other_rate",
        channel_names=["Fz"],
        window_centers_s=np.array([0.0, 0.1]),
    )

    assert result.betas.shape == (2, 1, 2)
    assert np.allclose(result.betas[0, 0, 0], 1.0)
    assert np.allclose(result.betas[1, 0, 0], 2.0)
    assert np.allclose(result.betas[0, 0, 1], 5.0)
    assert np.allclose(result.betas[1, 0, 1], 3.0)
    assert result.valid_counts.tolist() == [[4, 3]]


def test_fooof_mass_cli_writes_outputs_with_synthetic_fit(monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = {
        "paths": {"out_dir": str(out_dir)},
        "epochs": {},
        "fooof": {
            "windows": {"start_s": -0.5, "stop_s": 0.0, "length_s": 0.2, "sfreq_hz": 5.0},
            "spectrum": {"method": "welch", "fit_range_hz": [3.0, 20.0], "welch_n_per_seg_s": 0.2, "welch_overlap": 0.0},
            "bands": {"theta": [4.0, 7.0]},
            "models": {
                "theta_cf_other_rate": {
                    "outcome": "theta_cf",
                    "formula": "theta_cf ~ speech_rate",
                    "test_predictors": ["speech_rate"],
                }
            },
        },
    }
    row = {"dyad_id": "dyad-001", "subject_id": "sub-001", "task": "conversation", "run": "1", "status": "ok", "eeg_path": str(tmp_path / "placeholder_raw.fif")}
    metadata = pd.DataFrame({"speech_rate": [0.0, 1.0, 2.0]})
    data = np.random.default_rng(42).normal(scale=1e-6, size=(3, 2, 101))
    epochs = _make_epochs(data, metadata.copy(), tmin=-0.5)
    epochs_path = tmp_path / "epochs.fif"
    metadata_path = tmp_path / "metadata.csv"
    dataset_index_path = tmp_path / "index.parquet"
    epochs.save(epochs_path, overwrite=True, verbose="ERROR")
    metadata.to_csv(metadata_path, index=False)
    write_parquet_dataframe(pd.DataFrame([row]), dataset_index_path)

    def fake_fit_fooof_tensor(frequencies, psd_tensor, config):
        n_epochs, n_channels, n_windows, _ = psd_tensor.shape
        return FooofTensorResult(
            frequencies=np.asarray(frequencies, dtype=float),
            success_mask=np.ones((n_epochs, n_channels, n_windows), dtype=bool),
            aperiodic_params=np.zeros((n_epochs, n_channels, n_windows, 3), dtype=float),
            peak_params=np.zeros((n_epochs, n_channels, n_windows, 1, 3), dtype=float),
            error_values=np.zeros((n_epochs, n_channels, n_windows), dtype=float),
            r_squared_values=np.ones((n_epochs, n_channels, n_windows), dtype=float),
            n_attempted=int(n_epochs * n_channels * n_windows),
            n_failed=0,
            max_n_peaks=1,
        )

    def fake_extract_fooof_features(fit_result, config):
        n_epochs, n_channels, n_windows = fit_result.success_mask.shape
        speech_rate = metadata["speech_rate"].to_numpy(dtype=float)
        theta_cf = np.zeros((n_epochs, n_channels, n_windows), dtype=float)
        for window_index in range(n_windows):
            theta_cf[:, :, window_index] = (1.0 + window_index) + speech_rate[:, np.newaxis]
        return FooofFeatureResult(
            feature_tensors={"theta_cf": theta_cf, "offset": np.zeros_like(theta_cf), "exponent": np.zeros_like(theta_cf)},
            valid_fraction_by_feature={"theta_cf": 1.0, "offset": 1.0, "exponent": 1.0},
            success_mask=fit_result.success_mask,
        )

    monkeypatch.setattr(fooof_mass_subject, "fit_fooof_tensor", fake_fit_fooof_tensor)
    monkeypatch.setattr(fooof_mass_subject, "extract_fooof_features", fake_extract_fooof_features)

    args = Namespace(
        config="config/config.yaml",
        schema="config/schema.json",
        dataset_index=str(dataset_index_path),
        subject_id="sub-001",
        task="conversation",
        run="1",
        model_name="theta_cf_other_rate",
        epochs_path=str(epochs_path),
        metadata_path=str(metadata_path),
        run_all=False,
        output=None,
    )
    exit_code = fooof_mass_subject.run(args, config=config)
    assert exit_code == 0

    output_paths = build_fooof_mass_output_paths(
        config,
        row,
        model_name="theta_cf_other_rate",
        outcome_name="theta_cf",
        epochs_path=epochs_path,
        metadata_path=metadata_path,
    )
    assert output_paths.summary_output_path.exists()
    assert output_paths.betas_output_path.exists()
    assert output_paths.feature_tensor_output_path.exists()

    summary = json.loads(output_paths.summary_output_path.read_text(encoding="utf-8"))
    betas = np.load(output_paths.betas_output_path)
    valid_counts = np.load(output_paths.valid_counts_output_path)

    assert summary["status"] == "ok"
    assert summary["n_fooof_failed"] == 0
    assert summary["outcome_name"] == "theta_cf"
    assert betas.shape[0] == 2
    assert valid_counts.min() == 3


def test_fooof_epochs_cli_writes_feature_epoch_outputs(monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = {
        "paths": {"out_dir": str(out_dir)},
        "epochs": {},
        "margin_epochs": {
            "event_column": "self_ipu_onset",
            "output": {
                "epochs_filename": "margin-epochs-{lock_name}.fif",
                "metadata_filename": "margin-metadata-{lock_name}.csv",
                "events_array_filename": "margin-events-{lock_name}.npy",
                "summary_filename": "margin-epoching_summary-{lock_name}.json",
            },
        },
        "fooof": {
            "windows": {"start_s": -0.5, "stop_s": 0.0, "length_s": 0.2, "sfreq_hz": 5.0},
            "spectrum": {"method": "welch", "fit_range_hz": [3.0, 20.0], "welch_n_per_seg_s": 0.2, "welch_overlap": 0.0},
            "bands": {"theta": [4.0, 7.0], "alpha": [8.0, 12.0]},
            "features": ["offset", "theta"],
        },
    }
    row = {"dyad_id": "dyad-001", "subject_id": "sub-001", "task": "conversation", "run": "1", "status": "ok", "eeg_path": str(tmp_path / "placeholder_raw.fif")}
    metadata = pd.DataFrame({"speech_rate": [0.0, 1.0, 2.0]})
    data = np.random.default_rng(42).normal(scale=1e-6, size=(3, 2, 101))
    epochs = _make_epochs(data, metadata.copy(), tmin=-0.5)
    epochs_path = tmp_path / "margin-epochs.fif"
    metadata_path = tmp_path / "margin-metadata.csv"
    dataset_index_path = tmp_path / "index.parquet"
    epochs.save(epochs_path, overwrite=True, verbose="ERROR")
    metadata.to_csv(metadata_path, index=False)
    write_parquet_dataframe(pd.DataFrame([row]), dataset_index_path)

    def fake_fit_fooof_tensor(frequencies, psd_tensor, config):
        n_epochs, n_channels, n_windows, _ = psd_tensor.shape
        return FooofTensorResult(
            frequencies=np.asarray(frequencies, dtype=float),
            success_mask=np.ones((n_epochs, n_channels, n_windows), dtype=bool),
            aperiodic_params=np.zeros((n_epochs, n_channels, n_windows, 3), dtype=float),
            peak_params=np.zeros((n_epochs, n_channels, n_windows, 1, 3), dtype=float),
            error_values=np.zeros((n_epochs, n_channels, n_windows), dtype=float),
            r_squared_values=np.ones((n_epochs, n_channels, n_windows), dtype=float),
            n_attempted=int(n_epochs * n_channels * n_windows),
            n_failed=0,
            max_n_peaks=1,
        )

    def fake_extract_fooof_features(fit_result, config):
        n_epochs, n_channels, n_windows = fit_result.success_mask.shape
        offset = np.zeros((n_epochs, n_channels, n_windows), dtype=float)
        theta = np.zeros((n_epochs, n_channels, n_windows), dtype=float)
        for window_index in range(n_windows):
            offset[:, :, window_index] = 1.0 + window_index
            theta[:, :, window_index] = 5.0 + window_index
        return FooofFeatureResult(
            feature_tensors={"offset": offset, "theta": theta},
            valid_fraction_by_feature={"offset": 1.0, "theta": 1.0},
            success_mask=fit_result.success_mask,
        )

    monkeypatch.setattr(fooof_epochs_subject, "fit_fooof_tensor", fake_fit_fooof_tensor)
    monkeypatch.setattr(fooof_epochs_subject, "extract_fooof_features", fake_extract_fooof_features)

    args = Namespace(
        config="config/config.yaml",
        schema="config/schema.json",
        dataset_index=str(dataset_index_path),
        dyad_id="dyad-001",
        subject_id="sub-001",
        epochs_path=str(epochs_path),
        metadata_path=str(metadata_path),
        run_all=False,
        output=None,
    )
    exit_code = fooof_epochs_subject.run(args, config=config)
    assert exit_code == 0

    offset_paths = build_fooof_epoch_feature_paths(config, row, feature_name="offset")
    theta_paths = build_fooof_epoch_feature_paths(config, row, feature_name="theta")
    subject_summary_path = build_fooof_epochs_subject_summary_path(config, row)

    assert offset_paths.epochs_output_path.exists()
    assert offset_paths.summary_output_path.exists()
    assert theta_paths.epochs_output_path.exists()
    assert theta_paths.summary_output_path.exists()
    assert subject_summary_path.exists()

    offset_epochs = mne.read_epochs(offset_paths.epochs_output_path, preload=True, verbose="ERROR")
    assert offset_epochs.get_data().shape[2] == 2
    assert offset_epochs.tmin == -0.4
