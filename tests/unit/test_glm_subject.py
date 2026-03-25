"""Unit tests for subject-level GLM fitting and CLI wiring."""

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

from rate.cli.commands import glm_subject
from rate.design import build_design_matrix
from rate.glm.fit import fit_subject_glm
from rate.glm.io import build_glm_output_paths
from rate.util import write_parquet_dataframe


def _make_epochs(data: np.ndarray, metadata: pd.DataFrame) -> mne.EpochsArray:
    info = mne.create_info(["Fz", "Cz"], sfreq=100.0, ch_types=["eeg", "eeg"])
    events = np.column_stack(
        [
            np.arange(len(data), dtype=int),
            np.zeros(len(data), dtype=int),
            np.ones(len(data), dtype=int),
        ]
    )
    return mne.EpochsArray(data, info, events=events, tmin=-0.1, metadata=metadata, verbose="ERROR")


def _glm_config(out_dir: Path) -> dict:
    return {
        "paths": {"out_dir": str(out_dir)},
        "epochs": {},
        "glm": {
            "models": {
                "subject_level": {
                    "formula": "~ speech_rate",
                    "test_predictors": ["speech_rate"],
                }
            }
        },
    }


def test_fit_subject_glm_recovers_known_betas_with_design_row_filtering() -> None:
    intercept_map = np.asarray([[1.0, 2.0, 3.0], [0.5, 0.25, -0.5]])
    slope_map = np.asarray([[0.1, 0.2, 0.3], [0.4, -0.2, 0.05]])
    speech_rate = np.asarray([0.0, 1.0, np.nan, 3.0], dtype=float)

    data = np.zeros((4, 2, 3), dtype=float)
    for epoch_index, predictor_value in enumerate(speech_rate):
        value = 0.0 if np.isnan(predictor_value) else predictor_value
        data[epoch_index] = intercept_map + value * slope_map

    metadata = pd.DataFrame({"speech_rate": speech_rate})
    config = {"glm": {"models": {"subject_level": {"formula": "~ speech_rate", "test_predictors": ["speech_rate"]}}}}
    design_result = build_design_matrix(metadata, config=config, section="glm", model_name="subject_level")
    epochs = _make_epochs(data, metadata.fillna({"speech_rate": -99.0}))

    fit_result = fit_subject_glm(epochs, design_result)

    assert fit_result.n_epochs_input == 4
    assert fit_result.n_epochs_used == 3
    assert fit_result.design_matrix_shape == (3, 2)
    assert fit_result.target_matrix_shape == (3, 6)
    assert np.allclose(fit_result.betas[0], intercept_map)
    assert np.allclose(fit_result.betas[1], slope_map)
    assert fit_result.row_index.tolist() == [0, 1, 3]


def test_glm_subject_cli_writes_subject_level_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = _glm_config(out_dir)
    row_run_1 = {
        "dyad_id": "dyad-001",
        "subject_id": "sub-001",
        "task": "conversation",
        "run": "1",
        "status": "ok",
        "eeg_path": str(tmp_path / "placeholder_raw_1.fif"),
    }
    row_run_2 = {
        "dyad_id": "dyad-001",
        "subject_id": "sub-001",
        "task": "conversation",
        "run": "2",
        "status": "ok",
        "eeg_path": str(tmp_path / "placeholder_raw_2.fif"),
    }

    metadata = pd.DataFrame({"speech_rate": [0.0, 1.0, 2.0, 3.0]})
    intercept_map = np.asarray([[1.0, 1.5, 2.0], [0.0, -0.5, -1.0]])
    slope_map = np.asarray([[0.5, 0.25, -0.25], [0.2, 0.1, 0.0]])
    data = np.stack([intercept_map + value * slope_map for value in metadata["speech_rate"]], axis=0)
    epochs = _make_epochs(data, metadata.copy())

    epochs_path = tmp_path / "epochs.fif"
    metadata_path = tmp_path / "metadata.csv"
    dataset_index_path = tmp_path / "index.parquet"
    epochs.save(epochs_path, overwrite=True, verbose="ERROR")
    metadata.to_csv(metadata_path, index=False)
    write_parquet_dataframe(pd.DataFrame([row_run_1, row_run_2]), dataset_index_path)

    args = Namespace(
        config="config/config.yaml",
        schema="config/schema.json",
        dataset_index=str(dataset_index_path),
        subject_id="sub-001",
        model_name="subject_level",
        epochs_path=str(epochs_path),
        metadata_path=str(metadata_path),
        run_all=False,
        output=None,
    )

    exit_code = glm_subject.run(args, config=config)
    assert exit_code == 0

    output_paths = build_glm_output_paths(
        config,
        row_run_1,
        model_name="subject_level",
        epochs_path=epochs_path,
        metadata_path=metadata_path,
    )
    assert output_paths.summary_output_path.exists()
    assert output_paths.betas_output_path.exists()
    assert output_paths.predictor_betas_output_path.exists()
    assert output_paths.selection_output_path.exists()

    summary = json.loads(output_paths.summary_output_path.read_text(encoding="utf-8"))
    betas = np.load(output_paths.betas_output_path)
    predictor_betas = np.load(output_paths.predictor_betas_output_path)

    assert summary["status"] == "ok"
    assert summary["design_matrix_shape"] == [4, 2]
    assert summary["target_matrix_shape"] == [4, 6]
    assert summary["test_predictors"] == ["speech_rate"]
    assert predictor_betas.files == ["speech_rate"]
    assert betas.shape == (2, 2, 3)
    assert np.allclose(betas[1], slope_map)
    assert output_paths.summary_output_path == out_dir / "glm" / "sub-001" / "subject_level" / "summary.json"


def test_glm_subject_cli_writes_skipped_outputs_when_no_epochs_survive_filtering(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = {
        "paths": {"out_dir": str(out_dir)},
        "epochs": {},
        "glm": {
            "models": {
                "subject_level": {
                    "formula": "~ other_rate",
                    "eligibility": {"not_null": ["other_rate"]},
                    "test_predictors": ["other_rate"],
                }
            }
        },
    }
    row = {
        "dyad_id": "dyad-001",
        "subject_id": "sub-001",
        "task": "conversation",
        "run": "1",
        "status": "ok",
        "eeg_path": str(tmp_path / "placeholder_raw_1.fif"),
    }

    metadata = pd.DataFrame({"other_rate": [np.nan, np.nan, np.nan]})
    data = np.zeros((3, 2, 3), dtype=float)
    epochs = _make_epochs(data, metadata.fillna({"other_rate": -99.0}))

    epochs_path = tmp_path / "epochs.fif"
    metadata_path = tmp_path / "metadata.csv"
    dataset_index_path = tmp_path / "index.parquet"
    epochs.save(epochs_path, overwrite=True, verbose="ERROR")
    metadata.to_csv(metadata_path, index=False)
    write_parquet_dataframe(pd.DataFrame([row]), dataset_index_path)

    args = Namespace(
        config="config/config.yaml",
        schema="config/schema.json",
        dataset_index=str(dataset_index_path),
        subject_id="sub-001",
        model_name="subject_level",
        epochs_path=str(epochs_path),
        metadata_path=str(metadata_path),
        run_all=False,
        output=None,
    )

    exit_code = glm_subject.run(args, config=config)
    assert exit_code == 0

    output_paths = build_glm_output_paths(
        config,
        row,
        model_name="subject_level",
        epochs_path=epochs_path,
        metadata_path=metadata_path,
    )
    summary = json.loads(output_paths.summary_output_path.read_text(encoding="utf-8"))
    betas = np.load(output_paths.betas_output_path)

    assert summary["status"] == "skipped"
    assert summary["reason"] == "No epochs remain after design filtering."
    assert summary["n_epochs_used"] == 0
    assert betas.shape == (2, 2, 3)
    assert np.isnan(betas).all()
