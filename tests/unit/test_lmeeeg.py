"""Unit tests for the subject-level lmeEEG approximation stage."""

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
import yaml

from rate.cli.commands import lmeeeg_group_stats
from rate.cli.commands import lmeeeg_subject
from rate.epochs.io import build_epoch_output_paths
from rate.lmeeeg.io import build_lmeeeg_output_paths
from rate.lmeeeg.mass_model import fit_lmeeeg_mass_model
from rate.lmeeeg.trial_data import build_lmeeeg_trial_table
from rate.util import write_parquet_dataframe


def _make_epochs(data: np.ndarray, metadata: pd.DataFrame) -> mne.EpochsArray:
    info = mne.create_info(["Fz", "Cz"], sfreq=100.0, ch_types=["eeg", "eeg"])
    events = np.column_stack([np.arange(len(data), dtype=int), np.zeros(len(data), dtype=int), np.ones(len(data), dtype=int)])
    return mne.EpochsArray(data, info, events=events, tmin=-0.1, metadata=metadata, verbose="ERROR")


def _lmeeeg_config(out_dir: Path) -> dict:
    return {
        "paths": {"out_dir": str(out_dir)},
        "epochs": {},
        "lmeeeg": {
            "backend": "lmeeeg",
            "random_effects": {"group": "item_id", "structure": "random_intercept"},
            "models": {
                "speech_rate_item_intercept": {
                    "formula": "signal ~ speech_rate",
                    "test_predictors": ["speech_rate"],
                    "random_effect_group": "item_id",
                    "random_effect_structure": "random_intercept",
                }
            },
        },
    }


def test_fit_lmeeeg_mass_model_recovers_slope_after_random_intercept_removal() -> None:
    speech_rate = np.array([0.0, 1.0, 2.0, 0.5, 1.5, 2.5], dtype=float)
    item_id = np.array(["A", "A", "A", "B", "B", "B"], dtype=object)
    group_intercepts = {"A": 5.0, "B": -3.0}
    slope_map = np.array([[2.0, 1.0], [-1.0, 0.5]])
    intercept_map = np.array([[1.0, 0.0], [2.0, -2.0]])

    data = np.zeros((6, 2, 2), dtype=float)
    for index, (rate, group) in enumerate(zip(speech_rate, item_id, strict=True)):
        data[index] = intercept_map + (rate * slope_map) + group_intercepts[group]

    metadata = pd.DataFrame({"speech_rate": speech_rate, "item_id": item_id})
    epochs = _make_epochs(data, metadata.copy())
    config = _lmeeeg_config(Path("/tmp/unused"))

    trial_data = build_lmeeeg_trial_table(epochs, metadata, config, model_name="speech_rate_item_intercept")
    result = fit_lmeeeg_mass_model(trial_data, config, model_name="speech_rate_item_intercept")

    assert result.n_lmm_failed == 0
    assert result.residualised_data.shape == (6, 2, 2)
    assert result.betas.shape == (2, 2, 2)
    assert np.allclose(result.betas[1], slope_map, atol=0.2)


def test_lmeeeg_subject_cli_writes_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = _lmeeeg_config(out_dir)
    row = {
        "dyad_id": "dyad-001",
        "subject_id": "sub-001",
        "task": "conversation",
        "run": "1",
        "status": "ok",
        "eeg_path": str(tmp_path / "placeholder_raw.fif"),
    }

    metadata = pd.DataFrame(
        {
            "speech_rate": [0.0, 1.0, 2.0, 0.5, 1.5, 2.5],
            "item_id": ["A", "A", "A", "B", "B", "B"],
        }
    )
    slope_map = np.array([[1.5, -0.5], [0.25, 0.75]])
    intercept_map = np.array([[2.0, 1.0], [-1.0, 0.5]])
    group_intercepts = {"A": 4.0, "B": -2.0}
    data = np.zeros((6, 2, 2), dtype=float)
    for index, row_values in metadata.iterrows():
        data[index] = intercept_map + (float(row_values["speech_rate"]) * slope_map) + group_intercepts[str(row_values["item_id"])]

    epochs = _make_epochs(data, metadata.copy())
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
        task="conversation",
        run="1",
        model_name="speech_rate_item_intercept",
        epochs_path=str(epochs_path),
        metadata_path=str(metadata_path),
        run_all=False,
        output=None,
    )
    exit_code = lmeeeg_subject.run(args, config=config)
    assert exit_code == 0

    output_paths = build_lmeeeg_output_paths(config, row, model_name="speech_rate_item_intercept", epochs_path=epochs_path, metadata_path=metadata_path)
    assert output_paths.summary_output_path.exists()
    assert output_paths.betas_output_path.exists()
    assert output_paths.residualised_trials_output_path.exists()
    assert output_paths.lmm_diagnostics_output_path.exists()

    summary = json.loads(output_paths.summary_output_path.read_text(encoding="utf-8"))
    betas = np.load(output_paths.betas_output_path)

    assert summary["status"] == "ok"
    assert summary["backend"] == "lmeeeg"
    assert summary["random_effect_structure"] == "random_intercept"
    assert summary["n_lmm_failed"] == 0
    assert betas.shape == (2, 2, 2)


def test_lmeeeg_group_stats_cli_writes_real_inference_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = _lmeeeg_config(out_dir)
    row = {
        "dyad_id": "dyad-001",
        "subject_id": "sub-001",
        "task": "conversation",
        "run": "1",
        "status": "ok",
        "eeg_path": str(tmp_path / "placeholder_raw.fif"),
    }
    metadata = pd.DataFrame(
        {
            "speech_rate": [0.0, 1.0, 2.0, 0.5, 1.5, 2.5],
            "item_id": ["A", "A", "A", "B", "B", "B"],
        }
    )
    slope_map = np.array([[1.0, 0.0], [0.5, -0.25]])
    intercept_map = np.array([[1.0, 1.5], [-0.5, 0.25]])
    group_intercepts = {"A": 3.0, "B": -1.0}
    data = np.zeros((6, 2, 2), dtype=float)
    for index, row_values in metadata.iterrows():
        data[index] = intercept_map + (float(row_values["speech_rate"]) * slope_map) + group_intercepts[str(row_values["item_id"])]

    epochs = _make_epochs(data, metadata.copy())
    epoch_paths = build_epoch_output_paths(config, row)
    epochs_path = epoch_paths.epochs_output_path
    metadata_path = epoch_paths.metadata_output_path
    dataset_index_path = out_dir / "dataset" / "index.parquet"
    epochs_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_index_path.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(epochs_path, overwrite=True, verbose="ERROR")
    metadata.to_csv(metadata_path, index=False)
    write_parquet_dataframe(pd.DataFrame([row]), dataset_index_path)

    stats_cfg_path = tmp_path / "stats.yaml"
    stats_cfg_path.write_text(
        yaml.safe_dump(
            {
                "stats": {
                    "method": "maxstat",
                    "n_permutations": 8,
                    "tail": 0,
                    "adjacency": None,
                    "outputs": {"lmeeeg": "stats/lmeeeg_group_stats.json"},
                }
            }
        ),
        encoding="utf-8",
    )

    args = Namespace(
        config="config/config.yaml",
        schema="config/schema.json",
        stats_config=str(stats_cfg_path),
        dataset_index=str(dataset_index_path),
        subject_id="sub-001",
        task="conversation",
        run="1",
        model_name="speech_rate_item_intercept",
        output=None,
    )
    exit_code = lmeeeg_group_stats.run(args, config=config)
    assert exit_code == 0

    summary_path = out_dir / "stats" / "lmeeeg_group_stats.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["status"] == "ok"
    assert summary["method"] == "maxstat"
    assert summary["n_analyses"] == 1
    effect_summary = summary["analyses"][0]["effects"][0]
    assert effect_summary["predictor"] == "speech_rate"
    assert Path(effect_summary["observed_statistic"]).exists()
    assert Path(effect_summary["corrected_p_values"]).exists()
