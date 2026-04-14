"""Unit tests for the residualised decoding pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import mne
import numpy as np
import pandas as pd
import yaml

from rate.decoding.config import load_decoding_config
from rate.decoding.io import build_decoding_model_output_paths
from rate.decoding.pipeline import _residualise_target, resolve_ridge_alphas, run_decoding_model
from rate.decoding.stats import compute_tfce_2d, permute_decoding_targets, run_decoding_model_stats
from rate.epochs.io import build_epoch_output_paths
from rate.util import write_parquet_dataframe


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _make_epochs(data: np.ndarray, metadata: pd.DataFrame, *, sfreq: float = 64.0, tmin: float = -0.1) -> mne.EpochsArray:
    info = mne.create_info(["Fz", "Cz"], sfreq=sfreq, ch_types=["eeg", "eeg"])
    events = np.column_stack([np.arange(len(data), dtype=int), np.zeros(len(data), dtype=int), np.ones(len(data), dtype=int)])
    return mne.EpochsArray(data, info, events=events, tmin=tmin, metadata=metadata, verbose="ERROR")


def _project_config(out_dir: Path) -> dict:
    return {
        "project": {"random_seed": 123},
        "paths": {"out_dir": str(out_dir)},
        "epochs": {},
    }


def _decoding_config() -> dict:
    return {
        "enabled": True,
        "analysis_name": "decoding_temporal_generalisation",
        "output_dir": "decoding",
        "sfreq": 32,
        "random_state": 123,
        "input": {"epochs_section": "epochs"},
        "channels": {"picks": "eeg", "include": [], "exclude": []},
        "time_window": {"tmin_s": None, "tmax_s": None},
        "controls": ["latency", "duration_s", "time_within_run", "run"],
        "eeg_residualisation": {
            "random_effect": "subject_id",
            "controls": ["latency", "duration_s", "time_within_run", "run"],
        },
        "target_residualisation": {
            "random_effect": "subject_id",
            "controls": ["latency", "duration_s", "time_within_run", "run"],
        },
        "cv": {"n_splits": 2, "group_column": "subject_id", "shuffle": True, "random_state": 123},
        "nested_cv": {
            "enabled": True,
            "inner_n_splits": 2,
            "selection_metric": "mean_diagonal_pearson_r",
            "select_single_alpha_per_outer_fold": True,
        },
        "estimator": {"name": "ridge", "alpha_grid": [0.1, 1.0]},
        "scoring": {"name": "pearson_r"},
        "stats": {
            "enabled": True,
            "n_permutations": 8,
            "random_state": 123,
            "permutation_scheme": "within_subject",
            "n_jobs": 1,
            "tfce": {"enabled": True, "h_power": 2.0, "e_power": 0.5, "h_start": 0.0, "dh": 0.1, "tail": "positive"},
        },
        "models": [
            {"name": "self_temporal_generalisation", "train_target": "self_rate", "test_target": "self_rate"},
            {"name": "cross_other_to_self", "train_target": "other_rate", "test_target": "self_rate"},
        ],
    }


def test_load_decoding_config_applies_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "decoding.yaml"
    _write_yaml(
        config_path,
        {
            "decoding": {
                "models": [
                    {"name": "self_temporal_generalisation", "train_target": "self_rate", "test_target": "self_rate"}
                ]
            }
        },
    )

    resolved = load_decoding_config(config_path, project_config={"project": {"random_seed": 77}})

    assert resolved["random_state"] == 77
    assert resolved["cv"]["group_column"] == "subject_id"
    assert resolved["cv"]["random_state"] == 77
    assert resolved["estimator"]["name"] == "ridge"
    assert resolved["nested_cv"]["selection_metric"] == "mean_diagonal_pearson_r"
    assert resolved["stats"]["permutation_scheme"] == "within_subject"


def test_load_decoding_config_accepts_explicit_alpha_grid(tmp_path: Path) -> None:
    config_path = tmp_path / "decoding.yaml"
    _write_yaml(
        config_path,
        {
            "decoding": {
                "estimator": {"alpha_grid": [0.01, 0.1, 1.0]},
                "models": [
                    {"name": "self_temporal_generalisation", "train_target": "self_rate", "test_target": "self_rate"}
                ],
            }
        },
    )

    resolved = load_decoding_config(config_path)

    assert resolved["estimator"]["alpha_grid"] == [0.01, 0.1, 1.0]


def test_resolve_ridge_alphas_supports_logspace_grid() -> None:
    alphas = resolve_ridge_alphas({"name": "ridge", "alphas": {"start": 1, "stop": 3, "n_alphas": 3}})

    assert np.allclose(alphas, np.logspace(1, 3, 3))


def test_resolve_ridge_alphas_supports_explicit_alpha_grid() -> None:
    alphas = resolve_ridge_alphas({"name": "ridge", "alpha_grid": [0.01, 0.1, 1.0]})

    assert np.allclose(alphas, np.array([0.01, 0.1, 1.0], dtype=float))


def test_tfce_2d_boosts_connected_positive_clusters() -> None:
    scores = np.array([[0.0, 0.5, 0.5], [0.0, 0.5, 0.0], [0.2, 0.0, 0.0]], dtype=float)

    tfce = compute_tfce_2d(scores, h_power=2.0, e_power=0.5, h_start=0.0, dh=0.1, tail="positive")

    assert tfce.shape == scores.shape
    assert tfce[0, 1] > tfce[2, 0]
    assert tfce[1, 1] > 0.0


def test_target_residualisation_excludes_target_from_its_own_control_set() -> None:
    metadata = pd.DataFrame(
        {
            "subject_id": ["s1", "s1", "s2", "s2"],
            "self_rate": [1.0, 2.5, 3.0, 5.5],
            "run": [1, 2, 1, 2],
        }
    )

    residuals = _residualise_target(
        metadata,
        target_name="self_rate",
        random_effect="subject_id",
        controls=["self_rate", "run"],
    )

    assert not np.allclose(residuals, 0.0)
    assert residuals.shape == (4,)


def test_run_decoding_model_writes_temporal_generalisation_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    project_config = _project_config(out_dir)
    decoding_config = _decoding_config()

    rows: list[dict[str, str]] = []
    dataset_index_path = out_dir / "dataset" / "index.parquet"
    signal_map = np.array(
        [
            [0.8, -0.2, 0.6],
            [0.4, 0.7, -0.3],
        ],
        dtype=float,
    )
    base_self = np.array([-1.4, 0.2, 1.1, -0.6, 0.9, 1.7], dtype=float)
    base_other = np.array([0.5, -0.8, 0.7, 1.3, -0.2, 0.4], dtype=float)

    for subject_offset, subject_id in enumerate(("sub-001", "sub-002", "sub-003", "sub-004"), start=1):
        self_rate = base_self + (subject_offset * 0.15)
        other_rate = (0.6 * self_rate) + base_other
        metadata = pd.DataFrame(
            {
                "event_id": np.arange(6, dtype=int) + (subject_offset * 100),
                "subject_id": [subject_id] * 6,
                "dyad_id": [f"dyad-{subject_offset:03d}"] * 6,
                "task": ["conversation"] * 6,
                "run": ["1"] * 6,
                "latency": np.linspace(0.1, 0.4, 6),
                "duration_s": np.linspace(0.3, 0.8, 6),
                "time_within_run": np.linspace(0.0, 1.0, 6),
                "self_rate": self_rate,
                "other_rate": other_rate,
            }
        )
        data = np.zeros((6, 2, 3), dtype=float)
        for trial_idx, row in metadata.iterrows():
            nuisance = (0.5 * float(row["latency"])) + (0.25 * float(row["duration_s"])) + (0.1 * subject_offset)
            data[trial_idx] = (float(row["self_rate"]) * signal_map) + nuisance

        row = {
            "dyad_id": f"dyad-{subject_offset:03d}",
            "subject_id": subject_id,
            "task": "conversation",
            "run": "1",
            "status": "ok",
            "eeg_path": str(tmp_path / f"{subject_id}.fif"),
        }
        epoch_paths = build_epoch_output_paths(project_config, row)
        epoch_paths.epochs_output_path.parent.mkdir(parents=True, exist_ok=True)
        _make_epochs(data, metadata.copy()).save(epoch_paths.epochs_output_path, overwrite=True, verbose="ERROR")
        metadata.to_csv(epoch_paths.metadata_output_path, index=False)
        rows.append(row)

    dataset_index_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_dataframe(pd.DataFrame(rows), dataset_index_path)

    summary = run_decoding_model(
        project_config,
        decoding_config,
        model_name="self_temporal_generalisation",
        dataset_index=str(dataset_index_path),
    )

    output_paths = build_decoding_model_output_paths(project_config, decoding_config, model_name="self_temporal_generalisation")
    assert output_paths.summary_output_path.exists()
    assert output_paths.scores_output_path.exists()
    assert output_paths.fold_scores_output_path.exists()
    assert output_paths.alpha_scores_output_path.exists()
    assert output_paths.best_alphas_output_path.exists()
    assert output_paths.selection_output_path.exists()

    scores = np.load(output_paths.scores_output_path)
    fold_scores = np.load(output_paths.fold_scores_output_path)
    alpha_scores = np.load(output_paths.alpha_scores_output_path)
    best_alphas = np.load(output_paths.best_alphas_output_path)
    selection_df = pd.read_parquet(output_paths.selection_output_path)
    summary_json = json.loads(output_paths.summary_output_path.read_text(encoding="utf-8"))

    assert summary["status"] == "ok"
    assert summary_json["interpretation"].startswith("Temporal generalisation scores quantify")
    assert scores.shape == (2, 2)
    assert fold_scores.shape == (2, 2, 2)
    assert alpha_scores.shape == (2, 2, 2)
    assert best_alphas.shape == (2,)
    assert selection_df["residual_train_target_name"].eq("self_rate").all()
    assert float(np.nanmax(scores)) > 0.5
    assert set(np.unique(best_alphas)).issubset({0.1, 1.0})


def test_run_decoding_model_stats_writes_tfce_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    project_config = _project_config(out_dir)
    decoding_config = _decoding_config()
    rows: list[dict[str, str]] = []
    dataset_index_path = out_dir / "dataset" / "index.parquet"
    signal_map = np.array([[0.9, 0.5, 0.2], [0.1, 0.6, 0.4]], dtype=float)

    for subject_offset, subject_id in enumerate(("sub-001", "sub-002", "sub-003", "sub-004"), start=1):
        metadata = pd.DataFrame(
            {
                "event_id": np.arange(6, dtype=int) + (subject_offset * 100),
                "subject_id": [subject_id] * 6,
                "dyad_id": [f"dyad-{subject_offset:03d}"] * 6,
                "task": ["conversation"] * 6,
                "run": ["1"] * 6,
                "latency": [0.1, 0.4, 0.2, 0.35, 0.15, 0.3],
                "duration_s": [0.4, 0.7, 0.3, 0.8, 0.35, 0.65],
                "time_within_run": [0.0, 0.9, 0.3, 1.0, 0.2, 0.7],
                "self_rate": np.array([-1.2, 0.4, 1.3, -0.5, 0.8, 1.6]) + (0.2 * subject_offset),
                "other_rate": np.array([0.3, -0.7, 0.8, 1.1, -0.2, 0.5]) + (0.1 * subject_offset),
            }
        )
        data = np.zeros((6, 2, 3), dtype=float)
        for trial_idx, row in metadata.iterrows():
            nuisance = (0.4 * float(row["latency"])) + (0.15 * float(row["duration_s"])) + (0.12 * subject_offset)
            data[trial_idx] = (float(row["self_rate"]) * signal_map) + nuisance

        row = {
            "dyad_id": f"dyad-{subject_offset:03d}",
            "subject_id": subject_id,
            "task": "conversation",
            "run": "1",
            "status": "ok",
            "eeg_path": str(tmp_path / f"{subject_id}.fif"),
        }
        epoch_paths = build_epoch_output_paths(project_config, row)
        epoch_paths.epochs_output_path.parent.mkdir(parents=True, exist_ok=True)
        _make_epochs(data, metadata.copy()).save(epoch_paths.epochs_output_path, overwrite=True, verbose="ERROR")
        metadata.to_csv(epoch_paths.metadata_output_path, index=False)
        rows.append(row)

    dataset_index_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_dataframe(pd.DataFrame(rows), dataset_index_path)

    run_decoding_model(
        project_config,
        decoding_config,
        model_name="self_temporal_generalisation",
        dataset_index=str(dataset_index_path),
    )
    metadata = run_decoding_model_stats(
        project_config,
        decoding_config,
        model_name="self_temporal_generalisation",
        dataset_index=str(dataset_index_path),
    )

    output_paths = build_decoding_model_output_paths(project_config, decoding_config, model_name="self_temporal_generalisation")
    assert output_paths.tg_matrix_output_path.exists()
    assert output_paths.tg_tfce_output_path.exists()
    assert output_paths.tg_p_fwer_output_path.exists()
    assert output_paths.tg_null_max_tfce_output_path.exists()
    assert output_paths.tg_stats_metadata_output_path.exists()

    tg_matrix = np.load(output_paths.tg_matrix_output_path)
    tg_tfce = np.load(output_paths.tg_tfce_output_path)
    tg_p_fwer = np.load(output_paths.tg_p_fwer_output_path)
    tg_null = np.load(output_paths.tg_null_max_tfce_output_path)
    stats_json = json.loads(output_paths.tg_stats_metadata_output_path.read_text(encoding="utf-8"))

    assert metadata["status"] == "ok"
    assert tg_matrix.shape == (2, 2)
    assert tg_tfce.shape == (2, 2)
    assert tg_p_fwer.shape == (2, 2)
    assert tg_null.shape == (8,)
    assert np.all((tg_p_fwer >= 0.0) & (tg_p_fwer <= 1.0))
    assert stats_json["permutation_scheme"] == "within_subject"
    assert "no decodable mapping" in stats_json["null_hypothesis"]


def test_cross_condition_permutation_leaves_test_target_fixed() -> None:
    metadata_df = pd.DataFrame(
        {
            "subject_id": ["s1", "s1", "s2", "s2"],
            "run": ["1", "1", "1", "1"],
        }
    )
    from rate.decoding.pipeline import PreparedDecodingData

    prepared = PreparedDecodingData(
        model_name="cross_other_to_self",
        train_target="other_rate",
        test_target="self_rate",
        metadata_df=metadata_df,
        eeg_residuals=np.zeros((4, 1, 1), dtype=float),
        train_target_residuals=np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        test_target_residuals=np.array([10.0, 20.0, 30.0, 40.0], dtype=float),
        fold_indices=[(np.array([0, 2]), np.array([1, 3]))],
        times=np.array([0.0], dtype=float),
        channel_names=["Fz"],
        sfreq=32.0,
        summary={},
    )

    perm_train, perm_test = permute_decoding_targets(
        prepared,
        scheme="within_subject",
        rng=np.random.default_rng(123),
    )

    assert np.array_equal(perm_test, prepared.test_target_residuals)
    assert set(perm_train[:2]) == {1.0, 2.0}
    assert set(perm_train[2:]) == {3.0, 4.0}
