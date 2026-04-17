"""Unit tests for the residualised decoding pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from argparse import Namespace

from rate.cli.commands import figures_decoding
from rate.cli.commands import figures_decoding_results
from rate.decoding.config import load_decoding_config
from rate.decoding.io import build_decoding_model_output_paths
from rate.decoding.pipeline import _residualise_target, resolve_ridge_alphas, run_decoding_model
from rate.decoding.stats import (
    _build_tfce_thresholds,
    _compute_null_max_tfce,
    _resolve_permutation_fixed_alpha,
    build_permutation_grouping,
    compute_tfce_2d,
    permute_decoding_targets,
    run_decoding_model_stats,
)
from rate.epochs.io import build_epoch_output_paths
from rate.util import write_parquet_dataframe
from rate.viz.decoding import _DecodingFigurePayload
from rate.viz.decoding import _infer_diagonal_limits
from rate.viz.decoding import _plot_generalization
from rate.viz.decoding import build_decoding_manifest
from rate.viz.decoding import build_decoding_results_manifest


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
            "permutation_alpha_mode": "retune_per_permutation",
            "fixed_alpha_source": "median_observed_outer_folds",
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
    assert resolved["stats"]["permutation_alpha_mode"] == "retune_per_permutation"
    assert resolved["stats"]["fixed_alpha_source"] == "median_observed_outer_folds"


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


def test_resolve_permutation_fixed_alpha_uses_median_outer_fold_alpha() -> None:
    fixed_alpha = _resolve_permutation_fixed_alpha(
        np.array([0.1, 1.0, 1.0, 10.0], dtype=float),
        permutation_alpha_mode="fixed_alpha",
        fixed_alpha_source="median_observed_outer_folds",
    )

    assert fixed_alpha == 1.0


def test_tfce_2d_boosts_connected_positive_clusters() -> None:
    scores = np.array([[0.0, 0.5, 0.5], [0.0, 0.5, 0.0], [0.2, 0.0, 0.0]], dtype=float)

    tfce = compute_tfce_2d(scores, h_power=2.0, e_power=0.5, h_start=0.0, dh=0.1, tail="positive")

    assert tfce.shape == scores.shape
    assert tfce[0, 1] > tfce[2, 0]
    assert tfce[1, 1] > 0.0


def test_tfce_2d_matches_reference_implementation() -> None:
    scores = np.array(
        [
            [0.0, 0.5, 0.5, 0.1],
            [0.0, 0.5, 0.0, 0.1],
            [0.2, 0.2, 0.0, 0.0],
        ],
        dtype=float,
    )

    expected = _compute_tfce_2d_reference(scores, h_power=2.0, e_power=0.5, h_start=0.0, dh=0.1, tail="positive")
    observed = compute_tfce_2d(
        scores,
        h_power=2.0,
        e_power=0.5,
        h_start=0.0,
        dh=0.1,
        tail="positive",
        thresholds=_build_tfce_thresholds(scores, h_start=0.0, dh=0.1),
    )

    assert np.allclose(observed, expected, atol=1e-12, rtol=1e-12)


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
    assert stats_json["permutation_alpha_mode"] == "retune_per_permutation"
    assert stats_json["fixed_alpha"] is None
    assert "no decodable mapping" in stats_json["null_hypothesis"]


def test_run_decoding_model_stats_fixed_alpha_uses_observed_outer_fold_median(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    project_config = _project_config(out_dir)
    decoding_config = _decoding_config()
    decoding_config["stats"]["permutation_alpha_mode"] = "fixed_alpha"
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
    stats_metadata = run_decoding_model_stats(
        project_config,
        decoding_config,
        model_name="self_temporal_generalisation",
        dataset_index=str(dataset_index_path),
    )

    output_paths = build_decoding_model_output_paths(project_config, decoding_config, model_name="self_temporal_generalisation")
    observed_best_alphas = np.load(output_paths.best_alphas_output_path)
    stats_json = json.loads(output_paths.tg_stats_metadata_output_path.read_text(encoding="utf-8"))

    assert stats_metadata["permutation_alpha_mode"] == "fixed_alpha"
    assert stats_metadata["fixed_alpha_source"] == "median_observed_outer_folds"
    assert stats_metadata["fixed_alpha"] == float(np.median(observed_best_alphas))
    assert stats_json["permutation_inference_notes"]["methodological_status"] == "computational_approximation"
    assert "conditional on the chosen regularisation value" in stats_json["permutation_inference_notes"]["conditioning"]


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


def test_permutation_grouping_matches_reference_permutation() -> None:
    metadata_df = pd.DataFrame(
        {
            "subject_id": ["s1", "s1", "s2", "s2", "s2", "s3"],
            "run": ["1", "1", "1", "2", "2", "1"],
        }
    )
    from rate.decoding.pipeline import PreparedDecodingData

    prepared = PreparedDecodingData(
        model_name="self_temporal_generalisation",
        train_target="self_rate",
        test_target="self_rate",
        metadata_df=metadata_df,
        eeg_residuals=np.zeros((6, 1, 1), dtype=float),
        train_target_residuals=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float),
        test_target_residuals=np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=float),
        fold_indices=[(np.array([0, 2, 4]), np.array([1, 3, 5]))],
        times=np.array([0.0], dtype=float),
        channel_names=["Fz"],
        sfreq=32.0,
        summary={},
    )

    expected_train, expected_test = _permute_decoding_targets_reference(
        prepared,
        scheme="within_subject_within_run",
        rng=np.random.default_rng(123),
    )
    observed_train, observed_test = permute_decoding_targets(
        prepared,
        scheme="within_subject_within_run",
        rng=np.random.default_rng(123),
        grouping=build_permutation_grouping(prepared, scheme="within_subject_within_run"),
    )

    assert np.array_equal(observed_train, expected_train)
    assert np.array_equal(observed_test, expected_test)


def test_prepare_decoding_model_data_cache_roundtrip_matches_fresh(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    project_config = _project_config(out_dir)
    decoding_config = _decoding_config()
    dataset_index_path = _build_decoding_fixture_dataset(tmp_path, project_config)

    from rate.decoding.pipeline import prepare_decoding_model_data

    fresh = prepare_decoding_model_data(
        project_config,
        decoding_config,
        model_name="self_temporal_generalisation",
        dataset_index=str(dataset_index_path),
        use_prepared_data_cache=False,
    )
    prepare_decoding_model_data(
        project_config,
        decoding_config,
        model_name="self_temporal_generalisation",
        dataset_index=str(dataset_index_path),
        use_prepared_data_cache=True,
    )
    cached = prepare_decoding_model_data(
        project_config,
        decoding_config,
        model_name="self_temporal_generalisation",
        dataset_index=str(dataset_index_path),
        use_prepared_data_cache=True,
    )

    assert fresh.metadata_df.equals(cached.metadata_df)
    assert np.array_equal(np.asarray(fresh.eeg_residuals), np.asarray(cached.eeg_residuals))
    assert np.array_equal(fresh.train_target_residuals, np.asarray(cached.train_target_residuals))
    assert np.array_equal(fresh.test_target_residuals, np.asarray(cached.test_target_residuals))
    assert np.array_equal(fresh.times, np.asarray(cached.times))
    assert fresh.channel_names == cached.channel_names
    for (fresh_train, fresh_test), (cached_train, cached_test) in zip(fresh.fold_indices, cached.fold_indices, strict=True):
        assert np.array_equal(fresh_train, cached_train)
        assert np.array_equal(fresh_test, cached_test)


def test_observed_decoding_matches_fresh_and_cached_prepared_inputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    project_config = _project_config(out_dir)
    decoding_config = _decoding_config()
    dataset_index_path = _build_decoding_fixture_dataset(tmp_path, project_config)

    from rate.decoding.pipeline import _run_temporal_generalisation, prepare_decoding_model_data

    fresh = prepare_decoding_model_data(
        project_config,
        decoding_config,
        model_name="self_temporal_generalisation",
        dataset_index=str(dataset_index_path),
        use_prepared_data_cache=False,
    )
    cached = prepare_decoding_model_data(
        project_config,
        decoding_config,
        model_name="self_temporal_generalisation",
        dataset_index=str(dataset_index_path),
        use_prepared_data_cache=True,
    )

    fresh_result = _run_temporal_generalisation(
        np.asarray(fresh.eeg_residuals),
        fresh.train_target_residuals,
        fresh.test_target_residuals,
        estimator_config=decoding_config["estimator"],
        cv_config=decoding_config["cv"],
        nested_cv_config=decoding_config["nested_cv"],
        scoring_config=decoding_config["scoring"],
        fold_indices=fresh.fold_indices,
        groups=fresh.metadata_df["subject_id"].to_numpy(),
        model_name=fresh.model_name,
        show_progress=False,
        save_alpha_scores=True,
    )
    cached_result = _run_temporal_generalisation(
        np.asarray(cached.eeg_residuals),
        np.asarray(cached.train_target_residuals),
        np.asarray(cached.test_target_residuals),
        estimator_config=decoding_config["estimator"],
        cv_config=decoding_config["cv"],
        nested_cv_config=decoding_config["nested_cv"],
        scoring_config=decoding_config["scoring"],
        fold_indices=cached.fold_indices,
        groups=cached.metadata_df["subject_id"].to_numpy(),
        model_name=cached.model_name,
        show_progress=False,
        save_alpha_scores=True,
    )

    assert np.allclose(fresh_result.mean_scores, cached_result.mean_scores, atol=1e-12, rtol=1e-12)
    assert np.allclose(fresh_result.fold_scores, cached_result.fold_scores, atol=1e-12, rtol=1e-12)
    assert np.allclose(fresh_result.alpha_scores, cached_result.alpha_scores, atol=1e-12, rtol=1e-12)
    assert np.array_equal(fresh_result.best_alphas, cached_result.best_alphas)


def test_stats_null_tfce_matches_reference_implementation(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    project_config = _project_config(out_dir)
    decoding_config = _decoding_config()
    decoding_config["stats"]["n_permutations"] = 4
    dataset_index_path = _build_decoding_fixture_dataset(tmp_path, project_config)

    from rate.decoding.pipeline import prepare_decoding_model_data

    prepared = prepare_decoding_model_data(
        project_config,
        decoding_config,
        model_name="self_temporal_generalisation",
        dataset_index=str(dataset_index_path),
        use_prepared_data_cache=False,
    )

    expected = _compute_null_max_tfce_reference(prepared, decoding_config=decoding_config, stats_config=decoding_config["stats"])
    observed = _compute_null_max_tfce(
        prepared,
        decoding_config=decoding_config,
        stats_config=decoding_config["stats"],
        fixed_alpha=None,
        show_progress=False,
    )

    assert np.allclose(observed, expected, atol=1e-12, rtol=1e-12)


def test_build_decoding_manifest_writes_combined_plot(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    decoding_config = _decoding_config()
    manifest_path = out_dir / "figures" / "decoding" / "figure_manifest.json"
    times = np.asarray([-0.1, 0.0, 0.1], dtype=float)
    p_values = np.asarray(
        [
            [1.0, 0.04, 1.0],
            [1.0, 0.01, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    for model_index, model_cfg in enumerate(decoding_config["models"], start=1):
        output_paths = build_decoding_model_output_paths(
            {"paths": {"out_dir": str(out_dir)}},
            decoding_config,
            model_name=str(model_cfg["name"]),
        )
        output_paths.scores_output_path.parent.mkdir(parents=True, exist_ok=True)
        alpha_scores = np.asarray(
            [
                [
                    [0.02 * model_index, 0.01, 0.0],
                    [0.01, 0.03 * model_index, 0.01],
                    [0.0, 0.01, 0.02 * model_index],
                ],
                [
                    [0.08 * model_index, 0.02, 0.01],
                    [0.02, 0.12 * model_index, 0.03],
                    [0.01, 0.03, 0.09 * model_index],
                ],
            ],
            dtype=float,
        )
        np.save(output_paths.alpha_scores_output_path, alpha_scores)
        np.save(output_paths.scores_output_path, alpha_scores[1])
        np.save(output_paths.fold_scores_output_path, np.stack([alpha_scores[1], alpha_scores[1] * 0.9], axis=0))
        np.save(output_paths.best_alphas_output_path, np.asarray([1.0, 1.0], dtype=float))
        np.save(output_paths.times_output_path, times)
        np.save(output_paths.tg_p_fwer_output_path, p_values)

    manifest = build_decoding_manifest(
        out_dir=out_dir,
        decoding_config=decoding_config,
        manifest_path=manifest_path,
        figure_config={"diag_ymin": -0.05, "diag_ymax": 0.6},
        formats=("png",),
        dpi=72,
    )

    assert manifest["status"] == "ok"
    assert manifest["plot_count"] == 1
    assert manifest_path.exists()

    plot = manifest["plots"][0]
    assert plot["plot_name"] == "temporal_generalisation_summary"
    assert plot["diag_ylim"] == [-0.05, 0.6]
    assert len(plot["files"]) == 1
    assert Path(plot["files"][0]).exists()
    assert [model["model_name"] for model in plot["models"]] == [
        "self_temporal_generalisation",
        "cross_other_to_self",
    ]
    assert all(model["selected_alpha"] == 1.0 for model in plot["models"])
    assert all(model["alpha_selection_source"] == "alpha_scores_mean_diagonal" for model in plot["models"])


def test_infer_diagonal_limits_hugs_global_extrema() -> None:
    payloads = [
        _DecodingFigurePayload(
            model_name="self_temporal_generalisation",
            title="Self rate",
            color="#000000",
            times=np.asarray([-0.1, 0.0, 0.1], dtype=float),
            score_matrix=np.asarray(
                [
                    [0.02, 0.00, 0.00],
                    [0.00, 0.18, 0.00],
                    [0.00, 0.00, 0.26],
                ],
                dtype=float,
            ),
            significance_mask=None,
            selected_alpha=None,
            alpha_selection_source="mean_scores",
        ),
        _DecodingFigurePayload(
            model_name="cross_other_to_self",
            title="Other to self",
            color="#000000",
            times=np.asarray([-0.1, 0.0, 0.1], dtype=float),
            score_matrix=np.asarray(
                [
                    [-0.03, 0.00, 0.00],
                    [0.00, 0.14, 0.00],
                    [0.00, 0.00, 0.21],
                ],
                dtype=float,
            ),
            significance_mask=None,
            selected_alpha=None,
            alpha_selection_source="mean_scores",
        ),
    ]

    ymin, ymax = _infer_diagonal_limits(payloads, pad_fraction=0.02, min_half_range=0.015)

    assert np.isclose(ymin, -0.0358)
    assert np.isclose(ymax, 0.2658)


def test_plot_generalization_uses_square_matrix_axes() -> None:
    fig, ax = plt.subplots(figsize=(4, 3))

    _plot_generalization(
        ax=ax,
        times=np.asarray([-0.1, 0.0, 0.1], dtype=float),
        score_matrix=np.asarray(
            [
                [0.02, 0.01, 0.00],
                [0.01, 0.12, 0.03],
                [0.00, 0.03, 0.08],
            ],
            dtype=float,
        ),
        significance_mask=None,
        show_y_axis=True,
        abs_limit=0.15,
    )

    assert ax.get_aspect() == 1.0
    assert float(ax.get_box_aspect()) == 1.0

    plt.close(fig)


def test_figures_decoding_cli_writes_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    decoding_cfg_path = tmp_path / "decoding.yaml"
    viz_config_path = tmp_path / "viz.yaml"
    _write_yaml(tmp_path / "decoding.yaml", {"decoding": _decoding_config()})
    viz_config_path.write_text(
        yaml.safe_dump(
            {
                "viz": {
                    "formats": ["png"],
                    "dpi": 72,
                    "figures": {
                        "decoding": {
                            "diag_ymin": -0.05,
                            "diag_ymax": 0.6,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    for model_index, model_cfg in enumerate(_decoding_config()["models"], start=1):
        output_paths = build_decoding_model_output_paths(
            {"paths": {"out_dir": str(out_dir)}},
            _decoding_config(),
            model_name=str(model_cfg["name"]),
        )
        output_paths.scores_output_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.asarray(
            [
                [0.05 * model_index, 0.01],
                [0.02, 0.08 * model_index],
            ],
            dtype=float,
        )
        np.save(output_paths.scores_output_path, matrix)
        np.save(output_paths.fold_scores_output_path, np.stack([matrix, matrix * 0.95], axis=0))
        np.save(output_paths.alpha_scores_output_path, np.stack([matrix * 0.6, matrix], axis=0))
        np.save(output_paths.best_alphas_output_path, np.asarray([1.0, 1.0], dtype=float))
        np.save(output_paths.times_output_path, np.asarray([-0.05, 0.05], dtype=float))
        np.save(output_paths.tg_p_fwer_output_path, np.asarray([[1.0, 0.04], [1.0, 1.0]], dtype=float))

    args = Namespace(
        config="config/config.yaml",
        schema="config/schema.json",
        viz_config=str(viz_config_path),
        decoding_config=str(decoding_cfg_path),
        output=None,
    )
    exit_code = figures_decoding.run(args, config={"paths": {"out_dir": str(out_dir)}})

    assert exit_code == 0
    manifest_path = out_dir / "figures" / "decoding" / "figure_manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["plot_count"] == 1
    assert manifest["plots"][0]["plot_name"] == "temporal_generalisation_summary"


def test_build_decoding_results_manifest_writes_per_model_plots(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    decoding_config = _decoding_config()
    manifest_path = out_dir / "figures" / "decoding_results" / "figure_manifest.json"

    for model_index, model_cfg in enumerate(decoding_config["models"], start=1):
        output_paths = build_decoding_model_output_paths(
            {"paths": {"out_dir": str(out_dir)}},
            decoding_config,
            model_name=str(model_cfg["name"]),
        )
        output_paths.scores_output_path.parent.mkdir(parents=True, exist_ok=True)
        observed = np.asarray(
            [
                [0.05 * model_index, 0.01],
                [0.02, 0.08 * model_index],
            ],
            dtype=float,
        )
        np.save(output_paths.scores_output_path, observed)
        np.save(output_paths.fold_scores_output_path, np.stack([observed, observed * 0.95], axis=0))
        np.save(output_paths.alpha_scores_output_path, np.stack([observed * 0.6, observed], axis=0))
        np.save(output_paths.best_alphas_output_path, np.asarray([1.0, 1.0], dtype=float))
        np.save(output_paths.times_output_path, np.asarray([-0.05, 0.05], dtype=float))
        np.save(output_paths.tg_matrix_output_path, observed)
        np.save(output_paths.tg_tfce_output_path, np.asarray([[0.0, 1.5], [0.5, 2.0]], dtype=float))
        np.save(output_paths.tg_p_fwer_output_path, np.asarray([[1.0, 0.04], [1.0, 1.0]], dtype=float))

    manifest = build_decoding_results_manifest(
        out_dir=out_dir,
        decoding_config=decoding_config,
        manifest_path=manifest_path,
        figure_config={"p_threshold": 0.05},
        formats=("png",),
        dpi=72,
    )

    assert manifest["status"] == "ok"
    assert manifest["plot_count"] == 2
    assert manifest_path.exists()
    assert {plot["model_name"] for plot in manifest["plots"]} == {
        "self_temporal_generalisation",
        "cross_other_to_self",
    }
    assert all(plot["plot_name"] == "temporal_generalisation_results" for plot in manifest["plots"])
    assert all(len(plot["files"]) == 1 for plot in manifest["plots"])
    assert all(Path(plot["files"][0]).exists() for plot in manifest["plots"])


def test_figures_decoding_results_cli_writes_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    decoding_cfg_path = tmp_path / "decoding.yaml"
    viz_config_path = tmp_path / "viz.yaml"
    _write_yaml(decoding_cfg_path, {"decoding": _decoding_config()})
    viz_config_path.write_text(
        yaml.safe_dump(
            {
                "viz": {
                    "formats": ["png"],
                    "dpi": 72,
                    "figures": {
                        "decoding_results": {
                            "p_threshold": 0.05,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    for model_index, model_cfg in enumerate(_decoding_config()["models"], start=1):
        output_paths = build_decoding_model_output_paths(
            {"paths": {"out_dir": str(out_dir)}},
            _decoding_config(),
            model_name=str(model_cfg["name"]),
        )
        output_paths.scores_output_path.parent.mkdir(parents=True, exist_ok=True)
        observed = np.asarray([[0.05 * model_index, 0.01], [0.02, 0.08 * model_index]], dtype=float)
        np.save(output_paths.scores_output_path, observed)
        np.save(output_paths.fold_scores_output_path, np.stack([observed, observed * 0.95], axis=0))
        np.save(output_paths.alpha_scores_output_path, np.stack([observed * 0.6, observed], axis=0))
        np.save(output_paths.best_alphas_output_path, np.asarray([1.0, 1.0], dtype=float))
        np.save(output_paths.times_output_path, np.asarray([-0.05, 0.05], dtype=float))
        np.save(output_paths.tg_matrix_output_path, observed)
        np.save(output_paths.tg_tfce_output_path, np.asarray([[0.0, 1.5], [0.5, 2.0]], dtype=float))
        np.save(output_paths.tg_p_fwer_output_path, np.asarray([[1.0, 0.04], [1.0, 1.0]], dtype=float))

    args = Namespace(
        config="config/config.yaml",
        schema="config/schema.json",
        viz_config=str(viz_config_path),
        decoding_config=str(decoding_cfg_path),
        output=None,
    )
    exit_code = figures_decoding_results.run(args, config={"paths": {"out_dir": str(out_dir)}})

    assert exit_code == 0
    manifest_path = out_dir / "figures" / "decoding_results" / "figure_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["plot_count"] == 2
    assert {plot["plot_name"] for plot in manifest["plots"]} == {"temporal_generalisation_results"}


def _build_decoding_fixture_dataset(tmp_path: Path, project_config: dict) -> Path:
    rows: list[dict[str, str]] = []
    dataset_index_path = project_config["paths"]["out_dir"]
    dataset_index = Path(dataset_index_path) / "dataset" / "index.parquet"
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

    dataset_index.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_dataframe(pd.DataFrame(rows), dataset_index)
    return dataset_index


def _permute_decoding_targets_reference(prepared, *, scheme: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    train_values = np.asarray(prepared.train_target_residuals, dtype=float)
    test_values = np.asarray(prepared.test_target_residuals, dtype=float)
    permuted_train = train_values.copy()
    permuted_test = test_values.copy()

    if scheme == "within_subject":
        group_keys = prepared.metadata_df["subject_id"].astype(str)
    elif scheme == "within_subject_within_run":
        if "run" not in prepared.metadata_df.columns:
            raise ValueError("`within_subject_within_run` permutation requires a `run` column in decoding metadata.")
        group_keys = prepared.metadata_df["subject_id"].astype(str) + "::" + prepared.metadata_df["run"].astype(str)
    else:
        raise ValueError(f"Unsupported decoding permutation scheme `{scheme}`.")

    unique_keys = group_keys.drop_duplicates().tolist()
    for group_key in unique_keys:
        group_indices = np.flatnonzero(group_keys.to_numpy(dtype=str) == str(group_key))
        if group_indices.size <= 1:
            continue
        shuffled_indices = rng.permutation(group_indices)
        permuted_train[group_indices] = train_values[shuffled_indices]
        if prepared.train_target == prepared.test_target:
            permuted_test[group_indices] = test_values[shuffled_indices]
        else:
            permuted_test[group_indices] = test_values[group_indices]
    return permuted_train, permuted_test


def _compute_tfce_2d_reference(
    score_matrix: np.ndarray,
    *,
    h_power: float,
    e_power: float,
    h_start: float,
    dh: float,
    tail: str = "positive",
) -> np.ndarray:
    if tail != "positive":
        raise ValueError("Only positive-tail TFCE is implemented for decoding stats.")
    scores = np.maximum(np.asarray(score_matrix, dtype=float), 0.0)
    tfce = np.zeros_like(scores, dtype=float)
    max_height = float(np.nanmax(scores))
    if np.isclose(max_height, 0.0):
        return tfce

    threshold = max(float(h_start), 0.0)
    if np.isclose(threshold, 0.0):
        threshold = float(dh)
    while threshold <= (max_height + (0.5 * dh)):
        supra_threshold = scores >= threshold
        for component in _connected_components_2d_reference(supra_threshold):
            extent = float(len(component))
            increment = (threshold ** h_power) * (extent ** e_power) * dh
            for row_idx, col_idx in component:
                tfce[row_idx, col_idx] += increment
        threshold += dh
    return tfce


def _connected_components_2d_reference(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    n_rows, n_cols = mask.shape
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            if not bool(mask[row_idx, col_idx]) or bool(visited[row_idx, col_idx]):
                continue
            stack = [(row_idx, col_idx)]
            visited[row_idx, col_idx] = True
            component: list[tuple[int, int]] = []
            while stack:
                current_row, current_col = stack.pop()
                component.append((current_row, current_col))
                for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if next_row < 0 or next_row >= n_rows or next_col < 0 or next_col >= n_cols:
                        continue
                    if not bool(mask[next_row, next_col]) or bool(visited[next_row, next_col]):
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
            components.append(component)
    return components


def _compute_null_max_tfce_reference(prepared, *, decoding_config: dict, stats_config: dict) -> np.ndarray:
    from rate.decoding.pipeline import _run_temporal_generalisation

    n_permutations = int(stats_config.get("n_permutations", 1000))
    rng = np.random.default_rng(int(stats_config.get("random_state", decoding_config.get("random_state", 42))))
    permutation_scheme = str(stats_config.get("permutation_scheme", "within_subject"))
    tfce_cfg = dict(stats_config.get("tfce") or {})
    null_max = np.zeros(n_permutations, dtype=float)

    for permutation_idx in range(n_permutations):
        permuted_train, permuted_test = _permute_decoding_targets_reference(
            prepared,
            scheme=permutation_scheme,
            rng=rng,
        )
        permuted_scores = _run_temporal_generalisation(
            np.asarray(prepared.eeg_residuals),
            permuted_train,
            permuted_test,
            estimator_config=decoding_config["estimator"],
            cv_config=decoding_config["cv"],
            nested_cv_config=decoding_config.get("nested_cv") or {},
            scoring_config=decoding_config["scoring"],
            fold_indices=prepared.fold_indices,
            groups=prepared.metadata_df[str(decoding_config["cv"]["group_column"])].to_numpy(),
            model_name=f"{prepared.model_name} perm {permutation_idx + 1}",
            show_progress=False,
            save_alpha_scores=False,
        ).mean_scores
        permuted_tfce = _compute_tfce_2d_reference(
            permuted_scores,
            h_power=float(tfce_cfg.get("h_power", 2.0)),
            e_power=float(tfce_cfg.get("e_power", 0.5)),
            h_start=float(tfce_cfg.get("h_start", 0.0)),
            dh=float(tfce_cfg.get("dh", 0.1)),
            tail=str(tfce_cfg.get("tail", "positive")),
        )
        null_max[permutation_idx] = float(np.max(permuted_tfce))
    return null_max


def test_decoding_figure_workflow_rules_are_present() -> None:
    figures_rules = Path("workflow/rules/figures.smk").read_text(encoding="utf-8")
    targets_rules = Path("workflow/rules/targets.smk").read_text(encoding="utf-8")

    assert "rule figures_decoding" in figures_rules
    assert "DECODING_FIGURES_OUTPUT" in figures_rules
    assert "rule figures_decoding_results" in figures_rules
    assert "DECODING_RESULTS_FIGURES_OUTPUT" in figures_rules
    assert "rule decoding_figures_all" in targets_rules
    assert "DECODING_FIGURES_OUTPUT" in targets_rules
    assert "rule decoding_results_figures_all" in targets_rules
    assert "DECODING_RESULTS_FIGURES_OUTPUT" in targets_rules
