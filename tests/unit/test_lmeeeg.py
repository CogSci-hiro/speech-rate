"""Unit tests for the subject-level lmeEEG approximation stage."""

from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

from rate.cli.commands import lmeeeg_group_stats
from rate.cli.commands import lmeeeg_similarity
from rate.cli.commands import lmeeeg_subject
from rate.cli.commands import lmeeeg_variance_partition_fit
from rate.cli.commands import lmeeeg_variance_partition_group_stats
from rate.cli.commands import figures_lmeeeg_variance_partition
from rate.cli.commands import figures_lmeeeg_main_results
from rate.cli.commands import figures_lmeeeg
from rate.cli.commands import figures_lmeeeg_inference
from rate.cli.commands import figures_similarity
from rate.epochs.io import build_epoch_output_paths
from rate.lmeeeg.io import build_lmeeeg_model_output_paths, build_lmeeeg_output_paths, merge_trial_regressors_into_metadata
from rate.lmeeeg.pooling import compute_lmeeeg_pooled_beta_map, load_pooled_lmeeeg_inputs
from rate.lmeeeg.similarity import (
    SimilarityPermutationScheme,
    _build_similarity_permutation_grouping,
    _compute_tfce_1d,
    _generate_similarity_permutation_indices,
    _permute_similarity_predictor,
    _resolve_similarity_refit_estimator_strategy,
    build_similarity_output_paths,
    resolve_similarity_settings,
)
from rate.lmeeeg.mass_model import fit_lmeeeg_mass_model
from rate.lmeeeg.trial_data import build_lmeeeg_trial_table
from rate.lmeeeg.variance_partition import (
    build_variance_partition_fit_output_paths,
    build_variance_partition_group_output_paths,
    compute_variance_partition_components,
)
from rate.util import write_parquet_dataframe
from rate.viz import lmeeeg as lmeeeg_viz
from rate.viz import lmeeeg_results as lmeeeg_results_viz
from rate.viz import lmeeeg_similarity as lmeeeg_similarity_viz


def _make_epochs(data: np.ndarray, metadata: pd.DataFrame) -> mne.EpochsArray:
    info = mne.create_info(["Fz", "Cz"], sfreq=100.0, ch_types=["eeg", "eeg"])
    events = np.column_stack([np.arange(len(data), dtype=int), np.zeros(len(data), dtype=int), np.ones(len(data), dtype=int)])
    return mne.EpochsArray(data, info, events=events, tmin=-0.1, metadata=metadata, verbose="ERROR")


def _lmeeeg_config(out_dir: Path) -> dict:
    return {
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
        "preprocessing": {"montage": {"name": "standard_1020"}},
        "lmeeeg": {
            "backend": "lmeeeg",
            "random_effects": {"group": "item_id", "structure": "random_intercept"},
            "models": {
                "speech_rate_item_intercept": {
                    "formula": "signal ~ speech_rate",
                    "test_predictors": ["speech_rate"],
                    "random_effect_group": "item_id",
                    "random_effect_structure": "random_intercept",
                },
                "other_rate_effect": {
                    "formula": "~ other_rate + latency + duration_s + time_within_run + run",
                    "test_predictors": ["other_rate"],
                    "random_effect_group": "subject_id",
                    "random_effect_structure": "random_intercept",
                },
                "self_rate_effect": {
                    "formula": "~ self_rate + latency + duration_s + time_within_run + run",
                    "test_predictors": ["self_rate"],
                    "random_effect_group": "subject_id",
                    "random_effect_structure": "random_intercept",
                },
                "both_effect": {
                    "formula": "~ self_rate + other_rate + latency + duration_s + time_within_run + run",
                    "eligibility": {
                        "not_null": ["self_rate", "other_rate"],
                        "min_values": {
                            "self_n_syllables": 3,
                            "other_n_syllables": 3,
                            "self_ipu_duration": 0.10,
                            "other_ipu_duration": 0.10,
                        },
                    },
                    "test_predictors": ["self_rate", "other_rate"],
                    "standardize": ["self_rate", "other_rate", "latency", "duration_s", "time_within_run"],
                    "random_effect_group": "subject_id",
                    "random_effect_structure": "random_intercept",
                }
            },
        },
        "lmeeeg_variance_partition": {
            "enabled": True,
            "source_section": "lmeeeg",
            "source_model": "both_effect",
            "output_dirname": "lmeeeg_variance_partition",
            "predictors": {"self": "self_rate", "other": "other_rate"},
        },
        "similarity": {
            "enabled": True,
            "analysis_name": "other_vs_self",
            "input": {
                "domain": "lmeeeg",
                "result_kind": "group_stats",
                "predictor_a": "other_rate",
                "predictor_b": "self_rate",
                "value_field": "beta",
                "use_unthresholded": True,
            },
            "metrics": {
                "whole_map": ["pearson", "cosine"],
                "time_resolved_topography": {"enabled": True, "method": "pearson"},
            },
            "inferential_test": {
                "enabled": True,
                "n_permutations": 64,
                "random_seed": 7,
                "null_method": "sign_flip",
                "tail": "two-sided",
            },
            "outputs": {
                "save_null_distribution": True,
                "save_time_resolved_curve": True,
                "save_summary_table": True,
                "save_difference_map": True,
            },
        },
    }


def _write_variance_partition_epochs(tmp_path: Path, config: dict, rows: list[dict[str, str]], metadata_by_subject: dict[str, pd.DataFrame], data_by_subject: dict[str, np.ndarray]) -> Path:
    dataset_index_path = tmp_path / "out" / "dataset" / "index.parquet"
    dataset_index_path.parent.mkdir(parents=True, exist_ok=True)
    written_rows: list[dict[str, str]] = []
    for row in rows:
        metadata = metadata_by_subject[str(row["subject_id"])]
        data = data_by_subject[str(row["subject_id"])]
        epochs = _make_epochs(data, metadata.copy())
        epoch_paths = build_epoch_output_paths(config, row)
        epoch_paths.epochs_output_path.parent.mkdir(parents=True, exist_ok=True)
        epochs.save(epoch_paths.epochs_output_path, overwrite=True, verbose="ERROR")
        metadata.to_csv(epoch_paths.metadata_output_path, index=False)
        written_rows.append(row)
    write_parquet_dataframe(pd.DataFrame(written_rows), dataset_index_path)
    return dataset_index_path


def _write_similarity_epochs(
    tmp_path: Path,
    config: dict,
    rows: list[dict[str, str]],
    *,
    metadata_by_subject: dict[str, pd.DataFrame],
    data_by_subject: dict[str, np.ndarray],
) -> Path:
    dataset_index_path = tmp_path / "out" / "dataset" / "index.parquet"
    dataset_index_path.parent.mkdir(parents=True, exist_ok=True)
    written_rows: list[dict[str, str]] = []
    for row in rows:
        metadata = metadata_by_subject[str(row["subject_id"])]
        data = data_by_subject[str(row["subject_id"])]
        epochs = _make_epochs(data, metadata.copy())
        epoch_paths = build_epoch_output_paths(config, row)
        epoch_paths.epochs_output_path.parent.mkdir(parents=True, exist_ok=True)
        epochs.save(epoch_paths.epochs_output_path, overwrite=True, verbose="ERROR")
        metadata.to_csv(epoch_paths.metadata_output_path, index=False)
        written_rows.append(row)
    write_parquet_dataframe(pd.DataFrame(written_rows), dataset_index_path)
    return dataset_index_path


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
    assert summary["predictor_of_interest"] == "speech_rate"
    assert summary["n_lmm_failed"] == 0
    assert betas.shape == (2, 2, 2)


def test_merge_trial_regressors_into_metadata_preserves_row_order_and_values() -> None:
    metadata = pd.DataFrame(
        {
            "event_id": ["ev-002", "ev-001", "ev-003"],
            "subject_id": ["sub-001", "sub-001", "sub-002"],
            "run": ["1", "1", "2"],
            "task": ["conversation", "conversation", "conversation"],
        }
    )
    trial_regressors = pd.DataFrame(
        {
            "event_id": ["ev-001", "ev-002", "ev-003"],
            "subject_id": ["sub-001", "sub-001", "sub-002"],
            "run": ["1", "1", "2"],
            "task": ["conversation", "conversation", "conversation"],
            "predicted_self_rate": [4.1, 4.2, 5.0],
            "residual_signed": [-0.1, 0.2, 0.0],
            "residual_abs": [0.1, 0.2, 0.0],
        }
    )

    merged_df, merge_info = merge_trial_regressors_into_metadata(
        metadata,
        trial_regressors,
        source_path=Path("/tmp/trialwise_residuals.csv"),
    )

    assert merged_df["event_id"].tolist() == ["ev-002", "ev-001", "ev-003"]
    assert np.allclose(merged_df["predicted_self_rate"].to_numpy(dtype=float), [4.2, 4.1, 5.0])
    assert np.allclose(merged_df["residual_signed"].to_numpy(dtype=float), [0.2, -0.1, 0.0])
    assert merge_info["matched_rows"] == 3
    assert merge_info["unmatched_rows"] == 0


def test_compute_variance_partition_components_from_r2_values() -> None:
    components = compute_variance_partition_components(
        r2_null=0.10,
        r2_self=0.25,
        r2_other=0.30,
        r2_full=0.40,
    )

    assert components["total"] == pytest.approx(0.30)
    assert components["unique_self"] == pytest.approx(0.10)
    assert components["unique_other"] == pytest.approx(0.15)
    assert components["shared"] == pytest.approx(0.05)


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
    assert Path(effect_summary["p_values_csv"]).exists()


def test_lmeeeg_variance_partition_fit_cli_writes_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = _lmeeeg_config(out_dir)
    rows = [
        {
            "dyad_id": "dyad-001",
            "subject_id": "sub-001",
            "task": "conversation",
            "run": "1",
            "status": "ok",
            "eeg_path": str(tmp_path / "placeholder_raw_1.fif"),
        },
        {
            "dyad_id": "dyad-001",
            "subject_id": "sub-002",
            "task": "conversation",
            "run": "1",
            "status": "ok",
            "eeg_path": str(tmp_path / "placeholder_raw_2.fif"),
        },
    ]
    metadata_by_subject = {
        "sub-001": pd.DataFrame(
            {
                "self_rate": [0.0, 0.6, 1.2, 1.8],
                "other_rate": [0.3, 0.9, 1.1, 1.7],
                "latency": [0.10, 0.12, 0.14, 0.16],
                "duration_s": [0.60, 0.62, 0.64, 0.66],
                "time_within_run": [0.1, 0.2, 0.3, 0.4],
                "run": ["1", "1", "1", "1"],
                "subject_id": ["sub-001"] * 4,
                "self_n_syllables": [4, 4, 4, 4],
                "other_n_syllables": [4, 4, 4, 4],
                "self_ipu_duration": [0.20, 0.22, 0.24, 0.26],
                "other_ipu_duration": [0.21, 0.23, 0.25, 0.27],
            }
        ),
        "sub-002": pd.DataFrame(
            {
                "self_rate": [0.2, 0.8, 1.4, 2.0],
                "other_rate": [0.4, 1.0, 1.3, 1.9],
                "latency": [0.11, 0.13, 0.15, 0.17],
                "duration_s": [0.61, 0.63, 0.65, 0.67],
                "time_within_run": [0.1, 0.2, 0.3, 0.4],
                "run": ["1", "1", "1", "1"],
                "subject_id": ["sub-002"] * 4,
                "self_n_syllables": [4, 4, 4, 4],
                "other_n_syllables": [4, 4, 4, 4],
                "self_ipu_duration": [0.21, 0.23, 0.25, 0.27],
                "other_ipu_duration": [0.22, 0.24, 0.26, 0.28],
            }
        ),
    }
    data_by_subject: dict[str, np.ndarray] = {}
    slope_self = np.array([[1.0, 0.4], [0.2, 0.8]])
    slope_other = np.array([[0.5, 0.7], [0.6, 0.3]])
    intercept_map = np.array([[2.0, 1.5], [1.0, 0.5]])
    subject_offsets = {"sub-001": 1.0, "sub-002": -0.5}
    for subject_id, metadata in metadata_by_subject.items():
        data = np.zeros((len(metadata), 2, 2), dtype=float)
        for index, row_values in metadata.iterrows():
            data[index] = (
                intercept_map
                + (float(row_values["self_rate"]) * slope_self)
                + (float(row_values["other_rate"]) * slope_other)
                + subject_offsets[subject_id]
            )
        data_by_subject[subject_id] = data

    dataset_index_path = _write_variance_partition_epochs(tmp_path, config, rows, metadata_by_subject, data_by_subject)

    args = Namespace(
        config="config/config.yaml",
        schema="config/schema.json",
        dataset_index=str(dataset_index_path),
        subject_id=None,
        task=None,
        run=None,
        trial_regressors_csv=None,
    )
    exit_code = lmeeeg_variance_partition_fit.run(args, config=config)
    assert exit_code == 0

    output_paths = build_variance_partition_fit_output_paths(config)
    assert output_paths.table_output_path.exists()
    assert output_paths.summary_output_path.exists()
    assert output_paths.channel_names_output_path.exists()
    assert output_paths.times_output_path.exists()

    result_df = pd.read_parquet(output_paths.table_output_path)
    summary = json.loads(output_paths.summary_output_path.read_text(encoding="utf-8"))

    assert {"channel", "time", "r2_null", "r2_self", "r2_other", "r2_full", "total", "unique_self", "unique_other", "shared"} <= set(result_df.columns)
    assert len(result_df) == 4
    assert summary["status"] == "ok"
    assert summary["fit_scope"] == "pooled"
    assert summary["group_column"] == "subject_id"


def test_lmeeeg_variance_partition_group_stats_cli_writes_descriptive_maps(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = _lmeeeg_config(out_dir)
    fit_paths = build_variance_partition_fit_output_paths(config)
    fit_paths.root_dir.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(
        {
            "channel": ["Fz", "Fz", "Cz", "Cz"],
            "time": [-0.1, -0.09, -0.1, -0.09],
            "channel_index": [0, 0, 1, 1],
            "time_index": [0, 1, 0, 1],
            "r2_null": [0.1, 0.1, 0.1, 0.1],
            "r2_self": [0.2, 0.25, 0.3, 0.35],
            "r2_other": [0.18, 0.22, 0.28, 0.31],
            "r2_full": [0.26, 0.30, 0.36, 0.40],
            "total": [0.16, 0.20, 0.26, 0.30],
            "unique_self": [0.08, 0.08, 0.08, 0.09],
            "unique_other": [0.06, 0.05, 0.06, 0.05],
            "shared": [0.02, 0.07, 0.12, 0.16],
            "n_obs": [8, 8, 8, 8],
            "n_subjects": [2, 2, 2, 2],
            "converged_null": [True, True, True, True],
            "converged_self": [True, True, True, True],
            "converged_other": [True, True, True, True],
            "converged_full": [True, True, True, True],
            "singular_null": [False, False, False, False],
            "singular_self": [False, False, False, False],
            "singular_other": [False, False, False, False],
            "singular_full": [False, False, False, False],
        }
    )
    table.to_parquet(fit_paths.table_output_path, index=False)
    fit_paths.channel_names_output_path.write_text(json.dumps(["Fz", "Cz"]) + "\n", encoding="utf-8")
    np.save(fit_paths.times_output_path, np.array([-0.1, -0.09], dtype=float))
    fit_paths.summary_output_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "fit_scope": "pooled",
                "source_section": "lmeeeg",
                "source_model": "both_effect",
                "self_predictor": "self_rate",
                "other_predictor": "other_rate",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    args = Namespace(config="config/config.yaml", schema="config/schema.json", output=None)
    exit_code = lmeeeg_variance_partition_group_stats.run(args, config=config)
    assert exit_code == 0

    group_paths = build_variance_partition_group_output_paths(config)
    summary_path = out_dir / "stats" / "lmeeeg_variance_partition_group_stats.json"
    assert summary_path.exists()
    assert group_paths.table_output_path.exists()
    assert group_paths.total_map_output_path.exists()
    assert group_paths.unique_self_map_output_path.exists()
    assert group_paths.unique_other_map_output_path.exists()
    assert group_paths.shared_map_output_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    total_map = np.load(group_paths.total_map_output_path)
    assert total_map.shape == (2, 2)
    assert np.isclose(total_map[0, 0], 0.16)


def test_lmeeeg_similarity_cli_writes_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = _lmeeeg_config(out_dir)

    other_paths = build_lmeeeg_model_output_paths(config, model_name="other_rate_effect")
    self_paths = build_lmeeeg_model_output_paths(config, model_name="self_rate_effect")
    other_paths.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    self_paths.summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    channel_names = ["Fz", "Cz"]
    times = np.asarray([-0.1, 0.0, 0.1], dtype=float)
    other_beta = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]],
        ],
        dtype=float,
    )
    self_beta = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.8, 1.9, 3.1], [0.4, 0.9, 1.7]],
        ],
        dtype=float,
    )

    for output_paths, predictor_name, betas in (
        (other_paths, "other_rate", other_beta),
        (self_paths, "self_rate", self_beta),
    ):
        np.save(output_paths.betas_output_path, betas)
        np.save(output_paths.t_values_output_path, betas * 2.0)
        np.save(output_paths.times_output_path, times)
        output_paths.channel_names_output_path.write_text(json.dumps(channel_names) + "\n", encoding="utf-8")
        output_paths.column_names_output_path.write_text(json.dumps(["Intercept", predictor_name]) + "\n", encoding="utf-8")
        output_paths.test_predictors_output_path.write_text(json.dumps([predictor_name]) + "\n", encoding="utf-8")
        output_paths.summary_output_path.write_text(
            json.dumps({"status": "ok", "predictor_of_interest": predictor_name}) + "\n",
            encoding="utf-8",
        )

    args = Namespace(config="config/config.yaml", schema="config/schema.json")
    exit_code = lmeeeg_similarity.run(args, config=config)
    assert exit_code == 0

    output_paths = build_similarity_output_paths(config)
    assert output_paths.summary_output_path.exists()
    assert output_paths.summary_table_output_path.exists()
    assert output_paths.difference_map_output_path.exists()
    assert output_paths.time_resolved_curve_output_path.exists()
    assert output_paths.time_resolved_p_values_output_path.exists()
    assert output_paths.whole_map_null_output_path.exists()

    summary = json.loads(output_paths.summary_output_path.read_text(encoding="utf-8"))
    summary_table = pd.read_parquet(output_paths.summary_table_output_path)
    difference_map = np.load(output_paths.difference_map_output_path)
    time_curve = np.load(output_paths.time_resolved_curve_output_path)

    assert summary["status"] == "ok"
    assert summary["analysis"] == "lmeeeg_similarity"
    assert summary["input"]["predictor_a"] == "other_rate"
    assert summary["input"]["predictor_b"] == "self_rate"
    assert summary["whole_map_similarity"]["pearson"]["p_value"] is not None
    assert summary["time_resolved_topography"]["correction"] == "tfce"
    assert set(summary_table["metric"]) >= {"pearson", "cosine", "mean_signed_difference", "mean_absolute_difference", "rms_difference"}
    assert difference_map.shape == (2, 3)
    assert time_curve.shape == (3,)
    assert np.isclose(summary["whole_map_difference"]["mean_signed_difference"], float(np.mean(difference_map)))
    assert output_paths.figure_manifest_output_path.exists()
    figure_manifest = json.loads(output_paths.figure_manifest_output_path.read_text(encoding="utf-8"))
    assert figure_manifest["plot_count"] == 1
    assert figure_manifest["plots"][0]["plot_name"] == "similarity_summary"
    assert all(Path(path).exists() for path in figure_manifest["plots"][0]["files"])


def test_figures_similarity_cli_renders_from_existing_outputs_without_rerunning_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "out"
    config = _lmeeeg_config(out_dir)

    other_paths = build_lmeeeg_model_output_paths(config, model_name="other_rate_effect")
    self_paths = build_lmeeeg_model_output_paths(config, model_name="self_rate_effect")
    other_paths.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    self_paths.summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    channel_names = ["Fz", "Cz"]
    times = np.asarray([-0.1, 0.0, 0.1], dtype=float)
    other_beta = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]],
        ],
        dtype=float,
    )
    self_beta = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.8, 1.9, 3.1], [0.4, 0.9, 1.7]],
        ],
        dtype=float,
    )

    for output_paths, predictor_name, betas in (
        (other_paths, "other_rate", other_beta),
        (self_paths, "self_rate", self_beta),
    ):
        np.save(output_paths.betas_output_path, betas)
        np.save(output_paths.t_values_output_path, betas * 2.0)
        np.save(output_paths.times_output_path, times)
        output_paths.channel_names_output_path.write_text(json.dumps(channel_names) + "\n", encoding="utf-8")
        output_paths.column_names_output_path.write_text(json.dumps(["Intercept", predictor_name]) + "\n", encoding="utf-8")
        output_paths.test_predictors_output_path.write_text(json.dumps([predictor_name]) + "\n", encoding="utf-8")
        output_paths.summary_output_path.write_text(
            json.dumps({"status": "ok", "predictor_of_interest": predictor_name}) + "\n",
            encoding="utf-8",
        )

    similarity_args = Namespace(config="config/config.yaml", schema="config/schema.json", skip_figures=True)
    assert lmeeeg_similarity.run(similarity_args, config=config) == 0

    monkeypatch.setattr(
        figures_similarity,
        "build_similarity_summary_figure_manifest",
        lmeeeg_similarity_viz.build_similarity_summary_figure_manifest,
    )

    def _unexpected_rerun(_config: dict) -> dict[str, object]:
        raise AssertionError("Similarity analysis should not rerun while rendering figures.")

    monkeypatch.setattr(lmeeeg_similarity, "run_similarity_analysis", _unexpected_rerun)

    figure_args = Namespace(config="config/config.yaml", schema="config/schema.json", viz_config="config/viz.yaml", output=None)
    assert figures_similarity.run(figure_args, config=config) == 0

    output_paths = build_similarity_output_paths(config)
    assert output_paths.figure_manifest_output_path.exists()
    figure_manifest = json.loads(output_paths.figure_manifest_output_path.read_text(encoding="utf-8"))
    assert figure_manifest["plot_count"] == 1
    assert figure_manifest["plots"][0]["plot_name"] == "similarity_summary"
    assert all(Path(path).exists() for path in figure_manifest["plots"][0]["files"])


def test_resolve_similarity_settings_supports_explicit_refit_inference_config(tmp_path: Path) -> None:
    config = _lmeeeg_config(tmp_path / "out")
    config["similarity"] = {
        "enabled": True,
        "analysis_name": "other_vs_self_beta",
        "predictor_a": "other_rate",
        "predictor_b": "self_rate",
        "map_type": "beta",
        "metrics": {
            "whole_map": ["pearson", "cosine"],
            "time_resolved_topography": {"enabled": True, "method": "pearson"},
        },
        "inference": {
            "method": "refit_permutation",
            "n_permutations": 12,
            "tail": "two-sided",
            "tfce": {"enabled": True, "e_power": 0.5, "h_power": 2.0, "dh": 0.1},
            "permutation_scheme": {
                "kind": "circular_shift_within_subject",
                "group_by": ["subject_id"],
                "within_run": True,
                "random_seed": 42,
            },
        },
    }

    resolved = resolve_similarity_settings(config)

    assert resolved.analysis_name == "other_vs_self_beta"
    assert resolved.input_settings.predictor_a == "other_rate"
    assert resolved.input_settings.predictor_b == "self_rate"
    assert resolved.input_settings.value_field == "beta"
    assert resolved.inference.method == "refit_permutation"
    assert resolved.inference.n_permutations == 12
    assert resolved.inference.permutation_scheme.kind == "circular_shift_within_subject"
    assert resolved.inference.permutation_scheme.group_by == ("subject_id",)
    assert resolved.inference.permutation_scheme.within_run is True
    assert resolved.inference.random_seed == 42


def test_similarity_circular_shift_permutation_preserves_within_block_values() -> None:
    metadata = pd.DataFrame(
        {
            "subject_id": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "run": ["1", "1", "2", "1", "1", "2"],
            "other_rate": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "self_rate": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    grouping = _build_similarity_permutation_grouping(
        metadata,
        scheme=SimilarityPermutationScheme(
            kind="circular_shift_within_subject",
            group_by=("subject_id",),
            within_run=True,
        ),
    )

    permutation_indices = _generate_similarity_permutation_indices(
        n_rows=len(metadata),
        grouping=grouping,
        rng=np.random.default_rng(3),
    )
    permuted = _permute_similarity_predictor(
        metadata,
        predictor_names=("other_rate", "self_rate"),
        permutation_indices=permutation_indices,
    )

    assert permutation_indices.shape == (len(metadata),)
    for group_indices in grouping.group_indices:
        original_other = metadata.loc[group_indices, "other_rate"].to_numpy(dtype=float)
        permuted_other = permuted.loc[group_indices, "other_rate"].to_numpy(dtype=float)
        original_self = metadata.loc[group_indices, "self_rate"].to_numpy(dtype=float)
        permuted_self = permuted.loc[group_indices, "self_rate"].to_numpy(dtype=float)
        np.testing.assert_array_equal(np.sort(permuted_other), np.sort(original_other))
        np.testing.assert_array_equal(np.sort(permuted_self), np.sort(original_self))
    assert not np.array_equal(
        permuted.loc[grouping.group_indices[0], "other_rate"].to_numpy(dtype=float),
        metadata.loc[grouping.group_indices[0], "other_rate"].to_numpy(dtype=float),
    )


def test_compute_lmeeeg_pooled_beta_map_matches_saved_stage_betas(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = _lmeeeg_config(out_dir)
    config["lmeeeg"]["models"]["other_rate_effect"]["formula"] = "~ other_rate"
    config["lmeeeg"]["models"]["other_rate_effect"]["standardize"] = ["other_rate"]

    rows = [
        {"subject_id": "sub-001", "task": "conversation", "run": "1", "status": "ok"},
        {"subject_id": "sub-002", "task": "conversation", "run": "1", "status": "ok"},
    ]
    metadata_by_subject: dict[str, pd.DataFrame] = {}
    data_by_subject: dict[str, np.ndarray] = {}
    for subject_index, row in enumerate(rows):
        subject_id = str(row["subject_id"])
        other_rate = np.asarray([0.0, 1.4, -0.6, 2.1], dtype=float)
        metadata = pd.DataFrame(
            {
                "subject_id": [subject_id] * 4,
                "task": ["conversation"] * 4,
                "run": ["1"] * 4,
                "other_rate": other_rate,
            }
        )
        subject_intercept = float(subject_index + 1)
        other_map = np.asarray([[0.8, 0.4, 0.2], [0.5, 0.3, 0.1]], dtype=float)
        baseline = np.asarray([[0.1, 0.2, 0.0], [0.0, 0.1, -0.1]], dtype=float)
        data = np.zeros((len(metadata), 2, 3), dtype=float)
        for trial_index in range(len(metadata)):
            data[trial_index] = baseline + (other_rate[trial_index] * other_map) + subject_intercept
        metadata_by_subject[subject_id] = metadata
        data_by_subject[subject_id] = data

    dataset_index_path = _write_similarity_epochs(
        tmp_path,
        config,
        rows,
        metadata_by_subject=metadata_by_subject,
        data_by_subject=data_by_subject,
    )
    fit_args = Namespace(
        config="config/config.yaml",
        schema="config/schema.json",
        dataset_index=str(dataset_index_path),
        subject_id=None,
        task=None,
        run=None,
        epochs_path=None,
        metadata_path=None,
        run_all=False,
        output=None,
        target=None,
        trial_regressors_csv=None,
        config_section="lmeeeg",
        model_name="other_rate_effect",
    )
    assert lmeeeg_subject.run(fit_args, config=config) == 0

    pooled_inputs = load_pooled_lmeeeg_inputs(config, rows, config_section="lmeeeg", model_name="other_rate_effect")
    map_estimate = compute_lmeeeg_pooled_beta_map(
        config,
        pooled_inputs=pooled_inputs,
        config_section="lmeeeg",
        model_name="other_rate_effect",
        predictor_name="other_rate",
    )
    saved_paths = build_lmeeeg_model_output_paths(config, model_name="other_rate_effect")
    saved_betas = np.load(saved_paths.betas_output_path)
    saved_column_names = json.loads(saved_paths.column_names_output_path.read_text(encoding="utf-8"))
    saved_effect_index = saved_column_names.index("other_rate")

    assert map_estimate.effect_name == "other_rate"
    assert map_estimate.residualisation_depends_on_predictor is True
    np.testing.assert_allclose(map_estimate.beta_map, saved_betas[saved_effect_index])
    np.testing.assert_allclose(map_estimate.times, np.load(saved_paths.times_output_path))
    assert map_estimate.channel_names == json.loads(saved_paths.channel_names_output_path.read_text(encoding="utf-8"))


def test_similarity_refit_strategy_marks_residualisation_as_predictor_dependent() -> None:
    strategy = _resolve_similarity_refit_estimator_strategy()

    assert strategy.estimator_name.startswith("lmeEEG pooled beta map")
    assert strategy.observed_map_source == "saved pooled betas.npy from the standard lmeEEG stage"
    assert strategy.residualisation_reused is False
    assert strategy.residualisation_policy == "recomputed_per_permutation"
    assert "predictor-dependent" in strategy.residualisation_reason


def test_lmeeeg_similarity_none_mode_writes_descriptive_outputs_only(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = _lmeeeg_config(out_dir)
    config["similarity"]["inferential_test"]["enabled"] = False

    other_paths = build_lmeeeg_model_output_paths(config, model_name="other_rate_effect")
    self_paths = build_lmeeeg_model_output_paths(config, model_name="self_rate_effect")
    other_paths.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    self_paths.summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    channel_names = ["Fz", "Cz"]
    times = np.asarray([-0.1, 0.0, 0.1], dtype=float)
    other_beta = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]],
        ],
        dtype=float,
    )
    self_beta = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.8, 1.9, 3.1], [0.4, 0.9, 1.7]],
        ],
        dtype=float,
    )

    for output_paths, predictor_name, betas in (
        (other_paths, "other_rate", other_beta),
        (self_paths, "self_rate", self_beta),
    ):
        np.save(output_paths.betas_output_path, betas)
        np.save(output_paths.t_values_output_path, betas * 2.0)
        np.save(output_paths.times_output_path, times)
        output_paths.channel_names_output_path.write_text(json.dumps(channel_names) + "\n", encoding="utf-8")
        output_paths.column_names_output_path.write_text(json.dumps(["Intercept", predictor_name]) + "\n", encoding="utf-8")
        output_paths.test_predictors_output_path.write_text(json.dumps([predictor_name]) + "\n", encoding="utf-8")
        output_paths.summary_output_path.write_text(
            json.dumps({"status": "ok", "predictor_of_interest": predictor_name}) + "\n",
            encoding="utf-8",
        )

    args = Namespace(config="config/config.yaml", schema="config/schema.json")
    exit_code = lmeeeg_similarity.run(args, config=config)
    assert exit_code == 0

    output_paths = build_similarity_output_paths(config)
    summary = json.loads(output_paths.summary_output_path.read_text(encoding="utf-8"))

    assert summary["whole_map_similarity"]["pearson"]["p_value"] is None
    assert summary["inference"]["descriptive_only"] is True
    assert summary["whole_map_null_path"] is None
    assert summary["time_resolved_topography"]["corrected_p_values_path"] is None
    assert not output_paths.whole_map_null_output_path.exists()
    assert not output_paths.time_resolved_p_values_output_path.exists()
    assert output_paths.figure_manifest_output_path.exists()


def test_lmeeeg_similarity_sign_flip_metadata_preserves_legacy_mode(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = _lmeeeg_config(out_dir)

    other_paths = build_lmeeeg_model_output_paths(config, model_name="other_rate_effect")
    self_paths = build_lmeeeg_model_output_paths(config, model_name="self_rate_effect")
    other_paths.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    self_paths.summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    channel_names = ["Fz", "Cz"]
    times = np.asarray([-0.1, 0.0, 0.1], dtype=float)
    for output_paths, predictor_name, betas in (
        (
            other_paths,
            "other_rate",
            np.asarray([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]]], dtype=float),
        ),
        (
            self_paths,
            "self_rate",
            np.asarray([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.8, 1.9, 3.1], [0.4, 0.9, 1.7]]], dtype=float),
        ),
    ):
        np.save(output_paths.betas_output_path, betas)
        np.save(output_paths.t_values_output_path, betas * 2.0)
        np.save(output_paths.times_output_path, times)
        output_paths.channel_names_output_path.write_text(json.dumps(channel_names) + "\n", encoding="utf-8")
        output_paths.column_names_output_path.write_text(json.dumps(["Intercept", predictor_name]) + "\n", encoding="utf-8")
        output_paths.test_predictors_output_path.write_text(json.dumps([predictor_name]) + "\n", encoding="utf-8")
        output_paths.summary_output_path.write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")

    args = Namespace(config="config/config.yaml", schema="config/schema.json")
    assert lmeeeg_similarity.run(args, config=config) == 0

    output_paths = build_similarity_output_paths(config)
    metadata = json.loads(output_paths.metadata_output_path.read_text(encoding="utf-8"))
    assert metadata["inference"]["method"] == "sign_flip"
    assert metadata["inference"]["p_value_source"] == "legacy_sign_flip_over_existing_maps"
    assert metadata["inference"]["null_map_estimator_strategy"] is None


def test_similarity_topomap_selection_is_deterministic() -> None:
    times = np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
    observed_curve = np.asarray([0.01, 0.04, 0.78, 0.55, 0.02, 0.61, 0.08], dtype=float)
    significant_mask = np.asarray([False, False, True, True, False, True, False], dtype=bool)
    time_mask = np.ones(times.shape, dtype=bool)

    selected = lmeeeg_similarity_viz._select_topomap_indices(
        times=times,
        observed_curve=observed_curve,
        significant_mask=significant_mask,
        time_mask=time_mask,
        n_topomaps=4,
        window_radius_ms=50.0,
    )

    assert selected == [2, 5]


def test_similarity_topomap_window_average_returns_expected_shape() -> None:
    difference_map = np.asarray(
        [
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
        ],
        dtype=float,
    )
    times = np.asarray([-0.05, 0.0, 0.05], dtype=float)

    averaged_map, window_mask = lmeeeg_similarity_viz._average_map_window(
        values_map=difference_map,
        times=times,
        center_index=1,
        window_radius_ms=50.0,
    )

    assert averaged_map.shape == (2,)
    np.testing.assert_allclose(averaged_map, np.asarray([3.0, 4.0], dtype=float))
    np.testing.assert_array_equal(window_mask, np.asarray([True, True, True], dtype=bool))


def test_similarity_significance_bar_aligns_with_time_axis_length() -> None:
    figure, axis = plt.subplots()
    times_ms = np.asarray([-100.0, 0.0, 80.0], dtype=float)
    significant_mask = np.asarray([False, True, True], dtype=bool)

    mesh = lmeeeg_similarity_viz._plot_significance_bar(
        axis,
        times_ms=times_ms,
        significant_mask=significant_mask,
        color="#2F6B3B",
        label="TFCE-corrected p < 0.05",
        fontsize=9.0,
    )

    assert np.asarray(mesh.get_array()).shape == (1, 3)
    assert axis.get_xlim() == pytest.approx((-150.0, 120.0))
    plt.close(figure)


def test_similarity_topomap_color_scale_is_symmetric() -> None:
    vmin, vmax = lmeeeg_similarity_viz._compute_symmetric_topomap_limits(
        [
            np.asarray([1.0, -2.5, 0.2], dtype=float),
            np.asarray([-0.3, 0.8, 1.4], dtype=float),
        ],
        requested_vlim=None,
    )

    assert vmin == pytest.approx(-2.5)
    assert vmax == pytest.approx(2.5)


def test_similarity_summary_figure_manifest_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    config = _lmeeeg_config(out_dir)
    output_paths = build_similarity_output_paths(config)
    output_paths.root_dir.mkdir(parents=True, exist_ok=True)
    other_paths = build_lmeeeg_model_output_paths(config, model_name="other_rate_effect")
    self_paths = build_lmeeeg_model_output_paths(config, model_name="self_rate_effect")
    other_paths.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    self_paths.summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    difference_map = np.asarray(
        [
            [0.2, 0.8, 1.2, 0.6, 0.1],
            [-0.1, 0.4, 0.9, 0.5, 0.0],
            [0.0, 0.3, 0.6, 0.4, -0.1],
            [-0.2, 0.2, 0.7, 0.3, -0.2],
        ],
        dtype=float,
    )
    times = np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=float)
    other_map = np.asarray(
        [
            [0.4, 1.1, 1.6, 0.9, 0.2],
            [0.1, 0.7, 1.2, 0.8, 0.3],
            [0.2, 0.5, 0.9, 0.6, 0.1],
            [0.0, 0.4, 0.8, 0.5, 0.0],
        ],
        dtype=float,
    )
    self_map = other_map - difference_map
    observed_curve = np.asarray([0.01, 0.18, 0.62, 0.25, 0.05], dtype=float)
    corrected_p = np.asarray([0.20, 0.03, 0.01, 0.20, 0.60], dtype=float)
    null_curves = np.asarray(
        [
            [-0.10, 0.04, 0.08, 0.05, -0.02],
            [-0.08, 0.02, 0.10, 0.03, -0.01],
            [-0.12, 0.03, 0.07, 0.04, 0.00],
            [-0.09, 0.01, 0.09, 0.02, -0.03],
        ],
        dtype=float,
    )

    np.save(output_paths.difference_map_output_path, difference_map)
    np.save(output_paths.times_output_path, times)
    np.save(output_paths.time_resolved_curve_output_path, observed_curve)
    np.save(output_paths.time_resolved_p_values_output_path, corrected_p)
    np.save(output_paths.time_resolved_null_curves_output_path, null_curves)
    output_paths.channel_names_output_path.write_text(json.dumps(["Fz", "Cz", "Pz", "Oz"]) + "\n", encoding="utf-8")
    for model_paths, predictor_name, values in (
        (other_paths, "other_rate", other_map),
        (self_paths, "self_rate", self_map),
    ):
        betas = np.stack([np.zeros_like(values), values], axis=0)
        np.save(model_paths.betas_output_path, betas)
        np.save(model_paths.t_values_output_path, betas * 2.0)
        np.save(model_paths.times_output_path, times)
        model_paths.channel_names_output_path.write_text(json.dumps(["Fz", "Cz", "Pz", "Oz"]) + "\n", encoding="utf-8")
        model_paths.column_names_output_path.write_text(json.dumps(["Intercept", predictor_name]) + "\n", encoding="utf-8")
        model_paths.test_predictors_output_path.write_text(json.dumps([predictor_name]) + "\n", encoding="utf-8")
        model_paths.summary_output_path.write_text(
            json.dumps({"status": "ok", "predictor_of_interest": predictor_name}) + "\n",
            encoding="utf-8",
        )
    output_paths.summary_output_path.write_text(
        json.dumps(
            {
                "analysis_name": "other_vs_self",
                "input": {"predictor_a": "other_rate", "predictor_b": "self_rate"},
                "time_resolved_topography": {
                    "curve_path": str(output_paths.time_resolved_curve_output_path),
                    "corrected_p_values_path": str(output_paths.time_resolved_p_values_output_path),
                    "null_curves_path": str(output_paths.time_resolved_null_curves_output_path),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_draw_similarity_topomap(**kwargs):
        axis = kwargs["axis"]
        axis.set_title(str(kwargs["title"]))
        return axis.imshow(np.zeros((2, 2), dtype=float), vmin=kwargs["vmin"], vmax=kwargs["vmax"], cmap=kwargs["cmap"])

    monkeypatch.setattr(lmeeeg_similarity_viz, "_draw_similarity_topomap", fake_draw_similarity_topomap)

    manifest = lmeeeg_similarity_viz.build_similarity_summary_figure_manifest(
        config=config,
        out_dir=out_dir,
        manifest_path=output_paths.figure_manifest_output_path,
        figure_config={"n_topomaps": 4, "window_radius_ms": 50.0},
        theme_config={"colors": {"shared": "#2F6B3B", "neutral": "#4B5563", "reference": "#6B7280"}},
        formats=("png",),
        dpi=72,
    )

    assert manifest["plot_count"] == 1
    plot = manifest["plots"][0]
    assert plot["plot_name"] == "similarity_summary"
    assert plot["n_topomaps"] == 1
    assert plot["plotting_tmin_s"] == -0.2
    assert plot["plotting_tmax_s"] == 0.2
    assert len(plot["files"]) == 1
    assert Path(plot["files"][0]).exists()


def test_lmeeeg_similarity_refit_permutation_writes_null_shapes(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = _lmeeeg_config(out_dir)
    config["similarity"]["inference"] = {
        "method": "refit_permutation",
        "n_permutations": 4,
        "tail": "two-sided",
        "tfce": {"enabled": True, "e_power": 0.5, "h_power": 2.0, "dh": 0.1},
        "permutation_scheme": {
            "kind": "circular_shift_within_subject",
            "group_by": ["subject_id"],
            "within_run": True,
            "random_seed": 11,
        },
    }
    config["similarity"].pop("inferential_test", None)
    config["lmeeeg"]["models"]["other_rate_effect"]["formula"] = "~ other_rate"
    config["lmeeeg"]["models"]["other_rate_effect"]["standardize"] = ["other_rate"]
    config["lmeeeg"]["models"]["self_rate_effect"]["formula"] = "~ self_rate"
    config["lmeeeg"]["models"]["self_rate_effect"]["standardize"] = ["self_rate"]

    rows = [
        {"subject_id": "sub-001", "task": "conversation", "run": "1", "status": "ok"},
        {"subject_id": "sub-002", "task": "conversation", "run": "1", "status": "ok"},
    ]
    metadata_by_subject: dict[str, pd.DataFrame] = {}
    data_by_subject: dict[str, np.ndarray] = {}
    for subject_index, row in enumerate(rows):
        subject_id = str(row["subject_id"])
        other_rate = np.asarray([0.0, 1.4, -0.6, 2.1], dtype=float)
        self_rate = np.asarray([1.0, -0.2, 0.7, -1.1], dtype=float)
        latency = np.asarray([0.15, 0.55, 0.35, 0.9], dtype=float)
        duration = np.asarray([0.5, 0.72, 0.61, 0.83], dtype=float)
        time_within_run = np.asarray([0.0, 1.0, 1.5, 3.0], dtype=float)
        metadata = pd.DataFrame(
            {
                "subject_id": [subject_id] * 4,
                "task": ["conversation"] * 4,
                "run": ["1"] * 4,
                "other_rate": other_rate,
                "self_rate": self_rate,
                "latency": latency,
                "duration_s": duration,
                "time_within_run": time_within_run,
            }
        )
        subject_intercept = float(subject_index + 1)
        other_map = np.asarray([[0.8, 0.4, 0.2], [0.5, 0.3, 0.1]], dtype=float)
        self_map = np.asarray([[0.6, 0.1, -0.1], [0.2, -0.2, -0.4]], dtype=float)
        baseline = np.asarray([[0.1, 0.2, 0.0], [0.0, 0.1, -0.1]], dtype=float)
        data = np.zeros((len(metadata), 2, 3), dtype=float)
        for trial_index in range(len(metadata)):
            data[trial_index] = (
                baseline
                + (other_rate[trial_index] * other_map)
                + (self_rate[trial_index] * self_map)
                + subject_intercept
            )
        metadata_by_subject[subject_id] = metadata
        data_by_subject[subject_id] = data

    dataset_index_path = _write_similarity_epochs(
        tmp_path,
        config,
        rows,
        metadata_by_subject=metadata_by_subject,
        data_by_subject=data_by_subject,
    )

    fit_args = Namespace(
        config="config/config.yaml",
        schema="config/schema.json",
        dataset_index=str(dataset_index_path),
        subject_id=None,
        task=None,
        run=None,
        epochs_path=None,
        metadata_path=None,
        run_all=False,
        output=None,
        target=None,
        trial_regressors_csv=None,
        config_section="lmeeeg",
    )
    fit_args.model_name = "other_rate_effect"
    assert lmeeeg_subject.run(fit_args, config=config) == 0
    fit_args.model_name = "self_rate_effect"
    assert lmeeeg_subject.run(fit_args, config=config) == 0

    similarity_args = Namespace(config="config/config.yaml", schema="config/schema.json")
    assert lmeeeg_similarity.run(similarity_args, config=config) == 0

    output_paths = build_similarity_output_paths(config)
    summary = json.loads(output_paths.summary_output_path.read_text(encoding="utf-8"))
    metadata = json.loads(output_paths.metadata_output_path.read_text(encoding="utf-8"))
    whole_nulls = np.load(output_paths.whole_map_null_output_path)
    corrected_p = np.load(output_paths.time_resolved_p_values_output_path)
    null_curves = np.load(output_paths.time_resolved_null_curves_output_path)
    null_max = np.load(output_paths.time_resolved_null_output_path)

    assert summary["inference"]["method"] == "refit_permutation"
    assert metadata["inference"]["method"] == "refit_permutation"
    assert metadata["inference"]["p_value_source"] == "recomputed_lmeeeg_pooled_beta_maps"
    assert metadata["observed_map_estimator"]["source"] == "saved pooled betas.npy from the standard lmeEEG stage"
    assert metadata["inference"]["null_map_estimator_strategy"]["residualisation_reused"] is False
    assert whole_nulls["pearson"].shape == (4,)
    assert whole_nulls["cosine"].shape == (4,)
    assert corrected_p.shape == (3,)
    assert null_curves.shape == (4, 3)
    assert null_max.shape == (4,)
    assert summary["time_resolved_topography"]["correction"] == "tfce"


def test_compute_tfce_1d_boosts_contiguous_timepoints() -> None:
    tfce = _compute_tfce_1d(
        np.asarray([0.0, 1.0, 1.0, 0.4], dtype=float),
        h_power=2.0,
        e_power=0.5,
        h_start=0.0,
        dh=0.1,
    )

    assert tfce.shape == (4,)
    assert tfce[1] > tfce[3]
    assert tfce[2] > tfce[3]


def test_figures_lmeeeg_variance_partition_cli_writes_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = _lmeeeg_config(out_dir)
    group_paths = build_variance_partition_group_output_paths(config)
    group_paths.group_dir.mkdir(parents=True, exist_ok=True)

    write_parquet_dataframe(
        pd.DataFrame(
            {
                "channel": ["Fp1", "Fp1", "Oz", "Oz"],
                "time": [-0.1, 0.0, -0.1, 0.0],
                "channel_index": [0, 0, 1, 1],
                "time_index": [0, 1, 0, 1],
                "unique_self": [0.04, 0.06, 0.03, 0.05],
                "unique_other": [0.03, 0.02, 0.04, 0.03],
                "shared": [0.01, 0.02, 0.05, 0.06],
            }
        ),
        group_paths.table_output_path,
    )

    viz_config_path = tmp_path / "viz.yaml"
    viz_config_path.write_text(
        yaml.safe_dump(
            {
                "viz": {
                    "formats": ["png"],
                    "dpi": 72,
                    "theme": {
                        "colors": {
                            "self": "#8C2D2D",
                            "other": "#1F4E79",
                            "shared": "#2F6B3B",
                        }
                    },
                    "figures": {
                        "variance_partition": {
                            "use_proportions": False,
                            "rois": {
                                "anterior": ["Fp1"],
                                "posterior": ["Oz"],
                            },
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    args = Namespace(
        config="config/config.yaml",
        schema="config/schema.json",
        viz_config=str(viz_config_path),
        input=str(group_paths.table_output_path),
        output=None,
    )
    exit_code = figures_lmeeeg_variance_partition.run(args, config=config)

    assert exit_code == 0
    manifest_path = out_dir / "figures" / "lmeeeg_variance_partition" / "figure_manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["plot_count"] == 1
    plot = manifest["plots"][0]
    assert plot["plot_name"] == "variance_partition_summary"
    assert plot["use_proportions"] is False
    assert plot["rois"]["anterior"] == ["Fp1"]
    assert plot["rois"]["posterior"] == ["Oz"]
    assert len(plot["files"]) == 1
    assert Path(plot["files"][0]).exists()


def test_variance_partition_workflow_rules_are_present() -> None:
    figures_rules = Path("workflow/rules/figures.smk").read_text(encoding="utf-8")
    lmeeeg_rules = Path("workflow/rules/lmeeeg.smk").read_text(encoding="utf-8")
    stats_rules = Path("workflow/rules/stats.smk").read_text(encoding="utf-8")
    targets_rules = Path("workflow/rules/targets.smk").read_text(encoding="utf-8")

    assert "rule lmeeeg_variance_partition_fit" in lmeeeg_rules
    assert "rule lmeeeg_variance_partition_stage" in lmeeeg_rules
    assert "rule lmeeeg_variance_partition_group_stats" in stats_rules
    assert "rule lmeeeg_similarity" in stats_rules
    assert "rule figures_lmeeeg_main_results" in figures_rules
    assert "rule figures_similarity" in figures_rules
    assert "rule figures_lmeeeg_variance_partition" in figures_rules
    assert "rule similarity_all" in targets_rules
    assert "rule similarity_figures_all" in targets_rules
    assert "SIMILARITY_FIGURES_OUTPUT" in targets_rules
    assert "rule lmeeeg_variance_partition_all" in targets_rules
    assert "rule lmeeeg_main_results_figures_all" in targets_rules
    assert "LMEEEG_MAIN_RESULTS_FIGURES_OUTPUT" in targets_rules
    assert "rule lmeeeg_variance_partition_figures_all" in targets_rules
    assert "LMEEEG_VARIANCE_PARTITION_FIGURES_OUTPUT" in targets_rules
    assert "rule lmeeeg_all" in targets_rules


def test_write_inference_outputs_writes_cluster_pvalues_csv(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = _lmeeeg_config(out_dir)
    inference = SimpleNamespace(
        correction="cluster",
        observed_statistic=np.zeros((2, 3), dtype=float),
        corrected_p_values=np.asarray([[0.02, 0.02, 1.0], [1.0, 0.04, 0.04]], dtype=float),
        null_distribution=np.asarray([0.0], dtype=float),
        cluster_p_values=np.asarray([0.04, 0.02, 0.3], dtype=float),
        clusters=[
            np.asarray([[False, True], [False, True], [False, False]], dtype=bool),
            np.asarray([[True, False], [True, False], [False, False]], dtype=bool),
            np.asarray([[True, True], [False, False], [False, False]], dtype=bool),
        ],
        backend_metadata={},
    )

    summary = lmeeeg_group_stats._write_inference_outputs(
        config,
        model_name="speech_rate_item_intercept",
        predictor_name="speech_rate",
        effect_name="speech_rate",
        inference=inference,
        effect_t_map=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
        channel_names=["Fz", "Cz"],
        times=np.asarray([0.0, 0.1, 0.2], dtype=float),
    )

    csv_path = Path(summary["p_values_csv"])
    csv_df = pd.read_csv(csv_path)

    assert summary["observed_statistic_kind"] == "t_value"
    assert Path(summary["permutation_observed_statistic"]).exists()
    np.testing.assert_array_equal(
        np.load(summary["observed_statistic"]),
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
    )
    np.testing.assert_array_equal(np.load(summary["permutation_observed_statistic"]), inference.observed_statistic)
    assert list(csv_df.columns) == ["p_values", "cluster_size", "duration", "spatial_extent"]
    assert csv_df.to_dict(orient="records") == [
        {"p_values": 0.02, "cluster_size": 2, "duration": 0.2, "spatial_extent": 1},
        {"p_values": 0.04, "cluster_size": 2, "duration": 0.2, "spatial_extent": 1},
        {"p_values": 0.3, "cluster_size": 2, "duration": 0.1, "spatial_extent": 2},
    ]


def test_write_inference_outputs_writes_tfce_pvalues_csv_sorted_by_pvalue(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = _lmeeeg_config(out_dir)
    inference = SimpleNamespace(
        correction="tfce",
        observed_statistic=np.zeros((2, 2), dtype=float),
        corrected_p_values=np.asarray([[0.03, 0.2], [0.01, 0.04]], dtype=float),
        null_distribution=np.asarray([0.0], dtype=float),
        cluster_p_values=np.asarray([], dtype=float),
        clusters=[],
        backend_metadata={},
    )

    summary = lmeeeg_group_stats._write_inference_outputs(
        config,
        model_name="speech_rate_item_intercept",
        predictor_name="speech_rate",
        effect_name="speech_rate",
        inference=inference,
        effect_t_map=np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float),
        channel_names=["Fz", "Cz"],
        times=np.asarray([0.0, 0.1], dtype=float),
    )

    csv_path = Path(summary["p_values_csv"])
    csv_df = pd.read_csv(csv_path)

    assert summary["observed_statistic_kind"] == "t_value"
    assert summary["permutation_observed_statistic_kind"] == "tfce_observed_statistic"
    np.testing.assert_array_equal(
        np.load(summary["observed_statistic"]),
        np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float),
    )
    assert list(csv_df.columns) == ["p_values", "channel", "time"]
    assert csv_df.to_dict(orient="records") == [
        {"p_values": 0.01, "channel": "Cz", "time": 0.0},
        {"p_values": 0.03, "channel": "Fz", "time": 0.0},
        {"p_values": 0.04, "channel": "Cz", "time": 0.1},
        {"p_values": 0.2, "channel": "Fz", "time": 0.1},
    ]


def test_resolve_adjacency_uses_configured_preprocessing_montage(monkeypatch) -> None:
    import mne

    captured: dict[str, str] = {}

    original_make_standard_montage = mne.channels.make_standard_montage
    original_find_ch_adjacency = mne.channels.find_ch_adjacency

    def fake_make_standard_montage(name: str):
        captured["montage_name"] = name
        return original_make_standard_montage("standard_1020")

    def fake_find_ch_adjacency(info, ch_type):
        return np.eye(len(info.ch_names), dtype=float), info.ch_names

    monkeypatch.setattr(mne.channels, "make_standard_montage", fake_make_standard_montage)
    monkeypatch.setattr(mne.channels, "find_ch_adjacency", fake_find_ch_adjacency)

    config = {
        "preprocessing": {
            "montage": {
                "name": "biosemi64",
            }
        }
    }
    stats_cfg = {"stats": {"adjacency": "mne_default"}}

    adjacency = lmeeeg_group_stats._resolve_adjacency(config, stats_cfg, ["Fz", "Cz"])

    assert captured["montage_name"] == "biosemi64"
    assert adjacency is not None


def test_build_lmeeeg_qc_manifest_falls_back_to_nested_model_dir(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    lmeeeg_root = out_dir / "lmeeeg"
    nested_model_dir = lmeeeg_root / "sub-001" / "task-conversation" / "run-1" / "speech_rate_item_intercept"
    nested_model_dir.mkdir(parents=True, exist_ok=True)

    np.save(nested_model_dir / "t_values.npy", np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=float))
    np.save(nested_model_dir / "times.npy", np.asarray([-0.1, 0.0], dtype=float))
    (nested_model_dir / "channel_names.json").write_text(json.dumps(["Fz", "Cz"]) + "\n", encoding="utf-8")
    (nested_model_dir / "column_names.json").write_text(json.dumps(["speech_rate"]) + "\n", encoding="utf-8")
    (nested_model_dir / "summary.json").write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")

    captured_output_stems: list[Path] = []

    def fake_plot_joint_model_weights(*args, output_stem: Path, **kwargs) -> list[Path]:
        captured_output_stems.append(output_stem)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        written_path = output_stem.with_suffix(".png")
        written_path.write_text("stub\n", encoding="utf-8")
        return [written_path]

    monkeypatch.setattr(lmeeeg_viz, "plot_joint_model_weights", fake_plot_joint_model_weights)

    manifest_path = out_dir / "figures" / "lmeeeg" / "figure_manifest.json"
    manifest = lmeeeg_viz.build_lmeeeg_qc_manifest(
        out_dir=out_dir,
        lmeeeg_root=lmeeeg_root,
        manifest_path=manifest_path,
        formats=("png",),
    )

    assert manifest["plot_count"] == 1
    assert captured_output_stems == [out_dir / "figures" / "lmeeeg" / "t_values" / "speech_rate_item_intercept" / "speech_rate"]
    assert manifest["plots"][0]["model_name"] == "speech_rate_item_intercept"
    assert manifest_path.exists()


def test_figures_lmeeeg_falls_back_to_stats_outputs_when_lmeeeg_dir_missing(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    stats_effect_dir = out_dir / "stats" / "lmeeeg" / "other_rate_subject_intercept" / "latency"
    stats_effect_dir.mkdir(parents=True, exist_ok=True)
    observed_path = stats_effect_dir / "observed_statistic.npy"
    corrected_p_path = stats_effect_dir / "corrected_p_values.npy"
    np.save(observed_path, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float))
    np.save(corrected_p_path, np.asarray([[0.04, 0.2], [0.03, 0.5]], dtype=float))
    (stats_effect_dir / "summary.json").write_text(
        json.dumps({"observed_statistic": str(observed_path), "corrected_p_values": str(corrected_p_path)}) + "\n",
        encoding="utf-8",
    )

    viz_config_path = tmp_path / "viz.yaml"
    viz_config_path.write_text(yaml.safe_dump({"viz": {"formats": ["png"], "dpi": 72}}), encoding="utf-8")

    monkeypatch.setattr(figures_lmeeeg, "select_lmeeeg_rows", lambda config: [{"subject_id": "sub-001", "task": "conversation", "run": "1"}])
    monkeypatch.setattr(
        figures_lmeeeg,
        "_load_or_fit_model_payloads",
        lambda config, rows, config_section, target_name: {
            "other_rate_subject_intercept": {
                "betas": np.asarray([[[0.1, 0.2], [0.3, 0.4]]], dtype=float),
                "t_values": np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=float),
                "times": np.asarray([-0.1, 0.0], dtype=float),
                "channel_names": ["Fz", "Cz"],
                "column_names": ["latency"],
            }
        },
    )

    captured_output_stems: list[Path] = []

    def fake_plot_joint_model_weights(*args, output_stem: Path, **kwargs) -> list[Path]:
        captured_output_stems.append(output_stem)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        written_path = output_stem.with_suffix(".png")
        written_path.write_text("stub\n", encoding="utf-8")
        return [written_path]

    monkeypatch.setattr(lmeeeg_viz, "plot_joint_model_weights", fake_plot_joint_model_weights)

    args = Namespace(
        config="config/config.yaml",
        schema="config/schema.json",
        viz_config=str(viz_config_path),
        output=None,
    )
    exit_code = figures_lmeeeg.run(args, config=_lmeeeg_config(out_dir))

    assert exit_code == 0
    manifest_path = out_dir / "figures" / "lmeeeg" / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["plot_count"] == 3
    assert {plot["measure"] for plot in manifest["plots"] if "measure" in plot} == {"beta", "t_value"}
    assert any(plot.get("source") == "stats_fallback" for plot in manifest["plots"])
    assert captured_output_stems == [
        out_dir / "figures" / "lmeeeg" / "betas" / "other_rate_subject_intercept" / "latency",
        out_dir / "figures" / "lmeeeg" / "t_values" / "other_rate_subject_intercept" / "latency",
        out_dir / "figures" / "lmeeeg" / "statistics" / "other_rate_subject_intercept" / "latency",
    ]


def test_figures_lmeeeg_main_results_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    config = _lmeeeg_config(out_dir)
    channel_names = ["Fz", "Cz", "Pz", "Oz"]
    times = np.asarray([-0.8, -0.5, -0.3, 0.0, 0.2], dtype=float)

    for model_name, effect_name, offset in (
        ("self_rate_effect", "self_rate", 1.0),
        ("other_rate_effect", "other_rate", 2.0),
    ):
        stats_effect_dir = out_dir / "stats" / "lmeeeg" / model_name / effect_name
        stats_effect_dir.mkdir(parents=True, exist_ok=True)
        observed_path = stats_effect_dir / "observed_statistic.npy"
        corrected_p_path = stats_effect_dir / "corrected_p_values.npy"
        observed_map = np.asarray(
            [
                [offset, offset + 0.2, offset + 0.4, offset + 0.6, offset + 0.8],
                [offset + 0.1, offset + 0.3, offset + 0.5, offset + 0.7, offset + 0.9],
                [offset - 0.2, offset, offset + 0.2, offset + 0.4, offset + 0.6],
                [offset - 0.1, offset + 0.1, offset + 0.3, offset + 0.5, offset + 0.7],
            ],
            dtype=float,
        )
        corrected_p = np.asarray(
            [
                [0.20, 0.04, 0.30, 0.01, 0.50],
                [0.10, 0.03, 0.06, 0.02, 0.20],
                [0.50, 0.10, 0.03, 0.04, 0.60],
                [0.70, 0.02, 0.08, 0.03, 0.40],
            ],
            dtype=float,
        )
        np.save(observed_path, observed_map)
        np.save(corrected_p_path, corrected_p)
        (stats_effect_dir / "summary.json").write_text(
            json.dumps({"observed_statistic": str(observed_path), "corrected_p_values": str(corrected_p_path)}) + "\n",
            encoding="utf-8",
        )

        model_dir = out_dir / "lmeeeg" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        np.save(model_dir / "times.npy", times)
        (model_dir / "channel_names.json").write_text(json.dumps(channel_names) + "\n", encoding="utf-8")

    other_paths = build_lmeeeg_model_output_paths(config, model_name="other_rate_effect")
    self_paths = build_lmeeeg_model_output_paths(config, model_name="self_rate_effect")
    difference_map = np.asarray(
        [
            [0.2, 0.8, 1.2, 0.6, 0.1],
            [-0.1, 0.4, 0.9, 0.5, 0.0],
            [0.0, 0.3, 0.6, 0.4, -0.1],
            [-0.2, 0.2, 0.7, 0.3, -0.2],
        ],
        dtype=float,
    )
    other_map = np.asarray(
        [
            [0.4, 1.1, 1.6, 0.9, 0.2],
            [0.1, 0.7, 1.2, 0.8, 0.3],
            [0.2, 0.5, 0.9, 0.6, 0.1],
            [0.0, 0.4, 0.8, 0.5, 0.0],
        ],
        dtype=float,
    )
    self_map = other_map - difference_map
    for model_paths, predictor_name, values in (
        (other_paths, "other_rate", other_map),
        (self_paths, "self_rate", self_map),
    ):
        model_paths.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        betas = np.stack([np.zeros_like(values), values], axis=0)
        np.save(model_paths.betas_output_path, betas)
        np.save(model_paths.t_values_output_path, betas * 2.0)
        np.save(model_paths.times_output_path, times)
        model_paths.channel_names_output_path.write_text(json.dumps(channel_names) + "\n", encoding="utf-8")
        model_paths.column_names_output_path.write_text(json.dumps(["Intercept", predictor_name]) + "\n", encoding="utf-8")
        model_paths.test_predictors_output_path.write_text(json.dumps([predictor_name]) + "\n", encoding="utf-8")
        model_paths.summary_output_path.write_text(
            json.dumps({"status": "ok", "predictor_of_interest": predictor_name}) + "\n",
            encoding="utf-8",
        )

    similarity_paths = build_similarity_output_paths(config)
    similarity_paths.root_dir.mkdir(parents=True, exist_ok=True)
    observed_curve = np.asarray([0.01, 0.18, 0.62, 0.25, 0.05], dtype=float)
    corrected_p = np.asarray([0.20, 0.03, 0.01, 0.20, 0.60], dtype=float)
    null_curves = np.asarray(
        [
            [-0.10, 0.04, 0.08, 0.05, -0.02],
            [-0.08, 0.02, 0.10, 0.03, -0.01],
            [-0.12, 0.03, 0.07, 0.04, 0.00],
            [-0.09, 0.01, 0.09, 0.02, -0.03],
        ],
        dtype=float,
    )
    np.save(similarity_paths.difference_map_output_path, difference_map)
    np.save(similarity_paths.times_output_path, times)
    np.save(similarity_paths.time_resolved_curve_output_path, observed_curve)
    np.save(similarity_paths.time_resolved_p_values_output_path, corrected_p)
    np.save(similarity_paths.time_resolved_null_curves_output_path, null_curves)
    similarity_paths.channel_names_output_path.write_text(json.dumps(channel_names) + "\n", encoding="utf-8")
    similarity_paths.summary_output_path.write_text(
        json.dumps(
            {
                "analysis_name": "other_vs_self",
                "input": {"predictor_a": "other_rate", "predictor_b": "self_rate"},
                "time_resolved_topography": {
                    "curve_path": str(similarity_paths.time_resolved_curve_output_path),
                    "corrected_p_values_path": str(similarity_paths.time_resolved_p_values_output_path),
                    "null_curves_path": str(similarity_paths.time_resolved_null_curves_output_path),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    behavioural_dir = out_dir / "stats" / "behavioural_residuals"
    behavioural_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "subject_id": ["sub-001", "sub-001", "sub-001", "sub-002", "sub-002", "sub-002"],
            "dyad_id": ["dyad-001", "dyad-001", "dyad-001", "dyad-001", "dyad-001", "dyad-001"],
            "self_rate": [4.1, 4.4, 4.6, 3.3, 3.5, 3.7],
            "other_rate": [3.8, 4.0, 4.2, 3.0, 3.1, 3.4],
            "latency": [0.2, 0.3, 0.4, 0.5, 0.4, 0.3],
            "duration_s": [0.9, 1.0, 1.1, 1.0, 1.1, 0.9],
            "time_within_run": [0.1, 0.2, 0.3, 0.1, 0.3, 0.5],
            "run": [1, 1, 1, 1, 1, 1],
        }
    ).to_csv(behavioural_dir / "trialwise_residuals.csv", index=False)
    r_analysis_dir = out_dir / "stats" / "r_analysis"
    r_analysis_dir.mkdir(parents=True, exist_ok=True)
    (r_analysis_dir / "analysis_model_summaries.txt").write_text(
        "\n".join(
            [
                "analysis_id: self_rate_by_other_rate",
                "analysis_family: core",
                "outcome: self_rate",
                "predictor: other_rate",
                "controls: latency; duration_s; time_within_run; run",
                "random_effects: (1 | subject_id) + (1 | dyad_id)",
                "note: Local speech-rate adaptation",
                "n_obs: 6",
                "n_subjects: 2",
                "n_dyads: 1",
                "",
                "full_model_summary:",
                "Fixed effects:",
                "                  Estimate Std. Error         df t value Pr(>|t|)",
                "(Intercept)        1.50000    0.25000   10.00000   6.000  0.0001",
                "other_rate         0.70000    0.05000   10.00000  14.000  0.0000",
                "latency            0.20000    0.06000   10.00000   3.333  0.0070",
                "duration_s        -0.10000    0.04000   10.00000  -2.500  0.0310",
                "time_within_run    0.05000    0.03000   10.00000   1.667  0.1260",
                "run                0.02000    0.01000   10.00000   2.000  0.0730",
                "---",
                "Correlation of Fixed Effects:",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    variance_partition_group_paths = build_variance_partition_group_output_paths(config)
    variance_partition_group_paths.group_dir.mkdir(parents=True, exist_ok=True)
    write_parquet_dataframe(
        pd.DataFrame(
            {
                "channel": ["Fp1", "Fp1", "Oz", "Oz"],
                "time": [-0.1, 0.0, -0.1, 0.0],
                "channel_index": [0, 0, 1, 1],
                "time_index": [0, 1, 0, 1],
                "unique_self": [0.04, 0.06, 0.03, 0.05],
                "unique_other": [0.03, 0.02, 0.04, 0.03],
                "shared": [0.01, 0.02, 0.05, 0.06],
            }
        ),
        variance_partition_group_paths.table_output_path,
    )

    viz_config_path = tmp_path / "viz.yaml"
    viz_config_path.write_text(
        yaml.safe_dump(
            {
                "viz": {
                    "formats": ["png"],
                    "dpi": 72,
                    "theme": {
                        "colors": {
                            "self": "#8C2D2D",
                            "other": "#1F4E79",
                            "shared": "#2F6B3B",
                            "neutral": "#4B5563",
                            "reference": "#6B7280",
                            "background": "#FFFFFF",
                        }
                    },
                    "figures": {
                        "lmeeeg_main_results": {
                            "significance_threshold": 0.05,
                            "plotting_tmin_s": -0.8,
                            "plotting_tmax_s": 0.0,
                            "models": {
                                "self": {
                                    "label": "Unique self",
                                    "model_name": "self_rate_effect",
                                    "effect_name": "self_rate",
                                    "topomap_times_s": [-0.8, 0.0],
                                },
                                "other": {
                                    "label": "Unique other",
                                    "model_name": "other_rate_effect",
                                    "effect_name": "other_rate",
                                    "topomap_times_s": [-0.5],
                                },
                            },
                        },
                        "similarity_summary": {
                            "enabled": True,
                            "n_topomaps": 4,
                            "window_radius_ms": 50.0,
                            "plotting_tmin_s": -0.8,
                            "plotting_tmax_s": 0.2,
                        },
                        "behavioural": {
                            "speech_rate": {
                                "enabled": True,
                                "min_trials_per_subject": 3,
                                "font_scale": 3.0,
                            },
                        },
                        "variance_partition": {
                            "use_proportions": False,
                            "rois": {
                                "anterior": ["Fp1"],
                                "posterior": ["Oz"],
                            },
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_draw_topomap_with_significant_markers(**kwargs):
        axis = kwargs["axis"]
        axis.set_title(str(kwargs["title"]))
        return axis.imshow(np.zeros((2, 2), dtype=float), vmin=kwargs["vmin"], vmax=kwargs["vmax"], cmap=kwargs["cmap"])

    monkeypatch.setattr(lmeeeg_results_viz, "_draw_topomap_with_significant_markers", fake_draw_topomap_with_significant_markers)
    monkeypatch.setattr(
        lmeeeg_similarity_viz,
        "_draw_similarity_topomap",
        lambda **kwargs: kwargs["axis"].imshow(np.zeros((2, 2), dtype=float), vmin=kwargs["vmin"], vmax=kwargs["vmax"], cmap=kwargs["cmap"]),
    )

    args = Namespace(
        config="config/config.yaml",
        schema="config/schema.json",
        config_section="lmeeeg",
        viz_config=str(viz_config_path),
        output=None,
    )
    exit_code = figures_lmeeeg_main_results.run(args, config=config)

    assert exit_code == 0
    manifest_path = out_dir / "figures" / "main_figures" / "figure_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["plot_count"] == 4
    plot = manifest["plots"][0]
    assert plot["plot_name"] == "lmeeeg_main_results"
    assert plot["significance_threshold"] == 0.05
    assert plot["plotting_tmin_s"] == -0.8
    assert plot["plotting_tmax_s"] == 0.0
    assert [item["model_name"] for item in plot["models"]] == ["self_rate_effect", "other_rate_effect"]
    assert plot["models"][0]["topomap_times_s"] == [-0.8, 0.0]
    assert plot["models"][1]["topomap_times_s"] == [-0.5]
    assert len(plot["files"]) == 1
    assert Path(plot["files"][0]).exists()
    similarity_plot = manifest["plots"][1]
    assert similarity_plot["plot_name"] == "similarity_summary"
    assert all(Path(path).parent == out_dir / "figures" / "main_figures" for path in map(Path, similarity_plot["files"]))
    assert all(Path(path).exists() for path in similarity_plot["files"])
    behavioural_plot = manifest["plots"][2]
    assert behavioural_plot["plot_name"] == "behavioural_summary"
    assert all(Path(path).parent == out_dir / "figures" / "main_figures" for path in map(Path, behavioural_plot["files"]))
    assert all(Path(path).exists() for path in behavioural_plot["files"])
    variance_partition_plot = manifest["plots"][3]
    assert variance_partition_plot["plot_name"] == "variance_partition_summary"
    assert all(Path(path).parent == out_dir / "figures" / "main_figures" for path in map(Path, variance_partition_plot["files"]))
    assert all(Path(path).exists() for path in variance_partition_plot["files"])


def test_lmeeeg_group_stats_cluster_threshold_interprets_alpha_values() -> None:
    from scipy.stats import t

    threshold = lmeeeg_group_stats._resolve_threshold(
        {"stats": {"cluster_threshold": 0.01}},
        n_observations=32,
        correction="cluster",
        tail=0,
    )

    assert threshold == pytest.approx(float(t.ppf(1.0 - (0.01 / 2.0), df=31)))


def test_lmeeeg_group_stats_cluster_threshold_preserves_statistic_cutoffs() -> None:
    threshold = lmeeeg_group_stats._resolve_threshold(
        {"stats": {"cluster_threshold": 2.5}},
        n_observations=32,
        correction="cluster",
        tail=0,
    )

    assert threshold == pytest.approx(2.5)
