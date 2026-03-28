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
import numpy as np
import pandas as pd
import yaml

from rate.cli.commands import lmeeeg_group_stats
from rate.cli.commands import lmeeeg_subject
from rate.cli.commands import figures_lmeeeg
from rate.epochs.io import build_epoch_output_paths
from rate.lmeeeg.io import build_lmeeeg_output_paths
from rate.lmeeeg.mass_model import fit_lmeeeg_mass_model
from rate.lmeeeg.trial_data import build_lmeeeg_trial_data_from_arrays, build_lmeeeg_trial_table
from rate.util import write_parquet_dataframe
from rate.viz import lmeeeg as lmeeeg_viz


def _make_epochs(data: np.ndarray, metadata: pd.DataFrame) -> mne.EpochsArray:
    info = mne.create_info(["Fz", "Cz"], sfreq=100.0, ch_types=["eeg", "eeg"])
    events = np.column_stack([np.arange(len(data), dtype=int), np.zeros(len(data), dtype=int), np.ones(len(data), dtype=int)])
    return mne.EpochsArray(data, info, events=events, tmin=-0.1, metadata=metadata, verbose="ERROR")


def _lmeeeg_config(out_dir: Path) -> dict:
    return {
        "paths": {"out_dir": str(out_dir)},
        "epochs": {},
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


def test_build_lmeeeg_trial_data_backfills_proximity_from_alignment() -> None:
    config = {
        "lmeeeg": {
            "models": {
                "proximity_effect": {
                    "formula": "~ proximity + self_rate",
                    "eligibility": {"not_null": ["proximity", "self_rate"]},
                    "test_predictors": ["proximity"],
                    "standardize": ["proximity", "self_rate"],
                    "random_effect_group": "subject_id",
                    "random_effect_structure": "random_intercept",
                }
            }
        }
    }
    eeg_data = np.arange(4 * 2 * 3, dtype=float).reshape(4, 2, 3)
    metadata = pd.DataFrame(
        {
            "alignment": [-0.5, -0.25, -0.1, -0.05],
            "self_rate": [3.0, 3.5, 4.0, 4.5],
            "subject_id": ["sub-001", "sub-001", "sub-002", "sub-002"],
        }
    )

    trial_data = build_lmeeeg_trial_data_from_arrays(
        eeg_data=eeg_data,
        metadata_df=metadata,
        channel_names=["Fz", "Cz"],
        times=np.asarray([0.0, 0.1, 0.2], dtype=float),
        config=config,
        model_name="proximity_effect",
    )

    assert "proximity" in trial_data.trial_metadata.columns
    assert np.allclose(
        trial_data.trial_metadata["proximity"].to_numpy(dtype=float),
        metadata["alignment"].to_numpy(dtype=float),
    )
    assert trial_data.design_result.test_predictors == ["proximity"]


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
        channel_names=["Fz", "Cz"],
        times=np.asarray([0.0, 0.1, 0.2], dtype=float),
    )

    csv_path = Path(summary["p_values_csv"])
    csv_df = pd.read_csv(csv_path)

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
        channel_names=["Fz", "Cz"],
        times=np.asarray([0.0, 0.1], dtype=float),
    )

    csv_path = Path(summary["p_values_csv"])
    csv_df = pd.read_csv(csv_path)

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
    stats_effect_dir = out_dir / "stats" / "lmeeeg" / "convergence_subject_intercept" / "latency"
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
        lambda config, rows: {
            "convergence_subject_intercept": {
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
        out_dir / "figures" / "lmeeeg" / "betas" / "convergence_subject_intercept" / "latency",
        out_dir / "figures" / "lmeeeg" / "t_values" / "convergence_subject_intercept" / "latency",
        out_dir / "figures" / "lmeeeg" / "statistics" / "convergence_subject_intercept" / "latency",
    ]
