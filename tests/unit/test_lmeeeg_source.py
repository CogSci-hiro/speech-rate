"""Unit tests for the experimental source-space lmeEEG branch."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import mne
import numpy as np

from rate.lmeeeg_source.config import load_lmeeeg_source_settings
from rate.lmeeeg_source.plot import _resolve_significance_mask, build_lmeeeg_source_qc_figure
from rate.lmeeeg_source.source_space import build_source_channel_names


def _source_config(tmp_path: Path) -> dict:
    return {
        "paths": {"out_dir": str(tmp_path / "out"), "ica_dir": str(tmp_path / "ica")},
        "dataset": {},
        "preprocessing": {},
        "epochs": {},
        "glm": {},
        "lmeeeg": {},
        "lmeeeg_source": {
            "enabled": True,
            "source_space": {
                "subjects_dir": str(tmp_path / "subjects"),
                "fsaverage": "fsaverage",
                "spacing": "ico2",
                "surface": "white",
                "add_dist": False,
                "subject_template": "{subject_id}",
            },
            "forward": {
                "trans_pattern": str(tmp_path / "transforms" / "{subject_id}-trans.fif"),
                "bem_solution_pattern": str(tmp_path / "bem" / "{fs_subject}-bem-sol.fif"),
            },
            "covariance": {"method": "empirical", "tmin": None, "tmax": 0.0},
            "inverse": {"method": "MNE", "loose": 0.2, "depth": 0.8, "snr": 3.0, "pick_ori": "normal"},
            "fitting": {"n_jobs": 1, "batch_size_vertices": 25, "verbose": True, "progress": False},
            "outputs": {"namespace": "lmeeeg_source", "prepared_dirname": "prepared", "group_dirname": "group", "qc_subdir": "qc/lmeeeg_source"},
            "plotting": {
                "predictors": ["self_rate", "other_rate"],
                "view": "lateral",
                "hemi": "split",
                "measure": "beta",
                "time_summary": "mean",
                "time_step_ms": 100.0,
                "mask_by_significance": True,
                "significance_alpha": 0.05,
                "formats": ["png", "pdf"],
                "colormap": "icefire",
            },
            "epochs": {"section": "margin_epochs", "use_existing_margin_epochs": True},
            "models": {
                "self_rate_effect": {"formula": "~ self_rate", "test_predictors": ["self_rate"]},
                "other_rate_effect": {"formula": "~ other_rate", "test_predictors": ["other_rate"]},
            },
        },
    }


def test_load_lmeeeg_source_settings_derives_lambda2(tmp_path: Path) -> None:
    config = _source_config(tmp_path)
    settings = load_lmeeeg_source_settings(config)
    assert settings.source_space.spacing == "ico2"
    assert settings.inverse.lambda2 == 1.0 / 9.0
    assert settings.epochs.section == "margin_epochs"


def test_build_source_channel_names_labels_hemispheres() -> None:
    channel_names = build_source_channel_names(
        [
            np.asarray([1, 12], dtype=int),
            np.asarray([7], dtype=int),
        ]
    )
    assert channel_names == ["lh_000001", "lh_000012", "rh_000007"]


def test_build_lmeeeg_source_qc_figure_falls_back_when_brain_rendering_fails(tmp_path: Path, monkeypatch) -> None:
    config = _source_config(tmp_path)
    out_dir = Path(config["paths"]["out_dir"])
    subjects_dir = Path(config["lmeeeg_source"]["source_space"]["subjects_dir"])
    (subjects_dir / "fsaverage").mkdir(parents=True, exist_ok=True)

    vertices = [np.asarray([0, 1], dtype=int), np.asarray([2, 3], dtype=int)]
    times = np.asarray([0.0, 1.0], dtype=float)
    for predictor_name, scale in (("self_rate", 1.0), ("other_rate", -1.0)):
        group_dir = out_dir / "lmeeeg_source" / "group" / predictor_name
        group_dir.mkdir(parents=True, exist_ok=True)
        beta_full = mne.SourceEstimate(
            np.asarray(
                [
                    [1.0 * scale, 1.0 * scale],
                    [0.5 * scale, 0.5 * scale],
                    [-0.25 * scale, -0.25 * scale],
                    [-0.75 * scale, -0.75 * scale],
                ],
                dtype=float,
            ),
            vertices=vertices,
            tmin=0.0,
            tstep=1.0,
        )
        t_value_full = mne.SourceEstimate(
            np.asarray(
                [
                    [2.5, 2.5],
                    [2.1, 2.1],
                    [0.5, 0.5],
                    [0.2, 0.2],
                ],
                dtype=float,
            ),
            vertices=vertices,
            tmin=0.0,
            tstep=1.0,
        )
        corrected_p_values = np.asarray(
            [
                [0.01, 0.01],
                [0.03, 0.03],
                [0.2, 0.2],
                [1.0, 1.0],
            ],
            dtype=float,
        )
        beta_full.save(str(group_dir / "beta_full"), ftype="stc", overwrite=True)
        t_value_full.save(str(group_dir / "t_value_full"), ftype="stc", overwrite=True)
        np.save(group_dir / "corrected_p_values.npy", corrected_p_values)
        summary = {
            "status": "ok",
            "predictor": predictor_name,
            "model_name": f"{predictor_name}_effect",
            "measure": "beta",
            "full_stc_stem": str(group_dir / "beta_full"),
            "summary_stc_stem": str(group_dir / "beta_full"),
            "beta_full_stc_stem": str(group_dir / "beta_full"),
            "beta_summary_stc_stem": str(group_dir / "beta_full"),
            "t_value_full_stc_stem": str(group_dir / "t_value_full"),
            "t_value_summary_stc_stem": str(group_dir / "t_value_full"),
            "corrected_p_values": str(group_dir / "corrected_p_values.npy"),
            "vertices_path": str(group_dir / "vertices.json"),
            "time_summary": "mean",
            "time_window": None,
        }
        (group_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "rate.lmeeeg_source.plot._render_lateral_brain_panel",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic headless failure")),
    )

    manifest = build_lmeeeg_source_qc_figure(config)
    assert manifest["status"] == "ok"
    assert manifest["plot_count"] == 1
    assert (out_dir / "figures" / "qc" / "lmeeeg_source" / "self_other_lateral.png").exists()
    assert (out_dir / "figures" / "qc" / "lmeeeg_source" / "self_other_lateral.pdf").exists()
    assert (out_dir / "figures" / "lmeeeg_source" / "figure_manifest.json").exists()


def test_build_lmeeeg_source_qc_figure_can_disable_significance_masking(tmp_path: Path, monkeypatch) -> None:
    config = _source_config(tmp_path)
    config["lmeeeg_source"]["plotting"]["mask_by_significance"] = False
    out_dir = Path(config["paths"]["out_dir"])
    subjects_dir = Path(config["lmeeeg_source"]["source_space"]["subjects_dir"])
    (subjects_dir / "fsaverage").mkdir(parents=True, exist_ok=True)

    vertices = [np.asarray([0, 1], dtype=int), np.asarray([2, 3], dtype=int)]
    for predictor_name, scale in (("self_rate", 1.0), ("other_rate", -1.0)):
        group_dir = out_dir / "lmeeeg_source" / "group" / predictor_name
        group_dir.mkdir(parents=True, exist_ok=True)
        beta_full = mne.SourceEstimate(
            np.asarray(
                [
                    [1.0 * scale, 0.8 * scale],
                    [0.5 * scale, 0.4 * scale],
                    [-0.25 * scale, -0.2 * scale],
                    [-0.75 * scale, -0.6 * scale],
                ],
                dtype=float,
            ),
            vertices=vertices,
            tmin=0.0,
            tstep=0.1,
        )
        t_value_full = mne.SourceEstimate(
            np.asarray(
                [
                    [2.5, 2.5],
                    [2.1, 2.1],
                    [0.5, 0.5],
                    [0.2, 0.2],
                ],
                dtype=float,
            ),
            vertices=vertices,
            tmin=0.0,
            tstep=0.1,
        )
        corrected_p_values = np.asarray(
            [
                [0.01, 0.01],
                [0.03, 0.03],
                [0.2, 0.2],
                [1.0, 1.0],
            ],
            dtype=float,
        )
        beta_full.save(str(group_dir / "beta_full"), ftype="stc", overwrite=True)
        t_value_full.save(str(group_dir / "t_value_full"), ftype="stc", overwrite=True)
        np.save(group_dir / "corrected_p_values.npy", corrected_p_values)
        summary = {
            "status": "ok",
            "predictor": predictor_name,
            "model_name": f"{predictor_name}_effect",
            "measure": "beta",
            "full_stc_stem": str(group_dir / "beta_full"),
            "summary_stc_stem": str(group_dir / "beta_full"),
            "beta_full_stc_stem": str(group_dir / "beta_full"),
            "beta_summary_stc_stem": str(group_dir / "beta_full"),
            "t_value_full_stc_stem": str(group_dir / "t_value_full"),
            "t_value_summary_stc_stem": str(group_dir / "t_value_full"),
            "corrected_p_values": str(group_dir / "corrected_p_values.npy"),
            "vertices_path": str(group_dir / "vertices.json"),
            "time_summary": "mean",
            "time_window": None,
        }
        (group_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    captured_titles: list[str] = []

    def _record_title(_figure, title: str, *args, **kwargs) -> None:
        captured_titles.append(title)

    monkeypatch.setattr(
        "rate.lmeeeg_source.plot._render_lateral_brain_panel",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic headless failure")),
    )
    monkeypatch.setattr("matplotlib.figure.Figure.suptitle", _record_title)

    manifest = build_lmeeeg_source_qc_figure(config)
    assert manifest["plots"][0]["mask_by_significance"] is False
    assert any("unmasked effect maps" in title for title in captured_titles)


def test_resolve_significance_mask_uses_cluster_membership_for_cluster_correction() -> None:
    corrected_p_values = np.asarray(
        [
            [0.9, 0.9],
            [0.9, 0.9],
            [0.9, 0.9],
        ],
        dtype=float,
    )
    cluster_p_values = np.asarray([0.01, 0.2], dtype=float)
    cluster_masks = np.asarray(
        [
            [
                [True, False],
                [False, True],
                [False, False],
            ],
            [
                [False, False],
                [False, False],
                [True, True],
            ],
        ],
        dtype=bool,
    )
    t_value_data = np.zeros((3, 2), dtype=float)

    time0_mask = _resolve_significance_mask(
        correction="cluster",
        corrected_p_values=corrected_p_values,
        cluster_p_values=cluster_p_values,
        cluster_masks=cluster_masks,
        t_value_data=t_value_data,
        time_index=0,
        significance_alpha=0.05,
        t_threshold=1.96,
        predictor_name="self_rate",
    )
    time1_mask = _resolve_significance_mask(
        correction="cluster",
        corrected_p_values=corrected_p_values,
        cluster_p_values=cluster_p_values,
        cluster_masks=cluster_masks,
        t_value_data=t_value_data,
        time_index=1,
        significance_alpha=0.05,
        t_threshold=1.96,
        predictor_name="self_rate",
    )

    assert np.array_equal(time0_mask, np.asarray([True, False, False], dtype=bool))
    assert np.array_equal(time1_mask, np.asarray([False, True, False], dtype=bool))
