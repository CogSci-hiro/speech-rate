"""Unit tests for group-level GLM beta inference."""

from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import numpy as np
import yaml

from rate.legacy.cli.commands import glm_group_stats


def _glm_config(out_dir: Path) -> dict:
    return {
        "paths": {"out_dir": str(out_dir)},
        "glm": {
            "models": {
                "subject_level": {
                    "formula": "~ speech_rate",
                    "test_predictors": ["speech_rate"],
                }
            }
        },
    }


def test_glm_group_stats_cli_writes_real_inference_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = _glm_config(out_dir)
    times = np.array([-0.1, 0.0, 0.1], dtype=float)
    channel_names = ["Fz", "Cz"]

    beta_maps = {
        "sub-001": np.array([[1.5, 1.2, 0.9], [0.1, 0.0, -0.1]], dtype=float),
        "sub-002": np.array([[1.4, 1.1, 0.8], [0.0, -0.1, 0.0]], dtype=float),
        "sub-003": np.array([[1.6, 1.3, 1.0], [0.2, 0.0, -0.1]], dtype=float),
        "sub-004": np.array([[1.3, 1.0, 0.7], [0.1, -0.1, 0.0]], dtype=float),
    }
    for subject_id, beta_map in beta_maps.items():
        model_dir = out_dir / "glm" / subject_id / "subject_level"
        model_dir.mkdir(parents=True, exist_ok=True)
        np.save(model_dir / "times.npy", times)
        np.savez(model_dir / "predictor_betas.npz", speech_rate=beta_map)
        (model_dir / "channel_names.json").write_text(json.dumps(channel_names) + "\n", encoding="utf-8")
        (model_dir / "predictor_columns.json").write_text(
            json.dumps({"speech_rate": ["speech_rate"]}, indent=2) + "\n",
            encoding="utf-8",
        )
        (model_dir / "summary.json").write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")

    stats_cfg_path = tmp_path / "glm_stats.yaml"
    stats_cfg_path.write_text(
        yaml.safe_dump(
            {
                "stats": {
                    "method": "tfce",
                    "n_permutations": 8,
                    "tail": 0,
                    "adjacency": None,
                    "tfce_threshold": {"start": 0.0, "step": 0.2},
                    "outputs": {"glm": "stats/glm_group_stats.json"},
                }
            }
        ),
        encoding="utf-8",
    )

    args = Namespace(
        config="config/config.yaml",
        schema="config/schema.json",
        stats_config=str(stats_cfg_path),
        model_name="subject_level",
        output=None,
    )
    exit_code = glm_group_stats.run(args, config=config)
    assert exit_code == 0

    summary_path = out_dir / "stats" / "glm_group_stats.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["status"] == "ok"
    assert summary["method"] == "tfce"
    assert summary["n_analyses"] == 1
    effect_summary = summary["analyses"][0]["effects"][0]
    assert effect_summary["predictor"] == "speech_rate"
    assert effect_summary["effect"] == "speech_rate"
    assert effect_summary["n_subjects"] == 4
    assert set(effect_summary["subject_ids"]) == set(beta_maps)
    assert Path(effect_summary["observed_statistic"]).exists()
    assert Path(effect_summary["corrected_p_values"]).exists()
    assert np.load(effect_summary["observed_statistic"]).shape == (2, 3)
    assert np.load(effect_summary["corrected_p_values"]).shape == (2, 3)


def test_glm_group_stats_resolve_adjacency_uses_configured_preprocessing_montage(monkeypatch, tmp_path: Path) -> None:
    import mne

    captured: dict[str, str] = {}

    original_make_standard_montage = mne.channels.make_standard_montage

    def fake_make_standard_montage(name: str):
        captured["montage_name"] = name
        return original_make_standard_montage("standard_1020")

    def fake_find_ch_adjacency(info, ch_type):
        return np.eye(len(info.ch_names), dtype=float), info.ch_names

    monkeypatch.setattr(mne.channels, "make_standard_montage", fake_make_standard_montage)
    monkeypatch.setattr(mne.channels, "find_ch_adjacency", fake_find_ch_adjacency)

    out_dir = tmp_path / "out"
    model_dir = out_dir / "glm" / "sub-001" / "subject_level"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "channel_names.json").write_text(json.dumps(["Fz", "Cz"]) + "\n", encoding="utf-8")

    config = {
        "paths": {"out_dir": str(out_dir)},
        "preprocessing": {
            "montage": {
                "name": "biosemi64",
            }
        },
    }
    stats_cfg = {"stats": {"adjacency": "mne_default"}}
    subject_effects = {
        "speech_rate": {
            "beta_maps": np.ones((2, 2, 3), dtype=float),
        }
    }

    adjacency = glm_group_stats._resolve_adjacency(config, stats_cfg, subject_effects)

    assert captured["montage_name"] == "biosemi64"
    assert adjacency is not None
