from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from rate.legacy.cli.commands import model_weights_glm


def test_model_weights_glm_cli_writes_joint_plot_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    times = np.linspace(-0.1, 0.3, 25)
    channel_names = ["Fz", "Cz", "Pz", "Oz"]
    for subject_id, phase in [("sub-001", 0.0), ("sub-002", 0.4)]:
        model_dir = out_dir / "glm" / subject_id / "subject_level"
        model_dir.mkdir(parents=True, exist_ok=True)
        beta_map = np.vstack(
            [
                np.sin(np.linspace(0.0 + phase, np.pi + phase, 25)),
                np.cos(np.linspace(0.0 + phase, np.pi + phase, 25)),
                np.sin(np.linspace(0.0 + phase, 2.0 * np.pi + phase, 25)),
                np.cos(np.linspace(0.0 + phase, 2.0 * np.pi + phase, 25)),
            ]
        )
        np.save(model_dir / "times.npy", times)
        np.savez(model_dir / "predictor_betas.npz", speech_rate=beta_map)
        (model_dir / "channel_names.json").write_text(json.dumps(channel_names) + "\n", encoding="utf-8")
        (model_dir / "summary.json").write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")

    config = {"paths": {"out_dir": str(out_dir)}}
    viz_config_path = tmp_path / "viz.yaml"
    viz_config_path.write_text("viz:\n  formats: [\"png\"]\n  dpi: 100\n", encoding="utf-8")
    manifest_path = out_dir / "figures" / "glm" / "model_weights" / "manifest.json"

    args = Namespace(
        config="config/config.yaml",
        schema="config/schema.json",
        viz_config=str(viz_config_path),
        output=str(manifest_path),
    )

    exit_code = model_weights_glm.run(args, config=config)

    assert exit_code == 0
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["subject_plot_count"] == 2
    assert manifest["group_plot_count"] == 1
    assert manifest["plot_count"] == 3
    subject_plots = [plot for plot in manifest["plots"] if plot["level"] == "subject"]
    group_plots = [plot for plot in manifest["plots"] if plot["level"] == "group"]
    assert len(subject_plots) == 2
    assert len(group_plots) == 1
    assert Path(group_plots[0]["files"][0]).exists()
    assert group_plots[0]["n_subjects"] == 2


def test_model_weights_glm_cli_handles_overlapping_channel_positions(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    model_dir = out_dir / "glm" / "sub-001" / "subject_level"
    model_dir.mkdir(parents=True, exist_ok=True)

    times = np.linspace(-0.1, 0.3, 25)
    beta_map = np.vstack(
        [
            np.sin(np.linspace(0.0, np.pi, 25)),
            np.cos(np.linspace(0.0, np.pi, 25)),
            np.sin(np.linspace(0.0, 2.0 * np.pi, 25)),
            np.cos(np.linspace(0.0, 2.0 * np.pi, 25)),
        ]
    )
    np.save(model_dir / "times.npy", times)
    np.savez(model_dir / "predictor_betas.npz", speech_rate=beta_map)
    (model_dir / "channel_names.json").write_text(json.dumps(["Fz", "Cz", "rEAR", "lEAR"]) + "\n", encoding="utf-8")
    (model_dir / "summary.json").write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")

    config = {"paths": {"out_dir": str(out_dir)}}
    viz_config_path = tmp_path / "viz.yaml"
    viz_config_path.write_text("viz:\n  formats: [\"png\"]\n  dpi: 100\n", encoding="utf-8")
    manifest_path = out_dir / "figures" / "glm" / "model_weights" / "manifest.json"

    args = Namespace(
        config="config/config.yaml",
        schema="config/schema.json",
        viz_config=str(viz_config_path),
        output=str(manifest_path),
    )

    exit_code = model_weights_glm.run(args, config=config)

    assert exit_code == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["subject_plot_count"] == 1
    assert manifest["group_plot_count"] == 1
    plot_path = Path(manifest["plots"][0]["files"][0])
    assert plot_path.exists()
