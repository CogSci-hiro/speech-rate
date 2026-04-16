from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import yaml

from rate.cli.commands import figures_behavioural
from rate.viz.behavioural import build_behavioural_manifest


def _write_residuals_csv(path: Path) -> Path:
    dataframe = pd.DataFrame(
        {
            "subject_id": [
                "sub-001",
                "sub-001",
                "sub-001",
                "sub-002",
                "sub-002",
                "sub-002",
                "sub-003",
                "sub-003",
                "sub-003",
            ],
            "dyad_id": [
                "dyad-001",
                "dyad-001",
                "dyad-001",
                "dyad-001",
                "dyad-001",
                "dyad-001",
                "dyad-002",
                "dyad-002",
                "dyad-002",
            ],
            "self_rate": [4.1, 4.4, 4.6, 3.3, 3.5, 3.7, 5.0, 5.2, 5.3],
        }
    )
    dataframe.to_csv(path, index=False)
    return path


def _write_subject_slopes_csv(path: Path) -> Path:
    dataframe = pd.DataFrame(
        {
            "subject_id": ["sub-001", "sub-002", "sub-003", "sub-004"],
            "dyad_id": ["dyad-001", "dyad-001", "dyad-002", "dyad-002"],
            "slope": [-0.15, 0.05, 0.20, 0.35],
            "se": [0.04, 0.05, 0.06, 0.05],
        }
    )
    dataframe.to_csv(path, index=False)
    return path


def test_build_behavioural_manifest_writes_combined_plot(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    residuals_path = _write_residuals_csv(tmp_path / "trialwise_residuals.csv")
    subject_slopes_path = _write_subject_slopes_csv(tmp_path / "subject_slopes.csv")
    manifest_path = out_dir / "figures" / "behavioural" / "figure_manifest.json"

    manifest = build_behavioural_manifest(
        out_dir=out_dir,
        residuals_path=residuals_path,
        subject_slopes_path=subject_slopes_path,
        manifest_path=manifest_path,
        figure_config={
            "speech_rate": {"formats": ["png"], "min_trials_per_subject": 3, "random_seed": 1},
            "subject_slopes": {"formats": ["png"]},
        },
        formats=("png",),
        dpi=72,
    )

    assert manifest["status"] == "ok"
    assert manifest["plot_count"] == 1
    assert manifest_path.exists()

    plot = manifest["plots"][0]
    assert plot["plot_name"] == "behavioural_summary"
    assert plot["n_rows"] == 9
    assert plot["n_subjects"] == 3
    assert plot["n_dyads"] == 2
    assert plot["ordered_dyads"] == ["dyad-001", "dyad-002"]
    assert len(plot["files"]) == 1
    assert Path(plot["files"][0]).exists()


def test_speech_rate_plot_requires_dyad_column_in_residuals(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    residuals_path = tmp_path / "trialwise_residuals.csv"
    pd.DataFrame(
        {
            "subject_id": ["sub-001", "sub-001", "sub-001"],
            "self_rate": [4.1, 4.4, 4.6],
        }
    ).to_csv(residuals_path, index=False)
    subject_slopes_path = _write_subject_slopes_csv(tmp_path / "subject_slopes.csv")
    manifest_path = out_dir / "figures" / "behavioural" / "figure_manifest.json"

    try:
        build_behavioural_manifest(
            out_dir=out_dir,
            residuals_path=residuals_path,
            subject_slopes_path=subject_slopes_path,
            manifest_path=manifest_path,
            figure_config={
                "speech_rate": {"formats": ["png"], "min_trials_per_subject": 3},
                "subject_slopes": {"enabled": False},
            },
            formats=("png",),
            dpi=72,
        )
    except ValueError as exc:
        assert "dyad_id" in str(exc)
    else:
        raise AssertionError("Expected a missing-column error when dyad_id is absent.")


def test_figures_behavioural_cli_writes_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    residuals_dir = out_dir / "stats" / "behavioural_residuals"
    residuals_dir.mkdir(parents=True, exist_ok=True)
    residuals_path = _write_residuals_csv(residuals_dir / "trialwise_residuals.csv")
    slopes_path = _write_subject_slopes_csv(residuals_dir / "subject_slopes.csv")
    viz_config_path = tmp_path / "viz.yaml"
    viz_config_path.write_text(
        yaml.safe_dump(
            {
                "viz": {
                    "formats": ["png"],
                    "dpi": 72,
                    "figures": {
                        "behavioural": {
                            "speech_rate": {
                                "enabled": True,
                                "min_trials_per_subject": 3,
                            },
                            "subject_slopes": {
                                "enabled": True,
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
        input=str(residuals_path),
        slopes_input=str(slopes_path),
        output=None,
    )
    exit_code = figures_behavioural.run(args, config={"paths": {"out_dir": str(out_dir)}})

    assert exit_code == 0
    manifest_path = out_dir / "figures" / "behavioural" / "figure_manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["plot_count"] == 1
    assert manifest["plots"][0]["plot_name"] == "behavioural_summary"
