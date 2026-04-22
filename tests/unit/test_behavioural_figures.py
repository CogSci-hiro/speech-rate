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
            "other_rate": [3.8, 4.0, 4.2, 3.0, 3.1, 3.4, 4.6, 4.8, 4.9],
            "latency": [0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.4, 0.5],
            "duration_s": [0.9, 1.0, 1.1, 1.0, 1.1, 0.9, 1.2, 1.1, 1.0],
            "time_within_run": [0.1, 0.2, 0.3, 0.1, 0.3, 0.5, 0.2, 0.4, 0.6],
            "run": [1, 1, 1, 1, 1, 1, 2, 2, 2],
        }
    )
    dataframe.to_csv(path, index=False)
    return path


def _write_lmm_summary_txt(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "analysis_id: self_rate_by_other_rate",
                "analysis_family: core",
                "outcome: self_rate",
                "predictor: other_rate",
                "controls: latency; duration_s; time_within_run; run",
                "random_effects: (1 | subject_id) + (1 | dyad_id)",
                "note: Local speech-rate adaptation",
                "n_obs: 9",
                "n_subjects: 3",
                "n_dyads: 2",
                "",
                "full_model_summary:",
                "Linear mixed model fit by maximum likelihood . t-tests use Satterthwaite's",
                "  method [lmerModLmerTest]",
                "",
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
    return path


def _write_random_effects_csv(path: Path) -> Path:
    dataframe = pd.DataFrame(
        {
            "grouping_factor": ["subject_id", "subject_id", "subject_id"],
            "level_id": ["sub-001", "sub-002", "sub-003"],
            "subject_id": ["sub-001", "sub-002", "sub-003"],
            "dyad_id": ["dyad-001", "dyad-001", "dyad-002"],
            "random_intercept": [0.15, -0.10, 0.05],
            "se": [0.03, 0.04, 0.02],
        }
    )
    dataframe.to_csv(path, index=False)
    return path


def test_build_behavioural_manifest_writes_combined_plot(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    residuals_path = _write_residuals_csv(tmp_path / "trialwise_residuals.csv")
    random_effects_path = _write_random_effects_csv(tmp_path / "random_effects.csv")
    lmm_summary_path = _write_lmm_summary_txt(tmp_path / "analysis_model_summaries.txt")
    manifest_path = out_dir / "figures" / "behavioural" / "figure_manifest.json"

    manifest = build_behavioural_manifest(
        out_dir=out_dir,
        residuals_path=residuals_path,
        random_effects_path=random_effects_path,
        subject_slopes_path=None,
        lmm_summary_path=lmm_summary_path,
        manifest_path=manifest_path,
        figure_config={"speech_rate": {"formats": ["png"], "min_trials_per_subject": 3, "random_seed": 1}},
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
    lmm_summary_path = _write_lmm_summary_txt(tmp_path / "analysis_model_summaries.txt")
    manifest_path = out_dir / "figures" / "behavioural" / "figure_manifest.json"

    try:
        build_behavioural_manifest(
            out_dir=out_dir,
            residuals_path=residuals_path,
            random_effects_path=None,
            subject_slopes_path=None,
            lmm_summary_path=lmm_summary_path,
            manifest_path=manifest_path,
            figure_config={"speech_rate": {"formats": ["png"], "min_trials_per_subject": 3}},
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
    r_analysis_dir = out_dir / "stats" / "r_analysis"
    residuals_dir.mkdir(parents=True, exist_ok=True)
    r_analysis_dir.mkdir(parents=True, exist_ok=True)
    residuals_path = _write_residuals_csv(residuals_dir / "trialwise_residuals.csv")
    random_effects_path = _write_random_effects_csv(residuals_dir / "random_effects.csv")
    lmm_summary_path = _write_lmm_summary_txt(r_analysis_dir / "analysis_model_summaries.txt")
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
        random_effects_input=str(random_effects_path),
        lmm_summary_input=str(lmm_summary_path),
        output=None,
    )
    exit_code = figures_behavioural.run(args, config={"paths": {"out_dir": str(out_dir)}})

    assert exit_code == 0
    manifest_path = out_dir / "figures" / "behavioural" / "figure_manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["plot_count"] == 1
    assert manifest["plots"][0]["plot_name"] == "behavioural_summary"


def test_build_behavioural_manifest_supports_optional_marginalised_scatter(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    residuals_path = _write_residuals_csv(tmp_path / "trialwise_residuals.csv")
    random_effects_path = _write_random_effects_csv(tmp_path / "random_effects.csv")
    lmm_summary_path = _write_lmm_summary_txt(tmp_path / "analysis_model_summaries.txt")
    manifest_path = out_dir / "figures" / "behavioural" / "figure_manifest.json"

    manifest = build_behavioural_manifest(
        out_dir=out_dir,
        residuals_path=residuals_path,
        random_effects_path=random_effects_path,
        subject_slopes_path=None,
        lmm_summary_path=lmm_summary_path,
        manifest_path=manifest_path,
        figure_config={
            "speech_rate": {
                "formats": ["png"],
                "min_trials_per_subject": 3,
                "random_seed": 1,
                "show_marginalised_scatter": True,
            }
        },
        formats=("png",),
        dpi=72,
    )

    assert manifest["plot_count"] == 1
    assert Path(manifest["plots"][0]["files"][0]).exists()
