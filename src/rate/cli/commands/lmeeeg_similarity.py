"""`lmeeeg-similarity` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import yaml

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.lmeeeg.similarity import build_similarity_output_paths
from rate.paths import resolve_out_dir
from rate.lmeeeg.similarity import run_similarity_analysis
from rate.viz.lmeeeg_similarity import build_similarity_summary_figure_manifest

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "lmeeeg-similarity",
        help="Compare two existing pooled lmeEEG predictor maps with similarity and difference metrics.",
    )
    add_common_config_arguments(parser)
    parser.add_argument("--viz-config", default="config/viz.yaml", help="Path to the figure config file.")
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Write similarity statistics only and skip figure rendering.",
    )
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    try:
        summary = run_similarity_analysis(config)
        if not bool(getattr(args, "skip_figures", False)):
            out_dir = resolve_out_dir(config)
            viz_cfg = _load_viz_config(Path(getattr(args, "viz_config", "config/viz.yaml") or "config/viz.yaml"))
            viz_section = dict(viz_cfg.get("viz") or {})
            figure_section = dict(((viz_section.get("figures") or {}).get("similarity_summary")) or {})
            theme_section = dict(viz_section.get("theme") or {})
            if bool(figure_section.get("enabled", True)):
                output_paths = build_similarity_output_paths(config)
                figure_manifest = build_similarity_summary_figure_manifest(
                    config=config,
                    out_dir=out_dir,
                    manifest_path=output_paths.figure_manifest_output_path,
                    figure_config=figure_section,
                    theme_config=theme_section,
                    formats=tuple(viz_section.get("formats", ["png", "pdf"])),
                    dpi=int(viz_section.get("dpi", 300)),
                )
                LOGGER.info(
                    "Wrote similarity summary figure manifest with %d plots: %s",
                    int(figure_manifest["plot_count"]),
                    output_paths.figure_manifest_output_path,
                )
        LOGGER.info("Wrote lmeEEG similarity outputs: %s", summary["summary_path"])
        return 0
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1


def _load_viz_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        LOGGER.warning("Viz config not found, using default similarity figure settings: %s", path)
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Viz config must be a mapping: {path}")
    return payload
