"""`figures-behavioural` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import yaml

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.paths import resolve_out_dir
from rate.viz.behavioural import build_behavioural_manifest

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser("figures-behavioural", help="Render behavioural summary figures.")
    add_common_config_arguments(parser)
    parser.add_argument("--viz-config", default="config/viz.yaml", help="Path to the figure config file.")
    parser.add_argument("--input", help="Optional explicit behavioural residuals CSV path.")
    parser.add_argument("--slopes-input", help="Optional explicit behavioural subject slopes CSV path.")
    parser.add_argument("--output", help="Optional explicit output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    out_dir = resolve_out_dir(config)
    residuals_path = Path(args.input) if args.input else out_dir / "stats" / "behavioural_residuals" / "trialwise_residuals.csv"
    slopes_path = Path(args.slopes_input) if args.slopes_input else out_dir / "stats" / "behavioural_residuals" / "subject_slopes.csv"
    output_path = Path(args.output) if args.output else out_dir / "figures" / "behavioural" / "figure_manifest.json"
    viz_cfg = _load_viz_config(Path(args.viz_config))
    viz_section = dict(viz_cfg.get("viz") or {})
    behavioural_viz_section = dict(((viz_section.get("figures") or {}).get("behavioural")) or {})
    theme_section = dict(viz_section.get("theme") or {})

    manifest = build_behavioural_manifest(
        out_dir=out_dir,
        residuals_path=residuals_path,
        subject_slopes_path=slopes_path,
        manifest_path=output_path,
        figure_config=behavioural_viz_section,
        theme_config=theme_section,
        formats=tuple(viz_section.get("formats", ["png", "pdf"])),
        dpi=int(viz_section.get("dpi", 300)),
    )
    LOGGER.info("Wrote behavioural figure manifest with %d plots: %s", int(manifest["plot_count"]), output_path)
    return 0


def _load_viz_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Viz config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Viz config must be a mapping: {path}")
    return payload
