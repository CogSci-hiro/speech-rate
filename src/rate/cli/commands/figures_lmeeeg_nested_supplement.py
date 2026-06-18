"""`figures-lmeeeg-nested-supplement` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import yaml

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.paths import resolve_out_dir
from rate.viz.lmeeeg import build_lmeeeg_nested_supplement_manifest

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "figures-lmeeeg-nested-supplement",
        help="Render supplemental joint plots for nested lmeEEG results.",
    )
    add_common_config_arguments(parser)
    parser.add_argument("--config-section", default="lmeeeg_nested", help="Config section to render.")
    parser.add_argument("--viz-config", default="config/viz.yaml", help="Path to the figure config file.")
    parser.add_argument("--output", help="Optional explicit output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    out_dir = resolve_out_dir(config)
    config_section = str(getattr(args, "config_section", "lmeeeg_nested") or "lmeeeg_nested")
    output_path = Path(args.output) if args.output else out_dir / "figures" / "supplement" / "figure_manifest.json"
    viz_cfg = _load_viz_config(Path(args.viz_config))
    viz_section = dict(viz_cfg.get("viz") or {})
    figure_section = dict(((viz_section.get("figures") or {}).get("lmeeeg_nested_supplement")) or {})

    manifest = build_lmeeeg_nested_supplement_manifest(
        config=config,
        out_dir=out_dir,
        manifest_path=output_path,
        figure_config=figure_section,
        config_section=config_section,
        figure_subdir="supplement",
        formats=tuple(viz_section.get("formats", ["png", "pdf"])),
        dpi=int(viz_section.get("dpi", 300)),
    )
    LOGGER.info("Wrote nested lmeEEG supplement figure manifest with %d plots: %s", int(manifest["plot_count"]), output_path)
    return 0


def _load_viz_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Viz config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Viz config must be a mapping: {path}")
    return payload
