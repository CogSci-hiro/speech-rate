"""`figures-decoding` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import yaml

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.decoding.config import DEFAULT_DECODING_CONFIG_PATH, load_decoding_config
from rate.paths import resolve_out_dir
from rate.viz.decoding import build_decoding_manifest

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser("figures-decoding", help="Render temporal-generalisation decoding figures.")
    add_common_config_arguments(parser)
    parser.add_argument("--viz-config", default="config/viz.yaml", help="Path to the figure config file.")
    parser.add_argument("--decoding-config", default=str(DEFAULT_DECODING_CONFIG_PATH), help="Path to the standalone decoding YAML config.")
    parser.add_argument("--output", help="Optional explicit output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    out_dir = resolve_out_dir(config)
    output_path = Path(args.output) if args.output else out_dir / "figures" / "decoding" / "figure_manifest.json"
    viz_cfg = _load_viz_config(Path(args.viz_config))
    viz_section = dict(viz_cfg.get("viz") or {})
    decoding_viz_section = dict(((viz_section.get("figures") or {}).get("decoding")) or {})
    theme_section = dict(viz_section.get("theme") or {})
    decoding_config = load_decoding_config(Path(args.decoding_config), project_config=config)

    manifest = build_decoding_manifest(
        out_dir=out_dir,
        decoding_config=decoding_config,
        manifest_path=output_path,
        figure_config=decoding_viz_section,
        theme_config=theme_section,
        formats=tuple(viz_section.get("formats", ["png", "pdf"])),
        dpi=int(viz_section.get("dpi", 300)),
    )
    LOGGER.info("Wrote decoding figure manifest with %d plots: %s", int(manifest["plot_count"]), output_path)
    return 0


def _load_viz_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Viz config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Viz config must be a mapping: {path}")
    return payload
