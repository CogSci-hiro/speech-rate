"""`figures-lmeeeg-inference` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import yaml

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.paths import resolve_out_dir
from rate.viz.lmeeeg import build_lmeeeg_inference_manifest

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser("figures-lmeeeg-inference", help="Render pooled lmeEEG inference QC plots.")
    add_common_config_arguments(parser)
    parser.add_argument("--viz-config", default="config/viz.yaml", help="Path to the figure config file.")
    parser.add_argument("--output", help="Optional explicit output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    output_path = Path(args.output) if args.output else resolve_out_dir(config) / "figures" / "lmeeeg_inference" / "figure_manifest.json"
    viz_cfg = _load_viz_config(Path(args.viz_config))
    viz_section = dict(viz_cfg.get("viz") or {})
    manifest = build_lmeeeg_inference_manifest(
        out_dir=resolve_out_dir(config),
        stats_root=resolve_out_dir(config) / "stats" / "lmeeeg",
        lmeeeg_root=resolve_out_dir(config) / "lmeeeg",
        manifest_path=output_path,
        formats=tuple(viz_section.get("formats", ["png", "pdf"])),
        dpi=int(viz_section.get("dpi", 300)),
    )
    LOGGER.info("Wrote lmeEEG inference figure manifest with %d plots: %s", int(manifest["plot_count"]), output_path)
    return 0


def _load_viz_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Viz config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Viz config must be a mapping: {path}")
    return payload
