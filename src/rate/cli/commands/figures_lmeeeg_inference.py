"""`figures-lmeeeg-inference` CLI subcommand."""

from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import yaml

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.paths import resolve_out_dir
from rate.lmeeeg.targets import resolve_target_names, section_supports_targets
from rate.viz.lmeeeg import build_lmeeeg_inference_manifest

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser("figures-lmeeeg-inference", help="Render pooled lmeEEG inference QC plots.")
    add_common_config_arguments(parser)
    parser.add_argument("--config-section", default="lmeeeg", help="Config section to render.")
    parser.add_argument("--viz-config", default="config/viz.yaml", help="Path to the figure config file.")
    parser.add_argument("--output", help="Optional explicit output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    config_section = str(getattr(args, "config_section", "lmeeeg") or "lmeeeg")
    out_dir = resolve_out_dir(config)
    output_dirname = "lmeeeg_inference" if config_section == "lmeeeg" else f"{config_section}_inference"
    output_path = Path(args.output) if args.output else out_dir / "figures" / output_dirname / "figure_manifest.json"
    viz_cfg = _load_viz_config(Path(args.viz_config))
    viz_section = dict(viz_cfg.get("viz") or {})
    formats = tuple(viz_section.get("formats", ["png", "pdf"]))
    dpi = int(viz_section.get("dpi", 300))
    plots: list[dict[str, object]] = []
    if section_supports_targets(config, config_section=config_section):
        for target_name in resolve_target_names(config, config_section=config_section):
            assert target_name is not None
            target_manifest = build_lmeeeg_inference_manifest(
                out_dir=out_dir,
                stats_root=out_dir / "stats" / config_section / target_name,
                lmeeeg_root=out_dir / config_section / target_name,
                manifest_path=output_path,
                figure_subdir=f"{output_dirname}/{target_name}",
                plot_metadata={"target": target_name},
                formats=formats,
                dpi=dpi,
            )
            plots.extend(target_manifest.get("plots", []))
        manifest = {"status": "ok", "plot_count": len(plots), "plots": plots}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        manifest = build_lmeeeg_inference_manifest(
            out_dir=out_dir,
            stats_root=out_dir / "stats" / config_section,
            lmeeeg_root=out_dir / config_section,
            manifest_path=output_path,
            figure_subdir=output_dirname,
            formats=formats,
            dpi=dpi,
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
