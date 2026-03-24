"""`figures-fooof` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.cli.commands._workflow_common import add_common_config_arguments, build_stage_payload, write_placeholder_output
from rate.paths import resolve_out_dir

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser("figures-fooof", help="Run or scaffold FOOOF figure generation.")
    add_common_config_arguments(parser)
    parser.add_argument("--viz-config", default="config/viz.yaml", help="Path to the figure config file.")
    parser.add_argument("--output", help="Optional explicit output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    output_path = Path(args.output) if args.output else resolve_out_dir(config) / "figures" / "fooof" / "figure_manifest.json"
    payload = build_stage_payload(
        stage="figures_fooof",
        args=args,
        config=config,
        extra={"viz_config": args.viz_config},
    )
    write_placeholder_output(output_path, payload)
    LOGGER.info("Wrote FOOOF figure scaffold output: %s", output_path)
    return 0
