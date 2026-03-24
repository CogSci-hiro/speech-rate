"""`glm-group-stats` CLI subcommand."""

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
    parser = subparsers.add_parser("glm-group-stats", help="Run or scaffold group-level GLM statistics.")
    add_common_config_arguments(parser)
    parser.add_argument("--stats-config", default="config/stats.yaml", help="Path to the stats config file.")
    parser.add_argument("--output", help="Optional explicit output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    output_path = Path(args.output) if args.output else resolve_out_dir(config) / "stats" / "glm_group_stats.json"
    payload = build_stage_payload(
        stage="glm_group_stats",
        args=args,
        config=config,
        extra={"stats_config": args.stats_config},
    )
    write_placeholder_output(output_path, payload)
    LOGGER.info("Wrote GLM group-stats scaffold output: %s", output_path)
    return 0
