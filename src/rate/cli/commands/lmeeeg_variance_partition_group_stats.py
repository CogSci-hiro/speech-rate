"""`lmeeeg-variance-partition-group-stats` CLI subcommand."""

from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.lmeeeg.variance_partition import build_group_descriptive_summary
from rate.paths import ensure_parent_dir, resolve_out_dir

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "lmeeeg-variance-partition-group-stats",
        help="Write descriptive pooled maps for the explicit lmeEEG variance-partition analysis.",
    )
    add_common_config_arguments(parser)
    parser.add_argument("--output", help="Optional explicit summary JSON output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    try:
        summary = build_group_descriptive_summary(config)
        output_path = Path(args.output) if getattr(args, "output", None) else resolve_out_dir(config) / "stats" / "lmeeeg_variance_partition_group_stats.json"
        ensure_parent_dir(output_path)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        LOGGER.info("Wrote lmeEEG variance-partition group summary: %s", output_path)
        return 0
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
