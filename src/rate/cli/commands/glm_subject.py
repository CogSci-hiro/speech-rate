"""`glm-subject` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from typing import Any, Mapping

from rate.cli.commands._workflow_common import (
    add_common_config_arguments,
    add_subject_arguments,
    build_stage_payload,
    resolve_output_path,
    write_placeholder_output,
)

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser("glm-subject", help="Run or scaffold subject-level mass-univariate GLM fitting.")
    add_common_config_arguments(parser)
    add_subject_arguments(parser)
    parser.add_argument("--output", help="Optional explicit output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    default_name = "_stage/glm_all.done.json" if args.run_all else f"glm/{args.subject_id or 'unknown'}_glm.json"
    output_path = resolve_output_path(args, config, default_relative_path=default_name)
    payload = build_stage_payload(stage="glm_subject", args=args, config=config)
    write_placeholder_output(output_path, payload)
    LOGGER.info("Wrote GLM subject scaffold output: %s", output_path)
    return 0
