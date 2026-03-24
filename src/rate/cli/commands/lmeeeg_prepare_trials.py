"""`lmeeeg-prepare-trials` CLI subcommand."""

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
    parser = subparsers.add_parser("lmeeeg-prepare-trials", help="Run or scaffold lmeEEG trial preparation.")
    add_common_config_arguments(parser)
    add_subject_arguments(parser)
    parser.add_argument("--output", help="Optional explicit output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    default_name = "_stage/lmeeeg_all.done.json" if args.run_all else f"lmeeeg/{args.subject_id or 'unknown'}_prepared_trials.json"
    output_path = resolve_output_path(args, config, default_relative_path=default_name)
    payload = build_stage_payload(stage="lmeeeg_prepare_trials", args=args, config=config)
    write_placeholder_output(output_path, payload)
    LOGGER.info("Wrote lmeEEG scaffold output: %s", output_path)
    return 0
