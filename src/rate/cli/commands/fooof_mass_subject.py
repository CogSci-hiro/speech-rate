"""`fooof-mass-subject` CLI subcommand."""

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
    parser = subparsers.add_parser("fooof-mass-subject", help="Run or scaffold subject-level sliding-FOOOF extraction.")
    add_common_config_arguments(parser)
    add_subject_arguments(parser)
    parser.add_argument("--output", help="Optional explicit output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    default_name = "_stage/fooof_mass_all.done.json" if args.run_all else f"fooof_mass/{args.subject_id or 'unknown'}_fooof_mass.json"
    output_path = resolve_output_path(args, config, default_relative_path=default_name)
    payload = build_stage_payload(stage="fooof_mass_subject", args=args, config=config)
    write_placeholder_output(output_path, payload)
    LOGGER.info("Wrote FOOOF-mass scaffold output: %s", output_path)
    return 0
