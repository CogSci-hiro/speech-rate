"""`events-table` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace, _SubParsersAction
from pathlib import Path
from typing import Any, Mapping

from rate.config import ConfigValidationError, load_and_validate_config
from rate.events import EventsTableError, write_events_outputs

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers: _SubParsersAction) -> ArgumentParser:
    """Register the `events-table` subcommand."""
    parser = subparsers.add_parser("events-table", help="Build canonical events parquet and QC CSV outputs.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument("--dataset-index", help="Optional explicit SG1 dataset index parquet path.")
    parser.add_argument("--events-output", help="Optional explicit events parquet output path.")
    parser.add_argument("--qc-output", help="Optional explicit QC CSV output path.")
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers: _SubParsersAction) -> ArgumentParser:
    """Backward-compatible alias for CLI registration."""
    return add_subparser(subparsers)


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    """Execute SG2 events table and QC generation."""
    if config is None:
        try:
            resolved = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
        except ConfigValidationError as exc:
            LOGGER.error("%s", exc)
            return 1
    else:
        resolved = config

    try:
        events_path, qc_path = write_events_outputs(
            resolved,
            dataset_index_path=Path(args.dataset_index) if args.dataset_index else None,
            events_output_path=Path(args.events_output) if args.events_output else None,
            qc_output_path=Path(args.qc_output) if args.qc_output else None,
        )
    except EventsTableError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Wrote events outputs: %s, %s", events_path, qc_path)
    return 0
