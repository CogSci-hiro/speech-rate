"""`build-events` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.events import EventsTableError, write_events_outputs
from rate.paths import resolve_out_dir

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    """Register the `build-events` subcommand."""
    parser = subparsers.add_parser("build-events", help="Build the project-level canonical events table.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument("--dataset-index", help="Optional explicit dataset index parquet path.")
    parser.add_argument("--output", help="Optional explicit events.csv output path.")
    parser.add_argument("--qc-output", help="Optional explicit QC CSV output path.")
    parser.add_argument("--validation-output", help="Optional explicit validation JSON output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    """Execute project-level events table generation."""
    assert config is not None
    out_dir = resolve_out_dir(config)
    dataset_index_path = Path(args.dataset_index) if args.dataset_index else out_dir / "dataset" / "index.parquet"
    output_path = Path(args.output) if args.output else None
    qc_output_path = Path(args.qc_output) if args.qc_output else None
    validation_output_path = Path(args.validation_output) if args.validation_output else None

    try:
        events_path, qc_path, validation_path = write_events_outputs(
            config,
            dataset_index_path=dataset_index_path,
            events_output_path=output_path,
            qc_output_path=qc_output_path,
            validation_output_path=validation_output_path,
        )
    except EventsTableError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Wrote events table CSV: %s", events_path)
    LOGGER.info("Wrote events QC summary: %s", qc_path)
    LOGGER.info("Wrote events validation summary: %s", validation_path)
    return 0
