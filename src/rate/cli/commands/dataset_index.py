"""`dataset-index` CLI subcommand."""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.config import ConfigValidationError, load_and_validate_config
from rate.dataset import DatasetIndexError, write_dataset_index

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    """Register the `dataset-index` subcommand."""
    parser = subparsers.add_parser("dataset-index", help="Build canonical dataset index parquet.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument("--output", help="Optional explicit index parquet output path.")
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    """Backward-compatible alias for CLI registration."""
    return add_subparser(subparsers)


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    """Execute dataset index generation."""
    if config is None:
        try:
            resolved = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
        except ConfigValidationError as exc:
            LOGGER.error("%s", exc)
            return 1
    else:
        resolved = config

    output_path = Path(args.output) if args.output else None
    try:
        index_path = write_dataset_index(resolved, output_path=output_path)
    except DatasetIndexError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Wrote dataset index: %s", index_path)
    return 0
