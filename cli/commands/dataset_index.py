"""`dataset-index` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace, _SubParsersAction
from pathlib import Path

from rate.config import ConfigValidationError, load_and_validate_config
from rate.dataset import DatasetIndexError, write_dataset_index

LOGGER = logging.getLogger(__name__)


def register_subparser(subparsers: _SubParsersAction) -> ArgumentParser:
    """Register the `dataset-index` subcommand.

    Parameters
    ----------
    subparsers
        Parent subparser collection from the root CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured for the ``dataset-index`` command.
    """
    parser = subparsers.add_parser("dataset-index", help="Build canonical dataset index parquet.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument("--output", help="Optional explicit index parquet output path.")
    parser.set_defaults(command_func=run)
    return parser


def run(args: Namespace) -> int:
    """Execute dataset index generation.

    Parameters
    ----------
    args
        Parsed CLI namespace for this command.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on validation or generation error).
    """
    try:
        resolved = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
        output_path = Path(args.output) if args.output else None
        index_path = write_dataset_index(resolved, output_path=output_path)
    except (ConfigValidationError, DatasetIndexError) as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Wrote dataset index: %s", index_path)
    return 0
