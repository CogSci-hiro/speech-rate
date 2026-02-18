"""`write-manifest` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace, _SubParsersAction
from pathlib import Path

from rate.config import ConfigValidationError, load_and_validate_config
from rate.manifest import write_run_manifest

LOGGER = logging.getLogger(__name__)


def register_subparser(subparsers: _SubParsersAction) -> ArgumentParser:
    """Register the `write-manifest` subcommand.

    Parameters
    ----------
    subparsers
        Parent subparser collection from the root CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured for the ``write-manifest`` command.
    """
    parser = subparsers.add_parser("write-manifest", help="Write run manifest JSON.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument("--output", help="Optional explicit manifest output path.")
    parser.set_defaults(command_func=run)
    return parser


def run(args: Namespace) -> int:
    """Execute run manifest generation.

    Parameters
    ----------
    args
        Parsed CLI namespace for this command.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on configuration error).
    """
    try:
        resolved = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
        manifest_path = write_run_manifest(resolved, manifest_path=Path(args.output) if args.output else None)
    except ConfigValidationError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Wrote manifest: %s", manifest_path)
    return 0
