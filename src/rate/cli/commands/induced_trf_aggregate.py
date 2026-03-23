"""`induced-trf-aggregate` CLI subcommand."""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.config import ConfigValidationError, load_and_validate_config

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "induced-trf-aggregate",
        help="Aggregate precomputed subject-level induced TRF scores.",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument("--out-dir", help="Optional override for cfg['paths']['out_dir'].")
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    return add_subparser(subparsers)


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    from rate.induced_trf.score import write_aggregate_output

    if config is None:
        try:
            resolved = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
        except ConfigValidationError as exc:
            LOGGER.error("%s", exc)
            return 1
    else:
        resolved = config

    out_dir = Path(args.out_dir) if args.out_dir else Path(resolved["paths"]["out_dir"])

    try:
        out_path = write_aggregate_output(out_dir=out_dir)
    except Exception as exc:
        LOGGER.error("induced-trf-aggregate failed: %s", exc)
        raise

    LOGGER.info("Wrote aggregated induced TRF scores: %s", out_path)
    return 0
