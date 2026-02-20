"""`evoked-trf-subject` CLI subcommand."""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.config import ConfigValidationError, load_and_validate_config
from rate.evoked_trf.score import write_subject_output

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    """Register the `evoked-trf-subject` subcommand."""
    parser = subparsers.add_parser(
        "evoked-trf-subject",
        help="Compute evoked TRF scores for one dyad/subject pair.",
    )
    parser.add_argument("--dyad-id", required=True, help="Dyad ID (e.g., dyad-001).")
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., sub-001).")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument("--out-dir", help="Optional override for cfg['paths']['out_dir'].")

    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    """Backward-compatible alias for CLI registration."""
    return add_subparser(subparsers)


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    """Execute subject-level evoked TRF scoring and write one parquet output."""
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
        out_path = write_subject_output(
            cfg=resolved,
            out_dir=out_dir,
            dyad_id=str(args.dyad_id),
            subject=str(args.subject),
        )
    except Exception as exc:
        LOGGER.error("evoked-trf-subject failed for dyad_id=%s subject=%s: %s", args.dyad_id, args.subject, exc)
        raise

    LOGGER.info("Wrote evoked TRF subject scores: %s", out_path)
    return 0
