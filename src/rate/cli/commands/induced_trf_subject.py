"""`induced-trf-subject` CLI subcommand."""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.config import ConfigValidationError, load_and_validate_config

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "induced-trf-subject",
        help="Compute induced TRF scores for one dyad/subject pair and one band.",
    )
    parser.add_argument("--dyad-id", required=True, help="Dyad ID (e.g., dyad-001).")
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., sub-001).")
    parser.add_argument("--target", required=True, help="Induced TRF target (e.g., theta, alpha, offset, exponent).")
    parser.add_argument(
        "--band",
        help="Legacy alias for raw band targets. Prefer --target.",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument("--out-dir", help="Optional override for cfg['paths']['out_dir'].")
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    return add_subparser(subparsers)


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    from rate.induced_trf.score import write_subject_output

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
                target=str(args.target),
            )
    except Exception as exc:
        LOGGER.error(
            "induced-trf-subject failed for dyad_id=%s subject=%s target=%s: %s",
            args.dyad_id,
            args.subject,
            args.target,
            exc,
        )
        raise

    LOGGER.info("Wrote induced TRF subject scores: %s", out_path)
    return 0
