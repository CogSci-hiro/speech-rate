"""`validate-config` CLI subcommand."""

from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, Namespace, _SubParsersAction
from pathlib import Path

from rate.config import ConfigValidationError, load_and_validate_config
from rate.util import ensure_parent_dir

LOGGER = logging.getLogger(__name__)


def register_subparser(subparsers: _SubParsersAction[ArgumentParser]) -> ArgumentParser:
    """Register the `validate-config` subcommand."""
    parser = subparsers.add_parser("validate-config", help="Validate config against JSON schema.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument("--output", help="Optional validation marker output path.")
    parser.set_defaults(command_func=run)
    return parser


def run(args: Namespace) -> int:
    """Execute the `validate-config` command."""
    try:
        resolved = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
    except ConfigValidationError as exc:
        LOGGER.error("%s", exc)
        return 1

    if args.output:
        output_path = Path(args.output)
        ensure_parent_dir(output_path)
        payload = {
            "status": "ok",
            "resolved_config": resolved,
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        LOGGER.info("Wrote validation marker: %s", output_path)
    else:
        LOGGER.info("Config is valid.")
    return 0
