"""`fooof` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.config import ConfigValidationError, load_and_validate_config
from rate.fooof.pipeline import run_fooof_pipeline
from rate.paths import resolve_out_dir
from rate.util import read_parquet_dataframe

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    """Register the `fooof` subcommand."""
    parser = subparsers.add_parser("fooof", help="Run event-aligned FOOOF feature extraction.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument("--out-dir", help="Optional override for cfg['paths']['out_dir'].")
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    """Backward-compatible alias for CLI registration."""
    return add_subparser(subparsers)


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    """Execute the FOOOF feature extraction pipeline."""
    if config is None:
        try:
            resolved = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
        except ConfigValidationError as exc:
            LOGGER.error("%s", exc)
            return 1
    else:
        resolved = dict(config)

    out_dir = Path(args.out_dir) if args.out_dir else resolve_out_dir(resolved)
    dataset_index_path = out_dir / "dataset" / "index_resampled.parquet"
    events_path = out_dir / "events" / "events.parquet"

    try:
        LOGGER.info("Reading dataset index: %s", dataset_index_path)
        index_df = read_parquet_dataframe(dataset_index_path)
        LOGGER.info("Reading canonical events table: %s", events_path)
        events_df = read_parquet_dataframe(events_path)

        run_fooof_pipeline(
            config=dict(resolved),
            dataset_index=index_df,
            events_df=events_df,
            out_dir=out_dir,
        )
    except Exception as exc:
        LOGGER.error("fooof failed: %s", exc)
        raise

    LOGGER.info("Wrote FOOOF outputs under: %s", out_dir / "fooof")
    return 0
