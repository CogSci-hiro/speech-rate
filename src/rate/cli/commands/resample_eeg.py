from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from rate.config import ConfigValidationError, get_project_downsample_hz, load_and_validate_config
from rate.debug import limit_dataframe_by_debug_max_files
from rate.eeg.resample import resample_index_table

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "resample-eeg",
        help="Cache resampled EEG raws and write dataset/index_resampled.parquet.",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing resampled FIFs.")
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    return add_subparser(subparsers)


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    if config is None:
        try:
            cfg = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
        except ConfigValidationError as exc:
            LOGGER.error("%s", exc)
            return 1
    else:
        cfg = dict(config)

    out_dir = Path(cfg["paths"]["out_dir"])
    in_index = out_dir / "dataset" / "index.parquet"
    out_index = out_dir / "dataset" / "index_resampled.parquet"

    LOGGER.info("Reading dataset index: %s", in_index)
    ds_index = pd.read_parquet(in_index)
    ds_index = limit_dataframe_by_debug_max_files(
        ds_index,
        cfg,
        logger=LOGGER,
        label="resample dataset index",
    )

    downsample_hz = get_project_downsample_hz(cfg, fallback_section="evoked_trf")
    eeg_cache_dir = out_dir / "eeg_resampled"

    LOGGER.info("Caching resampled EEG (%.3f Hz) to: %s", downsample_hz, eeg_cache_dir)
    resampled = resample_index_table(
        ds_index=ds_index,
        eeg_cache_dir=eeg_cache_dir,
        downsample_hz=downsample_hz,
        overwrite=bool(args.overwrite),
    )

    out_index.parent.mkdir(parents=True, exist_ok=True)
    resampled.to_parquet(out_index, index=False)
    LOGGER.info("Wrote resampled dataset index: %s", out_index)
    return 0
