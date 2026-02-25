"""`fooof` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.config import ConfigValidationError, load_and_validate_config
from rate.debug import filter_dataframe_to_unit_keys, limit_dataframe_by_debug_max_files
from rate.fooof.pipeline import run_fooof_pipeline
from rate.paths import resolve_out_dir
from rate.util import read_parquet_dataframe

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def _is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _assert_non_resampled_dataset_index(dataset_index_path: Path, index_df) -> None:
    name = dataset_index_path.name.lower()
    if name == "index_resampled.parquet":
        raise ValueError(
            "FOOOF must use non-downsampled EEG input. "
            "Pass <out_dir>/dataset/index.parquet, not dataset/index_resampled.parquet."
        )

    if "eeg_is_resampled" in index_df.columns and index_df["eeg_is_resampled"].map(_is_truthy).any():
        raise ValueError(
            "FOOOF dataset index contains resampled EEG rows (eeg_is_resampled=True). "
            "Use the original non-downsampled dataset/index.parquet."
        )

    if "eeg_path" in index_df.columns:
        eeg_paths = index_df["eeg_path"].astype(str)
        if eeg_paths.str.contains(r"(?:^|[/\\\\])eeg_resampled(?:[/\\\\]|$)", regex=True, na=False).any():
            raise ValueError(
                "FOOOF dataset index points to eeg_resampled files. "
                "Use the original non-downsampled dataset/index.parquet."
            )


def add_subparser(subparsers) -> ArgumentParser:
    """Register the `fooof` subcommand."""
    parser = subparsers.add_parser("fooof", help="Run event-aligned FOOOF feature extraction.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument("--out-dir", help="Optional override for cfg['paths']['out_dir'].")
    parser.add_argument(
        "--dataset-index",
        help="Optional explicit dataset index parquet path. Must reference non-downsampled EEG (default: <out_dir>/dataset/index.parquet).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Optional override for fooof.n_jobs (parallel subject-run workers).",
    )
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
    if args.n_jobs is not None:
        if int(args.n_jobs) < 1:
            raise ValueError("--n-jobs must be >= 1")
        fooof_block = dict(resolved.get("fooof") or {})
        fooof_block["n_jobs"] = int(args.n_jobs)
        resolved["fooof"] = fooof_block

    out_dir = Path(args.out_dir) if args.out_dir else resolve_out_dir(resolved)
    if args.dataset_index:
        dataset_index_path = Path(args.dataset_index)
    else:
        dataset_index_path = out_dir / "dataset" / "index.parquet"
    events_path = out_dir / "events" / "events.parquet"

    try:
        if not dataset_index_path.exists():
            resampled_index_path = out_dir / "dataset" / "index_resampled.parquet"
            if dataset_index_path == (out_dir / "dataset" / "index.parquet") and resampled_index_path.exists():
                raise FileNotFoundError(
                    "FOOOF requires non-downsampled dataset index at "
                    f"{dataset_index_path}, but only {resampled_index_path} was found."
                )
        LOGGER.info("Reading dataset index: %s", dataset_index_path)
        index_df = read_parquet_dataframe(dataset_index_path)
        _assert_non_resampled_dataset_index(dataset_index_path, index_df)
        index_df = limit_dataframe_by_debug_max_files(
            index_df,
            resolved,
            logger=LOGGER,
            label="fooof dataset index",
        )
        LOGGER.info("Reading canonical events table: %s", events_path)
        events_df = read_parquet_dataframe(events_path)
        events_df = filter_dataframe_to_unit_keys(events_df, index_df)

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
