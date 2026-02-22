"""`dataset-subject-targets` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from rate.config import ConfigValidationError, load_and_validate_config
from rate.debug import limit_dataframe_by_debug_max_files

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "dataset-subject-targets",
        help="Write unique (dyad_id, subject_id) targets from dataset index.",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument("--out-dir", help="Optional override for cfg['paths']['out_dir'].")
    parser.add_argument(
        "--dataset-index",
        help="Optional explicit dataset index parquet path. Defaults to <out_dir>/dataset/index.parquet.",
    )
    parser.add_argument("--output", required=True, help="Output TSV path.")
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    return add_subparser(subparsers)


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    if config is None:
        try:
            resolved = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
        except ConfigValidationError as exc:
            LOGGER.error("%s", exc)
            return 1
    else:
        resolved = dict(config)

    out_dir = Path(args.out_dir) if args.out_dir else Path(resolved["paths"]["out_dir"])
    dataset_index_path = Path(args.dataset_index) if args.dataset_index else out_dir / "dataset" / "index.parquet"
    if not dataset_index_path.exists():
        raise FileNotFoundError(f"dataset index not found: {dataset_index_path}")

    ds = pd.read_parquet(dataset_index_path)
    ds = limit_dataframe_by_debug_max_files(
        ds,
        resolved,
        logger=LOGGER,
        label="dataset-subject-targets dataset index",
    )
    if "status" in ds.columns:
        ds = ds.loc[ds["status"].astype(str).str.lower() == "ok"].copy()

    targets = (
        ds[["dyad_id", "subject_id"]]
        .dropna()
        .astype({"dyad_id": "string", "subject_id": "string"})
        .drop_duplicates()
        .sort_values(["dyad_id", "subject_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if targets.empty:
        raise ValueError("No valid (dyad_id, subject_id) rows found in dataset index.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(out, sep="\t", index=False)
    LOGGER.info("Wrote subject targets: %s", out)
    return 0
