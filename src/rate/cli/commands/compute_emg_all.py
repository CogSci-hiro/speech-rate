"""`compute-emg-all` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from rate.cli.commands.compute_emg import compute_subject_emg
from rate.config import ConfigValidationError, load_and_validate_config

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "compute-emg-all",
        help="Compute subject-level EMG envelope features for all valid subjects.",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument("--out-dir", help="Optional override for cfg['paths']['out_dir'].")
    parser.add_argument(
        "--dataset-index",
        help="Optional explicit dataset index parquet path. Defaults to <out_dir>/dataset/index.parquet.",
    )
    parser.add_argument(
        "--done-file",
        help="Optional marker file path to touch when computation succeeds.",
    )
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
    ds_index = pd.read_parquet(dataset_index_path)

    if "status" in ds_index.columns:
        ds_index = ds_index.loc[ds_index["status"].astype(str).str.lower() == "ok"].copy()
    pairs = sorted(
        {
            (str(row["dyad_id"]).strip(), str(row["subject_id"]).strip())
            for _, row in ds_index[["dyad_id", "subject_id"]].drop_duplicates().iterrows()
            if str(row["dyad_id"]).strip() and str(row["subject_id"]).strip()
        }
    )
    if not pairs:
        raise ValueError("No valid (dyad_id, subject_id) rows found in dataset index.")

    for dyad_id, subject in pairs:
        output = compute_subject_emg(
            cfg=resolved,
            out_dir=out_dir,
            dataset_index_path=dataset_index_path,
            dyad_id=dyad_id,
            subject=subject,
            output_path=None,
        )
        LOGGER.info("Wrote EMG features: %s", output)

    if args.done_file:
        done = Path(args.done_file)
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text("ok\n", encoding="utf-8")
        LOGGER.info("Wrote marker: %s", done)
    return 0
