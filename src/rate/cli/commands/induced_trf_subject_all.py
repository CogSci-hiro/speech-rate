"""`induced-trf-subject-all` CLI subcommand."""

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
        "induced-trf-subject-all",
        help="Compute induced TRF subject artifacts for all valid subjects.",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument("--out-dir", help="Optional override for cfg['paths']['out_dir'].")
    parser.add_argument(
        "--dataset-index",
        help="Optional explicit dataset index parquet path. Defaults to <out_dir>/dataset/index.parquet.",
    )
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="Optional induced-target filter (e.g., theta, alpha, offset, exponent). Can be repeated.",
    )
    parser.add_argument("--done-file", help="Optional marker file path to write on success.")
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    return add_subparser(subparsers)


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    from rate.induced_trf.score import parse_induced_trf_config, write_subject_output

    if config is None:
        try:
            resolved = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
        except ConfigValidationError as exc:
            LOGGER.error("%s", exc)
            return 1
    else:
        resolved = dict(config)

    out_dir = Path(args.out_dir) if args.out_dir else Path(resolved["paths"]["out_dir"])
    if args.dataset_index:
        dataset_index_path = Path(args.dataset_index)
    else:
        preferred = out_dir / "dataset" / "index.parquet"
        fallback = out_dir / "dataset" / "index_resampled.parquet"
        dataset_index_path = preferred if preferred.exists() else fallback
    if not dataset_index_path.exists():
        raise FileNotFoundError(f"dataset index not found: {dataset_index_path}")

    ds = pd.read_parquet(dataset_index_path)
    ds = limit_dataframe_by_debug_max_files(
        ds,
        resolved,
        logger=LOGGER,
        label="induced-trf-subject-all dataset index",
    )
    if "status" in ds.columns:
        ds = ds.loc[ds["status"].astype(str).str.lower() == "ok"].copy()
    pairs = sorted(
        {
            (str(row["dyad_id"]).strip(), str(row["subject_id"]).strip())
            for _, row in ds[["dyad_id", "subject_id"]].drop_duplicates().iterrows()
            if str(row["dyad_id"]).strip() and str(row["subject_id"]).strip()
        }
    )
    if not pairs:
        raise ValueError("No valid (dyad_id, subject_id) rows found in dataset index.")

    cfg_ind = parse_induced_trf_config(resolved)
    enabled_targets = list(cfg_ind.requested_targets)
    requested_targets: set[str] | None = None
    if args.targets:
        requested = [str(target).strip().lower() for target in args.targets if str(target).strip()]
        bad = sorted(set(requested) - set(enabled_targets))
        if bad:
            raise ValueError(f"Requested induced targets not enabled in config induced_trf.models[*].targets: {bad}")
        requested_targets = set(requested)

    targets = [target for target in enabled_targets if requested_targets is None or target in requested_targets]

    for target in targets:
        for dyad_id, subject in pairs:
            out_path = write_subject_output(
                cfg=resolved,
                out_dir=out_dir,
                dyad_id=dyad_id,
                subject=subject,
                target=target,
            )
            LOGGER.info("Wrote induced TRF subject scores: %s", out_path)

    if args.done_file:
        done = Path(args.done_file)
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text("ok\n", encoding="utf-8")
        LOGGER.info("Wrote marker: %s", done)
    return 0
