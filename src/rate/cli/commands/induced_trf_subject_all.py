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
        "--band",
        action="append",
        dest="bands",
        help="Optional induced-band filter (e.g., theta). Can be repeated.",
    )
    parser.add_argument(
        "--target-kind",
        action="append",
        dest="target_kinds",
        help="Optional target-kind filter (raw or residual). Can be repeated.",
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
    enabled_bands = list(cfg_ind.bands)
    enabled_target_kinds = list(cfg_ind.target_kinds)
    requested_bands: set[str] | None = None
    if args.bands:
        requested = [str(b).strip() for b in args.bands if str(b).strip()]
        bad = sorted(set(requested) - set(enabled_bands))
        if bad:
            raise ValueError(f"Requested bands not enabled in config induced_trf.bands: {bad}")
        requested_bands = set(requested)

    if args.target_kinds:
        requested_target_kinds = [str(v).strip().lower() for v in args.target_kinds if str(v).strip()]
        bad_target_kinds = sorted(set(requested_target_kinds) - set(enabled_target_kinds))
        if bad_target_kinds:
            raise ValueError(f"Requested target kinds not enabled in config induced_trf.target_kinds: {bad_target_kinds}")
        target_kinds = [k for k in enabled_target_kinds if k in set(requested_target_kinds)]
    else:
        target_kinds = enabled_target_kinds

    for target_kind in target_kinds:
        bands_for_kind = list(cfg_ind.bands_for_target_kind(target_kind))
        if requested_bands is not None:
            bands_for_kind = [b for b in bands_for_kind if b in requested_bands]
        for band in bands_for_kind:
            for dyad_id, subject in pairs:
                out_path = write_subject_output(
                    cfg=resolved,
                    out_dir=out_dir,
                    dyad_id=dyad_id,
                    subject=subject,
                    band=band,
                    target_kind=target_kind,
                )
                LOGGER.info("Wrote induced TRF subject scores: %s", out_path)

    if args.done_file:
        done = Path(args.done_file)
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text("ok\n", encoding="utf-8")
        LOGGER.info("Wrote marker: %s", done)
    return 0
