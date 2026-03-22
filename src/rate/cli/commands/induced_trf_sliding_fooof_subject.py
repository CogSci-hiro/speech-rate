"""`induced-trf-sliding-fooof-subject` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd

from rate.induced_trf.sliding_fooof import (
    done_marker_path,
    load_sliding_fooof_config,
    write_subject_sliding_fooof_outputs,
)

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = False


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "induced-trf-sliding-fooof-subject",
        help="Precompute sliding-window FOOOF residual targets for one induced-TRF subject.",
    )
    parser.add_argument("--config", default="config/fooof.yaml", help="Path to sliding FOOOF YAML config.")
    parser.add_argument("--out-dir", required=True, help="Compute output directory.")
    parser.add_argument("--dyad-id", required=True, help="Dyad ID (e.g., dyad-001).")
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., sub-001).")
    parser.add_argument("--band", required=True, help="Band name (e.g., theta).")
    parser.add_argument("--band-fmin", required=True, type=float, help="Band lower edge in Hz.")
    parser.add_argument("--band-fmax", required=True, type=float, help="Band upper edge in Hz.")
    parser.add_argument("--done-file", help="Optional explicit done marker path.")
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    return add_subparser(subparsers)


def run(args: Namespace, config=None) -> int:
    del config
    out_dir = Path(args.out_dir)
    dataset_index_path = out_dir / "dataset" / "index.parquet"
    if not dataset_index_path.exists():
        raise FileNotFoundError(f"dataset index not found: {dataset_index_path}")

    sliding_cfg = load_sliding_fooof_config(Path(args.config))
    dataset_index = pd.read_parquet(dataset_index_path)
    write_subject_sliding_fooof_outputs(
        sliding_cfg=sliding_cfg,
        out_dir=out_dir,
        dyad_id=str(args.dyad_id),
        subject=str(args.subject),
        band=str(args.band),
        band_hz=(float(args.band_fmin), float(args.band_fmax)),
        dataset_index=dataset_index,
    )

    done_path = Path(args.done_file) if args.done_file else done_marker_path(
        out_dir=out_dir,
        band=str(args.band),
        dyad_id=str(args.dyad_id),
        subject=str(args.subject),
    )
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text("ok\n", encoding="utf-8")
    LOGGER.info("Wrote sliding FOOOF residual targets: %s", done_path.parent)
    return 0
