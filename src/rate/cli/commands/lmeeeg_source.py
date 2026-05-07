"""`lmeeeg-source` CLI subcommand."""

from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import yaml

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.lmeeeg.pooling import select_lmeeeg_rows
from rate.lmeeeg_source.pipeline import fit_source_models, write_source_group_maps
from rate.lmeeeg_source.plot import build_lmeeeg_source_qc_figure
from rate.lmeeeg_source.source_space import prepare_source_subject

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser("lmeeeg-source", help="Run the experimental source-space lmeEEG pipeline.")
    add_common_config_arguments(parser)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare-subject", "fit", "group-stats", "plot-qc"),
        help="Which experimental source-space stage to run.",
    )
    parser.add_argument("--dataset-index", help="Optional explicit dataset index parquet path.")
    parser.add_argument("--subject-id", help="Optional subject identifier for subject preparation.")
    parser.add_argument("--config-section", default="lmeeeg_source", help="Config section to use.")
    parser.add_argument("--model-name", help="Optional model name filter for pooled fitting.")
    parser.add_argument("--stats-config", default="config/stats.yaml", help="Path to the stats config file for source inference.")
    parser.add_argument("--output", help="Optional explicit stage-marker or manifest output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    config_section = str(getattr(args, "config_section", "lmeeeg_source") or "lmeeeg_source")
    try:
        if args.stage == "prepare-subject":
            if not args.subject_id:
                raise ValueError("`--subject-id` is required for `lmeeeg-source --stage prepare-subject`.")
            rows = select_lmeeeg_rows(config, dataset_index=args.dataset_index, subject_id=args.subject_id)
            row = rows[0]
            summary = prepare_source_subject(config, row, config_section=config_section)
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return 0

        if args.stage == "fit":
            summaries = fit_source_models(
                config,
                config_section=config_section,
                dataset_index=args.dataset_index,
                model_name=args.model_name,
            )
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps({"status": "ok", "fits": summaries}, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            return 0

        if args.stage == "group-stats":
            stats_cfg = _load_stats_config(Path(args.stats_config))
            summaries = write_source_group_maps(
                config,
                config_section=config_section,
                stats_cfg=stats_cfg,
                dataset_index=args.dataset_index,
            )
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps({"status": "ok", "group_maps": summaries}, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            return 0

        if args.stage == "plot-qc":
            manifest = build_lmeeeg_source_qc_figure(
                config,
                output_path=Path(args.output) if args.output else None,
                config_section=config_section,
            )
            LOGGER.info("Wrote lmeEEG source QC manifest: %s", args.output or manifest)
            return 0
    except (FileNotFoundError, ValueError, KeyError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


def _load_stats_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Stats config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Stats config must be a mapping: {path}")
    return payload
