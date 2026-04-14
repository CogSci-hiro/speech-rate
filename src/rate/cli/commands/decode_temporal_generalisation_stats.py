"""`decode-temporal-generalisation-stats` CLI subcommand."""

from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.config import ConfigValidationError
from rate.decoding import DEFAULT_DECODING_CONFIG_PATH, list_decoding_model_names, load_decoding_config, run_decoding_model_stats

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "decode-temporal-generalisation-stats",
        help="Run permutation TFCE inference for residualised temporal-generalisation decoding.",
    )
    add_common_config_arguments(parser)
    parser.add_argument("--decoding-config", default=str(DEFAULT_DECODING_CONFIG_PATH), help="Path to the standalone decoding YAML config.")
    parser.add_argument("--dataset-index", help="Optional explicit dataset index parquet path.")
    parser.add_argument("--model-name", help="Optional decoding model name. Defaults to all configured models.")
    parser.add_argument("--progress", action="store_true", help="Show a stats progress bar.")
    parser.add_argument("--all", action="store_true", dest="run_all", help="Process all configured decoding models.")
    parser.add_argument("--output", help="Optional stage marker path when using --all.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    try:
        decoding_config = load_decoding_config(Path(args.decoding_config), project_config=config)
        if not bool(decoding_config.get("stats", {}).get("enabled", True)):
            LOGGER.info("Decoding stats disabled in %s", args.decoding_config)
            if args.run_all and args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(json.dumps({"status": "skipped", "reason": "stats_disabled"}, indent=2) + "\n", encoding="utf-8")
            return 0

        model_names = [str(args.model_name)] if getattr(args, "model_name", None) else list_decoding_model_names(decoding_config)
        summaries = [
            run_decoding_model_stats(
                config,
                decoding_config,
                model_name=model_name,
                dataset_index=getattr(args, "dataset_index", None),
                show_progress=bool(getattr(args, "progress", False)),
            )
            for model_name in model_names
        ]
        if args.run_all and args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps({"status": "ok", "processed_models": summaries}, indent=2) + "\n", encoding="utf-8")
        return 0
    except (ConfigValidationError, FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
