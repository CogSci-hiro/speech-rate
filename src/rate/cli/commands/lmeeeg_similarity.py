"""`lmeeeg-similarity` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from typing import Any, Mapping

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.lmeeeg.similarity import run_similarity_analysis

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "lmeeeg-similarity",
        help="Compare two existing pooled lmeEEG predictor maps with similarity and difference metrics.",
    )
    add_common_config_arguments(parser)
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    try:
        summary = run_similarity_analysis(config)
        LOGGER.info("Wrote lmeEEG similarity outputs: %s", summary["summary_path"])
        return 0
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
