"""`lmeeeg-commonality` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.lmeeeg.commonality import DEFAULT_LMEEEG_COMMONALITY_SECTION, run_commonality_analysis
from rate.lmeeeg.pooling import select_lmeeeg_rows

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "lmeeeg-commonality",
        help="Run the independent pooled lmeEEG commonality analysis.",
    )
    add_common_config_arguments(parser)
    parser.add_argument("--dataset-index", help="Optional explicit dataset index parquet path.")
    parser.add_argument("--subject-id", help="Optional subject identifier filter.")
    parser.add_argument("--task", help="Optional task filter.")
    parser.add_argument("--run", help="Optional run filter.")
    parser.add_argument(
        "--config-section",
        default=DEFAULT_LMEEEG_COMMONALITY_SECTION,
        help="Config section describing the commonality analysis.",
    )
    parser.add_argument("--n-jobs", type=int, help="Optional worker count override for bootstrap and jackknife refits.")
    parser.add_argument("--trial-regressors-csv", help="Optional trialwise regressor CSV merged into epoch metadata.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    try:
        rows = select_lmeeeg_rows(
            config,
            dataset_index=args.dataset_index,
            subject_id=args.subject_id,
            task=args.task,
            run=args.run,
        )
        trial_regressors_path = Path(args.trial_regressors_csv) if getattr(args, "trial_regressors_csv", None) else None
        summary = run_commonality_analysis(
            config,
            rows,
            config_section=str(getattr(args, "config_section", DEFAULT_LMEEEG_COMMONALITY_SECTION)),
            trial_regressors_path=trial_regressors_path,
            n_jobs_override=getattr(args, "n_jobs", None),
        )
        LOGGER.info("Wrote lmeEEG commonality outputs: %s", summary["summary_path"])
        return 0
    except (FileNotFoundError, ValueError, KeyError) as exc:
        LOGGER.error("%s", exc)
        return 1
