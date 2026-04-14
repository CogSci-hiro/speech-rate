"""`lmeeeg-variance-partition-fit` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.lmeeeg.pooling import select_lmeeeg_rows
from rate.lmeeeg.variance_partition import fit_variance_partition_table, save_variance_partition_fit

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "lmeeeg-variance-partition-fit",
        help="Run the explicit pooled lmeEEG variance-partition analysis.",
    )
    add_common_config_arguments(parser)
    parser.add_argument("--dataset-index", help="Optional explicit dataset index parquet path.")
    parser.add_argument("--subject-id", help="Optional subject identifier filter.")
    parser.add_argument("--task", help="Optional task filter.")
    parser.add_argument("--run", help="Optional run filter.")
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
        LOGGER.info("Running pooled lmeEEG variance partition across %d selected dataset rows", len(rows))
        table, summary = fit_variance_partition_table(
            config,
            rows,
            trial_regressors_path=trial_regressors_path,
        )
        written_summary = save_variance_partition_fit(config=config, table=table, summary=summary)
        LOGGER.info("Wrote lmeEEG variance-partition outputs: %s", written_summary["summary_path"])
        return 0
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
