"""`preprocess-subject` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.preprocessing.io import build_preprocessing_output_paths, write_preprocessing_summary
from rate.preprocessing.pipeline import preprocess_run
from rate.util import read_parquet_dataframe

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser("preprocess-subject", help="Run subject-level preprocessing.")
    add_common_config_arguments(parser)
    parser.add_argument("--dataset-index", help="Optional explicit dataset index parquet path.")
    parser.add_argument("--dyad-id", help="Dyad identifier.")
    parser.add_argument("--subject-id", help="Subject identifier.")
    parser.add_argument("--task", help="Task label.")
    parser.add_argument("--run", help="Run number.")
    parser.add_argument("--raw-path", help="Optional explicit raw EEG path.")
    parser.add_argument("--channels-tsv", help="Optional explicit BIDS channels.tsv path.")
    parser.add_argument("--ica-path", help="Optional explicit precomputed ICA path.")
    parser.add_argument("--eeg-output", help="Optional explicit preprocessed EEG output path.")
    parser.add_argument("--emg-output", help="Optional explicit EMG artifact output path.")
    parser.add_argument("--events-output", help="Optional explicit events output path.")
    parser.add_argument("--summary-output", help="Optional explicit summary output path.")
    parser.add_argument("--all", action="store_true", dest="run_all", help="Process all runnable rows in the dataset index.")
    parser.add_argument("--output", help="Optional stage marker path when using --all.")
    return parser


def _dataset_index_path(config: Mapping[str, Any], args: Namespace) -> Path:
    if args.dataset_index:
        return Path(args.dataset_index)
    return Path(str(config["paths"]["out_dir"])) / "dataset" / "index.parquet"


def _load_dataset_rows(config: Mapping[str, Any], args: Namespace):
    dataset_index_path = _dataset_index_path(config, args)
    if not dataset_index_path.exists():
        raise FileNotFoundError(f"Dataset index not found: {dataset_index_path}")
    return read_parquet_dataframe(dataset_index_path)


def _select_single_row(config: Mapping[str, Any], args: Namespace) -> Mapping[str, Any]:
    if not all([args.dyad_id, args.subject_id, args.task, args.run]):
        raise ValueError("--dyad-id, --subject-id, --task, and --run are required unless --all is used.")

    index_df = _load_dataset_rows(config, args)
    mask = (
        (index_df["dyad_id"].astype(str) == str(args.dyad_id))
        & (index_df["subject_id"].astype(str) == str(args.subject_id))
        & (index_df["task"].astype(str) == str(args.task))
        & (index_df["run"].astype(str) == str(args.run))
    )
    rows = index_df.loc[mask].copy()
    if "status" in rows.columns:
        rows = rows.loc[rows["status"].astype(str).str.lower() == "ok"].copy()
    if rows.empty:
        raise ValueError(
            "No runnable dataset-index row found for "
            f"dyad_id={args.dyad_id}, subject_id={args.subject_id}, task={args.task}, run={args.run}"
        )
    return rows.iloc[0].to_dict()


def _iter_runnable_rows(config: Mapping[str, Any], args: Namespace) -> list[Mapping[str, Any]]:
    index_df = _load_dataset_rows(config, args)
    if "status" in index_df.columns:
        index_df = index_df.loc[index_df["status"].astype(str).str.lower() == "ok"].copy()
    required = ["dyad_id", "subject_id", "task", "run", "raw_eeg_path"]
    index_df = index_df.dropna(subset=required)
    return [row._asdict() for row in index_df.itertuples(index=False, name="Row")]


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None

    if args.run_all:
        rows = _iter_runnable_rows(config, args)
        summaries: list[dict[str, Any]] = []
        for row in rows:
            output_paths = build_preprocessing_output_paths(config, row)
            preprocess_run(config, row)
            summaries.append(
                {
                    "subject_id": row["subject_id"],
                    "task": row["task"],
                    "run": row["run"],
                    "summary_output": str(output_paths.summary_output_path),
                }
            )

        if args.output:
            marker_path = Path(args.output)
            write_preprocessing_summary(
                marker_path,
                {"status": "ok", "processed_runs": summaries},
            )
            LOGGER.info("Wrote preprocessing stage marker: %s", marker_path)
        return 0

    row = _select_single_row(config, args)
    preprocess_run(
        config,
        row,
        raw_eeg_path=Path(args.raw_path) if args.raw_path else None,
        channels_tsv_path=Path(args.channels_tsv) if args.channels_tsv else None,
        ica_path=Path(args.ica_path) if args.ica_path else None,
        eeg_output_path=Path(args.eeg_output) if args.eeg_output else None,
        emg_output_path=Path(args.emg_output) if args.emg_output else None,
        events_output_path=Path(args.events_output) if args.events_output else None,
        summary_output_path=Path(args.summary_output) if args.summary_output else None,
    )
    return 0
