"""`epoch-subject` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.epochs import EpochBuildError, build_epoch_output_paths
from rate.epochs.build import build_epochs_for_subject
from rate.epochs.io import write_epoch_events_array, write_epoch_metadata, write_epoch_summary, write_epochs
from rate.util import read_parquet_dataframe

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser("epoch-subject", help="Run subject-level epoch construction.")
    add_common_config_arguments(parser)
    parser.add_argument("--dataset-index", help="Optional explicit dataset index parquet path.")
    parser.add_argument("--dyad-id", help="Dyad identifier.")
    parser.add_argument("--subject-id", help="Subject identifier.")
    parser.add_argument("--task", help="Task label.")
    parser.add_argument("--run", help="Run number.")
    parser.add_argument("--raw-path", help="Optional explicit preprocessed raw path.")
    parser.add_argument("--events-table", help="Optional explicit events.csv path.")
    parser.add_argument(
        "--epochs-section",
        default="epochs",
        help="Config section to use for epoch construction and output naming.",
    )
    parser.add_argument("--epochs-output", help="Optional explicit epochs FIF output path.")
    parser.add_argument("--metadata-output", help="Optional explicit metadata CSV output path.")
    parser.add_argument("--events-array-output", help="Optional explicit epoch events-array output path.")
    parser.add_argument("--summary-output", help="Optional explicit JSON summary output path.")
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
    if not all([args.dyad_id, args.subject_id]):
        raise ValueError("--dyad-id and --subject-id are required unless --all is used.")

    index_df = _load_dataset_rows(config, args)
    mask = (
        (index_df["dyad_id"].astype(str) == str(args.dyad_id))
        & (index_df["subject_id"].astype(str) == str(args.subject_id))
    )
    rows = index_df.loc[mask].copy()
    if "status" in rows.columns:
        rows = rows.loc[rows["status"].astype(str).str.lower() == "ok"].copy()
    if rows.empty:
        raise ValueError(
            "No runnable dataset-index rows found for "
            f"dyad_id={args.dyad_id}, subject_id={args.subject_id}"
        )
    rows = rows.sort_values(by=["task", "run"]).reset_index(drop=True)
    return rows.iloc[0].to_dict()


def _iter_runnable_rows(config: Mapping[str, Any], args: Namespace) -> list[Mapping[str, Any]]:
    index_df = _load_dataset_rows(config, args)
    if "status" in index_df.columns:
        index_df = index_df.loc[index_df["status"].astype(str).str.lower() == "ok"].copy()
    required = ["dyad_id", "subject_id", "task", "run", "eeg_path"]
    index_df = index_df.dropna(subset=required)
    return [row._asdict() for row in index_df.itertuples(index=False, name="Row")]


def _rows_for_subject(config: Mapping[str, Any], args: Namespace, row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    index_df = _load_dataset_rows(config, args)
    mask = (
        (index_df["dyad_id"].astype(str) == str(row["dyad_id"]))
        & (index_df["subject_id"].astype(str) == str(row["subject_id"]))
    )
    subject_rows = index_df.loc[mask].copy()
    if "status" in subject_rows.columns:
        subject_rows = subject_rows.loc[subject_rows["status"].astype(str).str.lower() == "ok"].copy()
    subject_rows = subject_rows.dropna(subset=["dyad_id", "subject_id", "task", "run", "eeg_path"])
    subject_rows = subject_rows.sort_values(by=["task", "run"]).reset_index(drop=True)
    return [record._asdict() for record in subject_rows.itertuples(index=False, name="Row")]


def _write_subject_outputs(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    events_table_path: Path | None = None,
    epochs_output_path: Path | None = None,
    metadata_output_path: Path | None = None,
    events_array_output_path: Path | None = None,
    summary_output_path: Path | None = None,
    args: Namespace,
) -> dict[str, Any]:
    output_paths = build_epoch_output_paths(
        config,
        row,
        epochs_section=str(args.epochs_section),
        events_table_path=events_table_path,
    )
    subject_rows = _rows_for_subject(config, args, row)
    result = build_epochs_for_subject(
        config,
        subject_rows,
        epochs_section=str(args.epochs_section),
        events_table_path=events_table_path or output_paths.events_table_path,
    )

    written_epochs = write_epochs(result.epochs, epochs_output_path or output_paths.epochs_output_path)
    written_metadata = write_epoch_metadata(result.metadata, metadata_output_path or output_paths.metadata_output_path)
    written_events_array = write_epoch_events_array(result.events_array, events_array_output_path or output_paths.events_array_output_path)
    written_summary = write_epoch_summary(result.summary, summary_output_path or output_paths.summary_output_path)
    LOGGER.info(
        "Wrote subject-level epochs for %s -> %s",
        row["subject_id"],
        written_epochs,
    )
    return {
        "subject_id": row["subject_id"],
        "epochs_output": str(written_epochs),
        "metadata_output": str(written_metadata),
        "events_array_output": str(written_events_array),
        "summary_output": str(written_summary),
        "epochs_count": int(result.summary["epochs_count"]),
    }


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    try:
        if args.run_all:
            rows = _iter_runnable_rows(config, args)
            unique_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
            for row in rows:
                unique_rows[(str(row["dyad_id"]), str(row["subject_id"]))] = row
            summaries = [_write_subject_outputs(config, row, args=args) for row in unique_rows.values()]
            if args.output:
                write_epoch_summary({"status": "ok", "processed_subjects": summaries}, Path(args.output))
                LOGGER.info("Wrote epoching stage marker: %s", args.output)
            return 0

        row = _select_single_row(config, args)
        _write_subject_outputs(
            config,
            row,
            events_table_path=Path(args.events_table) if args.events_table else None,
            epochs_output_path=Path(args.epochs_output) if args.epochs_output else None,
            metadata_output_path=Path(args.metadata_output) if args.metadata_output else None,
            events_array_output_path=Path(args.events_array_output) if args.events_array_output else None,
            summary_output_path=Path(args.summary_output) if args.summary_output else None,
            args=args,
        )
    except (EpochBuildError, FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0
