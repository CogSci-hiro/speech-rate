"""`glm-subject` CLI subcommand."""

from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.design import list_model_names
from rate.glm.fit import fit_subject_glm, prepare_glm_design
from rate.glm.io import build_glm_output_paths, load_glm_inputs, save_glm_result, save_glm_skip_result
from rate.util import read_parquet_dataframe

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser("glm-subject", help="Run subject-level mass-univariate GLM fitting.")
    add_common_config_arguments(parser)
    parser.add_argument("--dataset-index", help="Optional explicit dataset index parquet path.")
    parser.add_argument("--subject-id", help="Subject identifier.")
    parser.add_argument("--model-name", help="Optional GLM model name. Defaults to all configured models.")
    parser.add_argument("--epochs-path", help="Optional explicit epochs FIF input path.")
    parser.add_argument("--metadata-path", help="Optional explicit metadata CSV input path.")
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


def _select_single_subject(config: Mapping[str, Any], args: Namespace) -> Mapping[str, Any]:
    if not args.subject_id:
        raise ValueError("--subject-id is required unless --all is used.")

    index_df = _load_dataset_rows(config, args)
    mask = index_df["subject_id"].astype(str) == str(args.subject_id)
    rows = index_df.loc[mask].copy()
    if "status" in rows.columns:
        rows = rows.loc[rows["status"].astype(str).str.lower() == "ok"].copy()
    if rows.empty:
        raise ValueError(f"No runnable dataset-index rows found for subject_id={args.subject_id}")
    rows = rows.sort_values(by=["task", "run"]).reset_index(drop=True)
    return rows.iloc[0].to_dict()


def _iter_runnable_subjects(config: Mapping[str, Any], args: Namespace) -> list[Mapping[str, Any]]:
    index_df = _load_dataset_rows(config, args)
    if "status" in index_df.columns:
        index_df = index_df.loc[index_df["status"].astype(str).str.lower() == "ok"].copy()
    required = ["subject_id", "task", "run", "eeg_path"]
    index_df = index_df.dropna(subset=required)
    index_df = index_df.sort_values(by=["subject_id", "task", "run"]).reset_index(drop=True)
    unique_rows: dict[str, Mapping[str, Any]] = {}
    for row in index_df.itertuples(index=False, name="Row"):
        unique_rows.setdefault(str(row.subject_id), row._asdict())
    return list(unique_rows.values())


def _resolve_model_names(config: Mapping[str, Any], args: Namespace) -> list[str]:
    if args.model_name:
        return [str(args.model_name)]
    return list_model_names(config, section="glm")


def _fit_one_run(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    model_name: str,
    epochs_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    output_paths = build_glm_output_paths(
        config,
        row,
        model_name=model_name,
        epochs_path=epochs_path,
        metadata_path=metadata_path,
    )
    epochs, metadata_df = load_glm_inputs(output_paths.epochs_input_path, output_paths.metadata_input_path)
    design_result = prepare_glm_design(metadata_df, config, model_name=model_name)
    if design_result.n_rows_final == 0:
        summary = save_glm_skip_result(
            epochs,
            design_result,
            output_paths,
            reason="No epochs remain after design filtering.",
        )
        LOGGER.warning(
            "Skipping GLM fit for %s model=%s because no epochs survived filtering.",
            row["subject_id"],
            model_name,
        )
        return {
            "subject_id": str(row["subject_id"]),
            "model_name": model_name,
            "summary_output": str(output_paths.summary_output_path),
            "n_epochs_used": int(summary["n_epochs_used"]),
        }
    fit_result = fit_subject_glm(epochs, design_result)
    summary = save_glm_result(fit_result, output_paths)
    LOGGER.info(
        "Wrote GLM outputs for %s model=%s -> %s",
        row["subject_id"],
        model_name,
        output_paths.summary_output_path,
    )
    return {
        "subject_id": str(row["subject_id"]),
        "model_name": model_name,
        "summary_output": str(output_paths.summary_output_path),
        "n_epochs_used": int(summary["n_epochs_used"]),
    }


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    try:
        model_names = _resolve_model_names(config, args)
        if args.run_all:
            rows = _iter_runnable_subjects(config, args)
            summaries = [
                _fit_one_run(config, row, model_name=model_name)
                for row in rows
                for model_name in model_names
            ]
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(json.dumps({"status": "ok", "processed_runs": summaries}, indent=2) + "\n", encoding="utf-8")
                LOGGER.info("Wrote GLM stage marker: %s", args.output)
            return 0

        row = _select_single_subject(config, args)
        for model_name in model_names:
            _fit_one_run(
                config,
                row,
                model_name=model_name,
                epochs_path=Path(args.epochs_path) if args.epochs_path else None,
                metadata_path=Path(args.metadata_path) if args.metadata_path else None,
            )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0
