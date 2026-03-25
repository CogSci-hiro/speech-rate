"""`fooof-mass-subject` CLI subcommand."""

from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.design import list_model_names, load_model_spec
from rate.fooof_mass.features import extract_fooof_features
from rate.fooof_mass.fit import fit_fooof_tensor
from rate.fooof_mass.glm import fit_fooof_mass_glm
from rate.fooof_mass.io import build_fooof_mass_output_paths, load_fooof_mass_inputs, save_fooof_mass_result
from rate.fooof_mass.spectra import compute_window_spectra
from rate.fooof_mass.windows import make_sliding_windows
from rate.util import read_parquet_dataframe

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser("fooof-mass-subject", help="Run subject-level sliding-window FOOOF mass GLM.")
    add_common_config_arguments(parser)
    parser.add_argument("--dataset-index", help="Optional explicit dataset index parquet path.")
    parser.add_argument("--subject-id", help="Subject identifier.")
    parser.add_argument("--task", help="Task label.")
    parser.add_argument("--run", help="Run number.")
    parser.add_argument("--model-name", help="Optional FOOOF-mass model name. Defaults to all configured models.")
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


def _select_single_row(config: Mapping[str, Any], args: Namespace) -> Mapping[str, Any]:
    if not all([args.subject_id, args.task, args.run]):
        raise ValueError("--subject-id, --task, and --run are required unless --all is used.")

    index_df = _load_dataset_rows(config, args)
    mask = (
        (index_df["subject_id"].astype(str) == str(args.subject_id))
        & (index_df["task"].astype(str) == str(args.task))
        & (index_df["run"].astype(str) == str(args.run))
    )
    rows = index_df.loc[mask].copy()
    if "status" in rows.columns:
        rows = rows.loc[rows["status"].astype(str).str.lower() == "ok"].copy()
    if rows.empty:
        raise ValueError(
            "No runnable dataset-index row found for "
            f"subject_id={args.subject_id}, task={args.task}, run={args.run}"
        )
    return rows.iloc[0].to_dict()


def _iter_runnable_rows(config: Mapping[str, Any], args: Namespace) -> list[Mapping[str, Any]]:
    index_df = _load_dataset_rows(config, args)
    if "status" in index_df.columns:
        index_df = index_df.loc[index_df["status"].astype(str).str.lower() == "ok"].copy()
    required = ["subject_id", "task", "run", "eeg_path"]
    index_df = index_df.dropna(subset=required)
    return [row._asdict() for row in index_df.itertuples(index=False, name="Row")]


def _resolve_model_names(config: Mapping[str, Any], args: Namespace) -> list[str]:
    if args.model_name:
        return [str(args.model_name)]
    return list_model_names(config, section="fooof")


def _run_one_subject(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    model_names: list[str],
    epochs_path: Path | None = None,
    metadata_path: Path | None = None,
) -> list[dict[str, Any]]:
    first_spec = load_model_spec(config, section="fooof", model_name=model_names[0])
    first_paths = build_fooof_mass_output_paths(
        config,
        row,
        model_name=first_spec.name,
        outcome_name=str(first_spec.outcome or first_spec.name),
        epochs_path=epochs_path,
        metadata_path=metadata_path,
    )
    epochs, metadata_df = load_fooof_mass_inputs(first_paths.epochs_input_path, first_paths.metadata_input_path)
    windows = make_sliding_windows(np.asarray(epochs.times, dtype=float), float(epochs.info["sfreq"]), config)
    spectra_result = compute_window_spectra(epochs, windows, config)
    fit_result = fit_fooof_tensor(spectra_result.frequencies, spectra_result.psd, config)
    feature_result = extract_fooof_features(fit_result, config)

    summaries: list[dict[str, Any]] = []
    for model_name in model_names:
        model_spec = load_model_spec(config, section="fooof", model_name=model_name)
        outcome_name = str(model_spec.outcome or model_name)
        if outcome_name not in feature_result.feature_tensors:
            raise ValueError(f"Requested FOOOF outcome `{outcome_name}` is not available in extracted features.")

        glm_result = fit_fooof_mass_glm(
            feature_result.feature_tensors[outcome_name],
            metadata_df,
            config,
            model_name=model_name,
            channel_names=list(epochs.ch_names),
            window_centers_s=np.asarray([window.center_s for window in windows], dtype=float),
        )
        output_paths = build_fooof_mass_output_paths(
            config,
            row,
            model_name=model_name,
            outcome_name=outcome_name,
            epochs_path=epochs_path,
            metadata_path=metadata_path,
        )
        summary = save_fooof_mass_result(
            output_paths=output_paths,
            windows=windows,
            fit_result=fit_result,
            feature_result=feature_result,
            glm_result=glm_result,
            feature_tensor=feature_result.feature_tensors[outcome_name],
        )
        LOGGER.info(
            "Wrote FOOOF-mass outputs for %s task=%s run=%s model=%s -> %s",
            row["subject_id"],
            row["task"],
            row["run"],
            model_name,
            output_paths.summary_output_path,
        )
        summaries.append(
            {
                "subject_id": str(row["subject_id"]),
                "task": str(row["task"]),
                "run": str(row["run"]),
                "model_name": model_name,
                "summary_output": str(output_paths.summary_output_path),
                "n_epochs_used": int(summary["n_epochs_used"]),
            }
        )
    return summaries


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    try:
        model_names = _resolve_model_names(config, args)
        if args.run_all:
            rows = _iter_runnable_rows(config, args)
            summaries = [
                summary
                for row in rows
                for summary in _run_one_subject(config, row, model_names=model_names)
            ]
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(json.dumps({"status": "ok", "processed_runs": summaries}, indent=2) + "\n", encoding="utf-8")
                LOGGER.info("Wrote FOOOF-mass stage marker: %s", args.output)
            return 0

        row = _select_single_row(config, args)
        _run_one_subject(
            config,
            row,
            model_names=model_names,
            epochs_path=Path(args.epochs_path) if args.epochs_path else None,
            metadata_path=Path(args.metadata_path) if args.metadata_path else None,
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0
