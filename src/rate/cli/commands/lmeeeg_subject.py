"""`lmeeeg-subject` CLI subcommand."""

from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.design import list_model_names
from rate.lmeeeg.io import (
    build_lmeeeg_model_output_paths,
    build_lmeeeg_output_paths,
    load_lmeeeg_inputs_with_trial_regressors,
    save_lmeeeg_result,
)
from rate.lmeeeg.mass_model import fit_lmeeeg_mass_model
from rate.lmeeeg.pooling import fit_pooled_lmeeeg_model, load_pooled_lmeeeg_inputs, select_lmeeeg_rows
from rate.lmeeeg.trial_data import build_lmeeeg_trial_table

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser("lmeeeg-subject", help="Run pooled lmeEEG model fitting.")
    add_common_config_arguments(parser)
    parser.add_argument("--dataset-index", help="Optional explicit dataset index parquet path.")
    parser.add_argument("--subject-id", help="Optional subject identifier filter.")
    parser.add_argument("--task", help="Optional task filter.")
    parser.add_argument("--run", help="Optional run filter.")
    parser.add_argument("--config-section", default="lmeeeg", help="Config section to use for lmeEEG model resolution.")
    parser.add_argument("--target", help="Optional target name for target-specific analyses.")
    parser.add_argument("--model-name", help="Optional lmeEEG model name. Defaults to all configured models.")
    parser.add_argument("--epochs-path", help="Optional explicit epochs FIF input path for legacy single-input fitting.")
    parser.add_argument("--metadata-path", help="Optional explicit metadata CSV input path for legacy single-input fitting.")
    parser.add_argument("--all", action="store_true", dest="run_all", help="Process all selected models and optionally write a stage marker.")
    parser.add_argument("--output", help="Optional stage marker path when using --all.")
    parser.add_argument("--trial-regressors-csv", help="Optional trialwise regressor CSV to merge into epoch metadata before lmeEEG fitting.")
    return parser


def _resolve_model_names(config: Mapping[str, Any], args: Namespace) -> list[str]:
    if args.model_name:
        return [str(args.model_name)]
    return list_model_names(config, section=str(getattr(args, "config_section", "lmeeeg")))


def _fit_one_run(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    config_section: str,
    target_name: str | None,
    model_name: str,
    epochs_path: Path | None = None,
    metadata_path: Path | None = None,
    trial_regressors_path: Path | None = None,
) -> dict[str, Any]:
    output_paths = build_lmeeeg_output_paths(
        config,
        row,
        config_section=config_section,
        target_name=target_name,
        model_name=model_name,
        epochs_path=epochs_path,
        metadata_path=metadata_path,
    )
    epochs, metadata_df = load_lmeeeg_inputs_with_trial_regressors(
        output_paths.epochs_input_path,
        output_paths.metadata_input_path,
        trial_regressors_path=trial_regressors_path,
    )
    trial_data = build_lmeeeg_trial_table(epochs, metadata_df, config, config_section=config_section, model_name=model_name)
    result = fit_lmeeeg_mass_model(trial_data, config, config_section=config_section, model_name=model_name)
    summary = save_lmeeeg_result(result, output_paths)
    LOGGER.info(
        "Wrote lmeEEG outputs for %s task=%s run=%s model=%s -> %s",
        row["subject_id"],
        row["task"],
        row["run"],
        model_name,
        output_paths.summary_output_path,
    )
    return {
        "subject_id": str(row["subject_id"]),
        "task": str(row["task"]),
        "run": str(row["run"]),
        "config_section": str(config_section),
        "target": None if target_name is None else str(target_name),
        "model_name": model_name,
        "summary_output": str(output_paths.summary_output_path),
        "n_trials_used": int(summary["n_trials_used"]),
    }


def _fit_one_model(
    config: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    *,
    config_section: str,
    target_name: str | None,
    model_name: str,
    trial_regressors_path: Path | None = None,
) -> dict[str, Any]:
    LOGGER.info("Preparing pooled lmeEEG fit for model=%s", model_name)
    pooled_inputs = load_pooled_lmeeeg_inputs(
        config,
        rows,
        config_section=config_section,
        target_name=target_name,
        model_name=model_name,
        trial_regressors_path=trial_regressors_path,
    )
    LOGGER.info(
        "Starting pooled lmeEEG fit for model=%s with %d pooled trials before model-specific filtering",
        model_name,
        int(pooled_inputs.eeg_data.shape[0]),
    )
    result = fit_pooled_lmeeeg_model(
        config,
        pooled_inputs=pooled_inputs,
        config_section=config_section,
        model_name=model_name,
    )
    output_paths = build_lmeeeg_model_output_paths(config, config_section=config_section, target_name=target_name, model_name=model_name)
    summary = save_lmeeeg_result(result, output_paths)
    LOGGER.info(
        "Wrote pooled lmeEEG outputs for model=%s across %d rows -> %s",
        model_name,
        len(rows),
        output_paths.summary_output_path,
    )
    return {
        "model_name": model_name,
        "config_section": str(config_section),
        "target": None if target_name is None else str(target_name),
        "summary_output": str(output_paths.summary_output_path),
        "n_trials_used": int(summary["n_trials_used"]),
        "n_input_rows": len(rows),
    }


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    try:
        config_section = str(getattr(args, "config_section", "lmeeeg"))
        target_name = getattr(args, "target", None)
        trial_regressors_path = Path(args.trial_regressors_csv) if getattr(args, "trial_regressors_csv", None) else None
        model_names = _resolve_model_names(config, args)
        LOGGER.info("Resolved %d lmeEEG model(s): %s", len(model_names), ", ".join(model_names))
        if args.epochs_path or args.metadata_path:
            if not (args.epochs_path and args.metadata_path):
                raise ValueError("Provide both --epochs-path and --metadata-path when using explicit lmeEEG inputs.")
            row = {
                "subject_id": args.subject_id or "unknown",
                "task": args.task or "unknown",
                "run": args.run or "unknown",
            }
            summaries = [
                _fit_one_run(
                    config,
                    row,
                    config_section=config_section,
                    target_name=target_name,
                    model_name=model_name,
                    epochs_path=Path(args.epochs_path),
                    metadata_path=Path(args.metadata_path),
                        trial_regressors_path=trial_regressors_path,
                )
                for model_name in model_names
            ]
        else:
            rows = select_lmeeeg_rows(
                config,
                dataset_index=args.dataset_index,
                subject_id=args.subject_id,
                task=args.task,
                run=args.run,
            )
            LOGGER.info("Running pooled lmeEEG fits across %d selected dataset rows", len(rows))
            summaries = [
                _fit_one_model(
                    config,
                    rows,
                    config_section=config_section,
                    target_name=target_name,
                    model_name=model_name,
                    trial_regressors_path=trial_regressors_path,
                )
                for model_name in model_names
            ]

        if args.run_all and args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps({"status": "ok", "processed_runs": summaries}, indent=2) + "\n", encoding="utf-8")
            LOGGER.info("Wrote lmeEEG stage marker: %s", args.output)
            return 0

        if args.epochs_path and args.metadata_path:
            return 0

        if not args.run_all and len(summaries) == 0:
            raise ValueError("No lmeEEG models were processed.")
        if args.run_all:
            return 0

        if len(model_names) == 1:
            return 0
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0
