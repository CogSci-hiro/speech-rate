"""`lmeeeg-nested-group-stats` CLI subcommand."""

from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.cli.commands.lmeeeg_group_stats import _resolve_adjacency, _safe_token, _write_p_values_csv
from rate.lmeeeg.mass_model import fit_lmeeeg_backend_result
from rate.lmeeeg.nested import (
    DEFAULT_LMEEEG_NESTED_SECTION,
    LMEEEGNestedAnalysisSpec,
    build_lmeeeg_nested_group_output_paths,
    resolve_lmeeeg_nested_settings,
)
from rate.lmeeeg.pooling import load_pooled_lmeeeg_trial_data, select_lmeeeg_rows
from rate.lmeeeg.targets import resolve_target_names, section_supports_targets
from rate.paths import ensure_parent_dir

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "lmeeeg-nested-group-stats",
        help="Run nested-block lmeEEG permutation inference from a pooled full model.",
    )
    add_common_config_arguments(parser)
    parser.add_argument("--dataset-index", help="Optional explicit dataset index parquet path.")
    parser.add_argument("--subject-id", help="Optional subject identifier filter.")
    parser.add_argument("--task", help="Optional task filter.")
    parser.add_argument("--run", help="Optional run filter.")
    parser.add_argument(
        "--config-section",
        default=DEFAULT_LMEEEG_NESTED_SECTION,
        help="Config section describing the nested-block lmeEEG analyses.",
    )
    parser.add_argument("--target", help="Optional target name for target-specific analyses.")
    parser.add_argument("--trial-regressors-csv", help="Optional trialwise regressor CSV to merge into epoch metadata.")
    parser.add_argument("--output", help="Optional explicit summary output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    try:
        _resolve_permute_fixed_block()
        config_section = str(getattr(args, "config_section", DEFAULT_LMEEEG_NESTED_SECTION) or DEFAULT_LMEEEG_NESTED_SECTION)
        settings = resolve_lmeeeg_nested_settings(config, config_section=config_section)
        output_paths = build_lmeeeg_nested_group_output_paths(config, config_section=config_section)
        output_path = Path(args.output) if args.output else output_paths.summary_output_path
        trial_regressors_path = Path(args.trial_regressors_csv) if getattr(args, "trial_regressors_csv", None) else None

        rows = select_lmeeeg_rows(
            config,
            dataset_index=args.dataset_index,
            subject_id=args.subject_id,
            task=args.task,
            run=args.run,
        )
        target_names = _resolve_nested_target_names(
            config,
            config_section=config_section,
            source_section=settings.source_section,
            requested_target=getattr(args, "target", None),
        )
        LOGGER.info(
            "Running nested-block lmeEEG inference for section=%s across %d target(s) and %d dataset rows",
            config_section,
            len(target_names),
            len(rows),
        )

        analyses = [
            _run_target_analyses(
                config,
                rows,
                settings=settings,
                config_section=config_section,
                target_name=target_name,
                trial_regressors_path=trial_regressors_path,
            )
            for target_name in target_names
        ]
    except (FileNotFoundError, ValueError, KeyError, ImportError) as exc:
        LOGGER.error("%s", exc)
        return 1

    summary = {
        "status": "ok",
        "stage": f"{config_section}_group_stats",
        "config_section": config_section,
        "source_section": settings.source_section,
        "method": settings.correction,
        "n_permutations": settings.n_permutations,
        "n_targets": len(analyses),
        "n_nested_tests": sum(len(item["nested_tests"]) for item in analyses),
        "analyses": analyses,
    }
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Wrote nested-block lmeEEG inference summary: %s", output_path)
    return 0


def _run_target_analyses(
    config: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    *,
    settings,
    config_section: str,
    target_name: str | None,
    trial_regressors_path: Path | None,
) -> dict[str, Any]:
    fit_cache: dict[str, tuple[Any, list[str], np.ndarray, int]] = {}
    nested_tests: list[dict[str, Any]] = []
    max_trials_used = 0
    for analysis in settings.analyses:
        cached = fit_cache.get(analysis.full_model)
        if cached is None:
            LOGGER.info(
                "Preparing pooled nested-block lmeEEG inputs for target=%s full_model=%s",
                target_name or "<none>",
                analysis.full_model,
            )
            trial_data = load_pooled_lmeeeg_trial_data(
                config,
                rows,
                config_section=settings.source_section,
                target_name=target_name,
                model_name=analysis.full_model,
                trial_regressors_path=trial_regressors_path,
            )
            fit_result = fit_lmeeeg_backend_result(
                trial_data,
                config,
                config_section=settings.source_section,
                model_name=analysis.full_model,
            )
            cached = (fit_result, trial_data.channel_names, trial_data.times, int(trial_data.eeg_data.shape[0]))
            fit_cache[analysis.full_model] = cached
        fit_result, channel_names, times, n_trials_used = cached
        max_trials_used = max(max_trials_used, n_trials_used)
        adjacency = _resolve_adjacency(config, {"stats": {"adjacency": settings.adjacency}}, channel_names)
        nested_tests.append(
            _run_nested_test(
                config,
                settings=settings,
                config_section=config_section,
                target_name=target_name,
                analysis=analysis,
                fit_result=fit_result,
                channel_names=channel_names,
                times=times,
                adjacency=adjacency,
            )
        )

    summary = {
        "full_models": sorted(fit_cache),
        "n_input_rows": len(rows),
        "n_trials_used": max_trials_used,
        "n_nested_tests": len(nested_tests),
        "nested_tests": nested_tests,
    }
    if target_name is not None:
        summary["target"] = target_name
    return summary


def _run_nested_test(
    config: Mapping[str, Any],
    *,
    settings,
    config_section: str,
    target_name: str | None,
    analysis: LMEEEGNestedAnalysisSpec,
    fit_result,
    channel_names: list[str],
    times: np.ndarray,
    adjacency,
) -> dict[str, Any]:
    permute_fixed_block = _resolve_permute_fixed_block()
    reduced_formula = _normalise_reduced_formula(analysis.reduced_formula)
    kwargs: dict[str, Any] = {
        "fit_result": fit_result,
        "reduced_formula": reduced_formula,
        "correction": settings.correction,
        "n_permutations": settings.n_permutations,
        "seed": settings.seed,
        "tail": settings.tail,
        "verbose": "info",
    }
    if settings.threshold is not None:
        kwargs["threshold"] = settings.threshold
    if adjacency is not None:
        kwargs["adjacency"] = adjacency
    if settings.tfce_h_power is not None:
        kwargs["tfce_h_power"] = settings.tfce_h_power
    if settings.tfce_e_power is not None:
        kwargs["tfce_e_power"] = settings.tfce_e_power

    LOGGER.info(
        "Running nested-block lmeEEG %s inference for analysis=%s reduced=%s with %d permutations",
        settings.correction,
        analysis.name,
        reduced_formula,
        settings.n_permutations,
    )
    inference = permute_fixed_block(**kwargs)
    return _write_nested_inference_outputs(
        config,
        config_section=config_section,
        target_name=target_name,
        analysis=analysis,
        reduced_formula=reduced_formula,
        inference=inference,
        channel_names=channel_names,
        times=np.asarray(times, dtype=float),
    )


def _resolve_permute_fixed_block():
    try:
        import lmeeeg.api.infer as infer_module
    except ImportError as exc:
        raise ImportError(
            "Nested-block lmeEEG inference requires `lmeeeg.api.infer.permute_fixed_block`. "
            "Install lmeeeg>=0.4.0 or run against the updated working tree copy."
        ) from exc
    permute_fixed_block = getattr(infer_module, "permute_fixed_block", None)
    if not callable(permute_fixed_block):
        raise ImportError(
            "Installed `lmeeeg` does not expose `permute_fixed_block` "
            f"(loaded from {getattr(infer_module, '__file__', '<unknown>')}). "
            "Install lmeeeg>=0.4.0 or run against the updated working tree copy."
        )
    return permute_fixed_block


def _resolve_nested_target_names(
    config: Mapping[str, Any],
    *,
    config_section: str,
    source_section: str,
    requested_target: str | None,
) -> list[str | None]:
    if section_supports_targets(config, config_section=config_section):
        return resolve_target_names(config, config_section=config_section, target_name=requested_target)
    if section_supports_targets(config, config_section=source_section):
        return resolve_target_names(config, config_section=source_section, target_name=requested_target)
    return [None]


def _normalise_reduced_formula(reduced_formula: str | list[str]) -> str | list[str]:
    if isinstance(reduced_formula, list):
        return [str(value) for value in reduced_formula]
    text = str(reduced_formula).strip()
    if not text:
        raise ValueError("Nested-block reduced formula must not be empty.")
    if "~" in text:
        lhs, rhs = text.split("~", maxsplit=1)
        if lhs.strip():
            return text
        return f"y ~ {rhs.strip()}"
    return f"y ~ {text}"


def _write_nested_inference_outputs(
    config: Mapping[str, Any],
    *,
    config_section: str,
    target_name: str | None,
    analysis: LMEEEGNestedAnalysisSpec,
    reduced_formula: str | list[str],
    inference,
    channel_names: list[str],
    times: np.ndarray,
) -> dict[str, Any]:
    output_paths = build_lmeeeg_nested_group_output_paths(config, config_section=config_section)
    root_dir = output_paths.root_dir
    if target_name is not None:
        effect_dir = root_dir / _safe_token(target_name) / _safe_token(analysis.name)
    else:
        effect_dir = root_dir / _safe_token(analysis.name)

    observed_path = effect_dir / "observed_statistic.npy"
    permutation_observed_path = effect_dir / "permutation_observed_statistic.npy"
    corrected_p_path = effect_dir / "corrected_p_values.npy"
    null_path = effect_dir / "null_distribution.npy"
    cluster_p_path = effect_dir / "cluster_p_values.npy"
    p_values_csv_path = effect_dir / "p_values.csv"
    metadata_path = effect_dir / "summary.json"

    observed_map = np.asarray(inference.observed_statistic, dtype=float)
    expected_shape = (len(channel_names), int(times.shape[0]))
    if observed_map.shape != expected_shape:
        raise ValueError(
            "Nested-block observed_statistic shape does not match provided channel/time axes: "
            f"{observed_map.shape} != {expected_shape}"
        )

    ensure_parent_dir(metadata_path)
    np.save(observed_path, observed_map)
    np.save(permutation_observed_path, observed_map)
    np.save(corrected_p_path, np.asarray(inference.corrected_p_values, dtype=float))
    np.save(null_path, np.asarray(inference.null_distribution, dtype=float))
    csv_columns = _write_p_values_csv(
        p_values_csv_path,
        inference=inference,
        channel_names=channel_names,
        times=times,
    )

    corrected_p_values = np.asarray(inference.corrected_p_values, dtype=float)
    significant_mask = corrected_p_values < 0.05
    backend_metadata = dict(getattr(inference, "backend_metadata", {}) or {})
    statistic_kind = str(backend_metadata.get("statistic", "f_value"))
    cluster_p_values = getattr(inference, "cluster_p_values", None)
    summary = {
        "status": "ok",
        "analysis_name": analysis.name,
        "full_model": analysis.full_model,
        "reduced_formula": reduced_formula,
        "tested_predictor": analysis.tested_predictor,
        "correction": str(inference.correction),
        "backend_metadata": backend_metadata,
        "observed_shape": list(observed_map.shape),
        "corrected_p_shape": list(corrected_p_values.shape),
        "n_significant_p_lt_0_05": int(np.count_nonzero(significant_mask)),
        "min_corrected_p": float(np.nanmin(corrected_p_values)),
        "observed_statistic": str(observed_path),
        "observed_statistic_kind": statistic_kind,
        "permutation_observed_statistic": str(permutation_observed_path),
        "permutation_observed_statistic_kind": statistic_kind,
        "corrected_p_values": str(corrected_p_path),
        "null_distribution": str(null_path),
    }
    if cluster_p_values is not None:
        np.save(cluster_p_path, np.asarray(cluster_p_values, dtype=float))
        summary["cluster_p_values"] = str(cluster_p_path)
    if target_name is not None:
        summary["target"] = target_name
    if csv_columns is not None:
        summary["p_values_csv"] = str(p_values_csv_path)
        summary["p_values_csv_columns"] = csv_columns
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
