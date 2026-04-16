"""`lmeeeg-group-stats` CLI subcommand."""

from __future__ import annotations

import csv
import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from lmeeeg import permute_fixed_effect
from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.design import get_predictor_column_indices, list_model_names
from rate.lmeeeg.mass_model import fit_lmeeeg_backend_result
from rate.lmeeeg.pooling import load_pooled_lmeeeg_trial_data, select_lmeeeg_rows
from rate.lmeeeg.targets import resolve_target_names, section_supports_targets
from rate.paths import ensure_parent_dir, resolve_out_dir

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser("lmeeeg-group-stats", help="Run lmeEEG permutation inference over fitted effects.")
    add_common_config_arguments(parser)
    parser.add_argument("--stats-config", default="config/stats.yaml", help="Path to the stats config file.")
    parser.add_argument("--dataset-index", help="Optional explicit dataset index parquet path.")
    parser.add_argument("--subject-id", help="Optional subject identifier filter.")
    parser.add_argument("--task", help="Optional task filter.")
    parser.add_argument("--run", help="Optional run filter.")
    parser.add_argument("--config-section", default="lmeeeg", help="Config section to analyse.")
    parser.add_argument("--target", help="Optional target name when running a target-specific lmeEEG section.")
    parser.add_argument("--model-name", help="Optional lmeEEG model name filter.")
    parser.add_argument("--trial-regressors-csv", help="Optional trialwise regressor CSV to merge into epoch metadata before inference refits.")
    parser.add_argument("--output", help="Optional explicit output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    stats_cfg = _load_stats_config(Path(args.stats_config))
    config_section = str(getattr(args, "config_section", "lmeeeg") or "lmeeeg")
    output_filename = "lmeeeg_group_stats.json" if config_section == "lmeeeg" else f"{config_section}_group_stats.json"
    output_path = Path(args.output) if args.output else resolve_out_dir(config) / "stats" / output_filename
    trial_regressors_path = Path(args.trial_regressors_csv) if getattr(args, "trial_regressors_csv", None) else None

    try:
        rows = select_lmeeeg_rows(
            config,
            dataset_index=args.dataset_index,
            subject_id=args.subject_id,
            task=args.task,
            run=args.run,
        )
        model_names = [str(args.model_name)] if args.model_name else list_model_names(config, section=config_section)
        target_names = resolve_target_names(config, config_section=config_section, target_name=getattr(args, "target", None))
        LOGGER.info(
            "Running lmeEEG inference for section=%s across %d model(s), %d target(s), and %d selected dataset rows",
            config_section,
            len(model_names),
            len(target_names),
            len(rows),
        )
        analyses = [
            _run_one_analysis(
                config,
                stats_cfg,
                rows,
                config_section=config_section,
                target_name=target_name,
                model_name=model_name,
                trial_regressors_path=trial_regressors_path,
            )
            for target_name in target_names
            for model_name in model_names
        ]
    except (FileNotFoundError, ValueError, KeyError) as exc:
        LOGGER.error("%s", exc)
        return 1

    summary = {
        "status": "ok",
        "stage": f"{config_section}_group_stats",
        "config_section": config_section,
        "method": _resolve_correction_name(stats_cfg),
        "n_permutations": int((stats_cfg.get("stats") or {}).get("n_permutations", 1024)),
        "n_analyses": len(analyses),
        "analyses": analyses,
    }
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Wrote lmeEEG inference summary: %s", output_path)
    return 0


def _load_stats_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Stats config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Stats config must be a mapping: {path}")
    return payload


def _run_one_analysis(
    config: Mapping[str, Any],
    stats_cfg: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    *,
    config_section: str = "lmeeeg",
    target_name: str | None = None,
    model_name: str,
    trial_regressors_path: Path | None = None,
) -> dict[str, Any]:
    LOGGER.info(
        "Preparing pooled lmeEEG inference inputs for section=%s target=%s model=%s",
        config_section,
        target_name or "<none>",
        model_name,
    )
    trial_data = load_pooled_lmeeeg_trial_data(
        config,
        rows,
        config_section=config_section,
        target_name=target_name,
        model_name=model_name,
        trial_regressors_path=trial_regressors_path,
    )
    LOGGER.info(
        "Refitting pooled lmeEEG section=%s target=%s model=%s for inference with %d filtered trials",
        config_section,
        target_name or "<none>",
        model_name,
        int(trial_data.eeg_data.shape[0]),
    )
    fit_result = fit_lmeeeg_backend_result(
        trial_data,
        config,
        config_section=config_section,
        model_name=model_name,
    )

    correction = _resolve_correction_name(stats_cfg)
    tail = int((stats_cfg.get("stats") or {}).get("tail", 0))
    n_permutations = int((stats_cfg.get("stats") or {}).get("n_permutations", 1024))
    threshold = _resolve_threshold(stats_cfg, n_observations=int(trial_data.eeg_data.shape[0]), correction=correction, tail=tail)
    adjacency = _resolve_adjacency(config, stats_cfg, trial_data.channel_names)
    effect_specs = _resolve_effects(trial_data)

    outputs: list[dict[str, Any]] = []
    for predictor_name, effect_name in effect_specs:
        LOGGER.info(
            "Running lmeEEG %s inference for model=%s predictor=%s effect=%s with %d permutations",
            correction,
            model_name,
            predictor_name,
            effect_name,
            n_permutations,
        )
        inference = permute_fixed_effect(
            fit_result=fit_result,
            effect=effect_name,
            correction=correction,
            n_permutations=n_permutations,
            seed=int((stats_cfg.get("stats") or {}).get("seed", 0)),
            tail=tail,
            threshold=threshold,
            adjacency=adjacency,
        )
        outputs.append(
            _write_inference_outputs(
                config,
                config_section=config_section,
                target_name=target_name,
                model_name=model_name,
                predictor_name=predictor_name,
                effect_name=effect_name,
                inference=inference,
                channel_names=trial_data.channel_names,
                times=trial_data.times,
            )
        )

    analysis_summary = {
        "model_name": model_name,
        "n_input_rows": len(rows),
        "n_effects": len(outputs),
        "effects": outputs,
    }
    if target_name is not None:
        analysis_summary["target"] = target_name
    return analysis_summary


def _resolve_effects(trial_data) -> list[tuple[str, str]]:
    effects: list[tuple[str, str]] = []
    for predictor_name in trial_data.design_result.test_predictors:
        for column_index in get_predictor_column_indices(
            predictor_name,
            column_names=trial_data.design_result.column_names,
            term_name_slices=trial_data.design_result.term_name_slices,
        ):
            effect_name = str(trial_data.design_result.column_names[column_index])
            effects.append((predictor_name, effect_name))

    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in effects:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    if not deduped:
        raise ValueError("No lmeEEG test predictors resolved to concrete fixed-effect columns for inference.")
    return deduped


def _resolve_correction_name(stats_cfg: Mapping[str, Any]) -> str:
    method = str((stats_cfg.get("stats") or {}).get("method", "mne_cluster")).lower()
    if method in {"mne_cluster", "cluster"}:
        return "cluster"
    if method in {"mne_tfce", "tfce"}:
        return "tfce"
    if method == "maxstat":
        return "maxstat"
    raise ValueError(f"Unsupported lmeEEG stats method `{method}`.")


def _resolve_threshold(
    stats_cfg: Mapping[str, Any],
    *,
    n_observations: int,
    correction: str,
    tail: int,
):
    section = stats_cfg.get("stats") or {}
    if correction == "cluster":
        if "cluster_threshold" in section:
            return float(section["cluster_threshold"])
        alpha = section.get("cluster_forming_alpha")
        if alpha is None:
            return None
        from scipy.stats import t

        alpha_value = float(alpha)
        dof = max(1, n_observations - 1)
        if tail == 0:
            return float(t.ppf(1.0 - (alpha_value / 2.0), df=dof))
        return float(t.ppf(1.0 - alpha_value, df=dof))

    if correction == "tfce":
        threshold = section.get("tfce_threshold")
        if threshold is None:
            return None
        if isinstance(threshold, Mapping):
            return {str(key): float(value) for key, value in threshold.items()}
        raise ValueError("TFCE threshold must be a mapping like {start: ..., step: ...}.")

    return None


def _resolve_adjacency(
    config: Mapping[str, Any],
    stats_cfg: Mapping[str, Any],
    channel_names: list[str],
):
    adjacency_cfg = (stats_cfg.get("stats") or {}).get("adjacency")
    if adjacency_cfg in (None, "none", "None"):
        return None
    if adjacency_cfg != "mne_default":
        raise ValueError(f"Unsupported lmeEEG adjacency setting `{adjacency_cfg}`.")

    import mne

    info = mne.create_info(channel_names, sfreq=1.0, ch_types=["eeg"] * len(channel_names))
    try:
        montage = mne.channels.make_standard_montage(_resolve_preprocessing_montage_name(config))
        info.set_montage(montage, on_missing="ignore")
        adjacency, _ = mne.channels.find_ch_adjacency(info, ch_type="eeg")
        return adjacency
    except Exception as exc:
        LOGGER.warning(
            "Unable to construct EEG adjacency from channel locations; falling back to no adjacency: %s",
            exc,
        )
        return None


def _resolve_preprocessing_montage_name(config: Mapping[str, Any]) -> str:
    preprocessing_cfg = dict(config.get("preprocessing") or {})
    montage_cfg = preprocessing_cfg.get("montage", preprocessing_cfg.get("montage_name"))
    if isinstance(montage_cfg, Mapping):
        return str(montage_cfg.get("name", "standard_1020"))
    return str(montage_cfg or "standard_1020")


def _write_inference_outputs(
    config: Mapping[str, Any],
    *,
    config_section: str = "lmeeeg",
    target_name: str | None = None,
    model_name: str,
    predictor_name: str,
    effect_name: str,
    inference,
    channel_names: list[str],
    times: np.ndarray,
) -> dict[str, Any]:
    out_dir = resolve_out_dir(config)
    if section_supports_targets(config, config_section=config_section):
        if target_name is None:
            raise ValueError("`target_name` is required when writing target-specific lmeEEG inference outputs.")
        effect_dir = out_dir / "stats" / str(config_section) / _safe_token(target_name) / str(model_name) / _safe_token(effect_name)
    else:
        effect_dir = out_dir / "stats" / str(config_section) / str(model_name) / _safe_token(effect_name)
    observed_path = effect_dir / "observed_statistic.npy"
    corrected_p_path = effect_dir / "corrected_p_values.npy"
    null_path = effect_dir / "null_distribution.npy"
    cluster_p_path = effect_dir / "cluster_p_values.npy"
    p_values_csv_path = effect_dir / "p_values.csv"
    metadata_path = effect_dir / "summary.json"

    ensure_parent_dir(metadata_path)
    np.save(observed_path, np.asarray(inference.observed_statistic, dtype=float))
    np.save(corrected_p_path, np.asarray(inference.corrected_p_values, dtype=float))
    np.save(null_path, np.asarray(inference.null_distribution, dtype=float))
    np.save(cluster_p_path, np.asarray([] if inference.cluster_p_values is None else inference.cluster_p_values, dtype=float))
    csv_columns = _write_p_values_csv(
        p_values_csv_path,
        inference=inference,
        channel_names=channel_names,
        times=np.asarray(times, dtype=float),
    )

    significant_mask = np.asarray(inference.corrected_p_values, dtype=float) < 0.05
    summary = {
        "status": "ok",
        "predictor": predictor_name,
        "effect": effect_name,
        "correction": str(inference.correction),
        "backend_metadata": dict(inference.backend_metadata),
        "observed_shape": list(np.asarray(inference.observed_statistic).shape),
        "corrected_p_shape": list(np.asarray(inference.corrected_p_values).shape),
        "n_significant_p_lt_0_05": int(np.count_nonzero(significant_mask)),
        "min_corrected_p": float(np.nanmin(np.asarray(inference.corrected_p_values, dtype=float))),
        "observed_statistic": str(observed_path),
        "corrected_p_values": str(corrected_p_path),
        "null_distribution": str(null_path),
        "cluster_p_values": str(cluster_p_path),
    }
    if target_name is not None:
        summary["target"] = target_name
    if csv_columns is not None:
        summary["p_values_csv"] = str(p_values_csv_path)
        summary["p_values_csv_columns"] = csv_columns
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _safe_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value) or "effect"


def _write_p_values_csv(
    path: Path,
    *,
    inference,
    channel_names: list[str],
    times: np.ndarray,
) -> list[str] | None:
    correction = str(getattr(inference, "correction", "")).lower()
    if correction == "cluster":
        fieldnames = ["p_values", "cluster_size", "duration", "spatial_extent"]
        rows = _build_cluster_rows(inference, channel_names=channel_names, times=times)
    elif correction in {"tfce", "maxstat"}:
        fieldnames = ["p_values", "channel", "time"]
        rows = _build_pointwise_rows(inference, channel_names=channel_names, times=times)
    else:
        return None

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return fieldnames


def _build_cluster_rows(
    inference,
    *,
    channel_names: list[str],
    times: np.ndarray,
) -> list[dict[str, float | int]]:
    cluster_p_values = np.asarray([] if inference.cluster_p_values is None else inference.cluster_p_values, dtype=float)
    clusters = list(getattr(inference, "clusters", []) or [])
    rows: list[dict[str, float | int]] = []
    for cluster_mask, cluster_p_value in zip(clusters, cluster_p_values, strict=False):
        if not np.isfinite(cluster_p_value):
            continue
        time_channel_mask = _normalize_cluster_mask(cluster_mask, n_channels=len(channel_names), n_times=int(times.shape[0]))
        time_indices, channel_indices = np.where(time_channel_mask)
        if time_indices.size == 0:
            continue
        rows.append(
            {
                "p_values": float(cluster_p_value),
                "cluster_size": int(np.count_nonzero(time_channel_mask)),
                "duration": _cluster_duration(times[np.unique(time_indices)], all_times=times),
                "spatial_extent": int(np.unique(channel_indices).size),
            }
        )
    return sorted(rows, key=lambda row: (float(row["p_values"]), -int(row["cluster_size"]), -int(row["spatial_extent"])))


def _build_pointwise_rows(
    inference,
    *,
    channel_names: list[str],
    times: np.ndarray,
) -> list[dict[str, float | str]]:
    corrected_p_values = np.asarray(inference.corrected_p_values, dtype=float)
    if corrected_p_values.ndim != 2:
        raise ValueError("Expected corrected_p_values to have shape (n_channels, n_times).")
    if corrected_p_values.shape != (len(channel_names), int(times.shape[0])):
        raise ValueError(
            "corrected_p_values shape does not match provided channel/time axes: "
            f"{corrected_p_values.shape} != {(len(channel_names), int(times.shape[0]))}"
        )

    rows: list[dict[str, float | str]] = []
    point_indices = np.argwhere(np.isfinite(corrected_p_values))
    for channel_index, time_index in point_indices:
        rows.append(
            {
                "p_values": float(corrected_p_values[channel_index, time_index]),
                "channel": str(channel_names[channel_index]),
                "time": float(times[time_index]),
            }
        )
    return sorted(rows, key=lambda row: (float(row["p_values"]), str(row["channel"]), float(row["time"])))


def _normalize_cluster_mask(cluster_mask: Any, *, n_channels: int, n_times: int) -> np.ndarray:
    mask = np.asarray(cluster_mask, dtype=bool)
    if mask.shape == (n_times, n_channels):
        return mask
    if mask.shape == (n_channels, n_times):
        return mask.T
    raise ValueError(f"Unexpected cluster mask shape {mask.shape}; expected {(n_times, n_channels)} or {(n_channels, n_times)}.")


def _cluster_duration(cluster_times: np.ndarray, *, all_times: np.ndarray) -> float:
    selected = np.sort(np.asarray(cluster_times, dtype=float))
    if selected.size == 0:
        return 0.0
    if all_times.size < 2:
        return 0.0
    sample_step = float(np.median(np.diff(np.sort(np.asarray(all_times, dtype=float)))))
    if selected.size == 1:
        return abs(sample_step)
    return float((selected[-1] - selected[0]) + abs(sample_step))
