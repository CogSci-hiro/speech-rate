"""`glm-group-stats` CLI subcommand."""

from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.design import list_model_names
from rate.paths import ensure_parent_dir, resolve_out_dir
from rate.stats import run_cluster_permutation

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser("glm-group-stats", help="Run group-level GLM beta inference against zero.")
    add_common_config_arguments(parser)
    parser.add_argument("--stats-config", default="config/stats.yaml", help="Path to the stats config file.")
    parser.add_argument("--model-name", help="Optional GLM model name filter.")
    parser.add_argument("--output", help="Optional explicit output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    stats_cfg = _load_stats_config(Path(args.stats_config))
    output_path = _resolve_summary_output_path(config, stats_cfg, explicit_output=args.output)

    try:
        model_names = [str(args.model_name)] if args.model_name else list_model_names(config, section="glm")
        analyses = [_run_one_analysis(config, stats_cfg, model_name=model_name) for model_name in model_names]
    except (FileNotFoundError, ValueError, KeyError) as exc:
        LOGGER.error("%s", exc)
        return 1

    summary = {
        "status": "ok",
        "stage": "glm_group_stats",
        "method": _resolve_correction_name(stats_cfg),
        "n_permutations": int((stats_cfg.get("stats") or {}).get("n_permutations", 1024)),
        "n_analyses": len(analyses),
        "analyses": analyses,
    }
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Wrote GLM group-stats summary: %s", output_path)
    return 0


def _load_stats_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Stats config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Stats config must be a mapping: {path}")
    return payload


def _resolve_summary_output_path(
    config: Mapping[str, Any],
    stats_cfg: Mapping[str, Any],
    *,
    explicit_output: str | None,
) -> Path:
    if explicit_output:
        return Path(explicit_output)
    relative_output = ((stats_cfg.get("stats") or {}).get("outputs") or {}).get("glm", "stats/glm_group_stats.json")
    return resolve_out_dir(config) / str(relative_output)


def _run_one_analysis(
    config: Mapping[str, Any],
    stats_cfg: Mapping[str, Any],
    *,
    model_name: str,
) -> dict[str, Any]:
    subject_effects = _load_subject_effects(config, model_name=model_name)
    correction = _resolve_correction_name(stats_cfg)
    tail = int((stats_cfg.get("stats") or {}).get("tail", 0))
    n_permutations = int((stats_cfg.get("stats") or {}).get("n_permutations", 1024))
    threshold = _resolve_threshold(
        stats_cfg,
        n_observations=min(effect["n_subjects"] for effect in subject_effects.values()),
        correction=correction,
        tail=tail,
    )
    adjacency = _resolve_adjacency(config, stats_cfg, subject_effects)

    outputs: list[dict[str, Any]] = []
    for effect_name, effect_payload in subject_effects.items():
        LOGGER.info(
            "Running GLM %s inference for model=%s effect=%s with %d subjects and %d permutations",
            correction,
            model_name,
            effect_name,
            int(effect_payload["n_subjects"]),
            n_permutations,
        )
        threshold_for_effect = _resolve_threshold(
            stats_cfg,
            n_observations=int(effect_payload["n_subjects"]),
            correction=correction,
            tail=tail,
        )
        inference = run_cluster_permutation(
            np.asarray(effect_payload["beta_maps"], dtype=float),
            correction=correction,
            n_permutations=n_permutations,
            tail=tail,
            threshold=threshold_for_effect,
            adjacency=adjacency,
            seed=int((stats_cfg.get("stats") or {}).get("seed", 0)),
        )
        outputs.append(
            _write_inference_outputs(
                config,
                model_name=model_name,
                predictor_name=str(effect_payload["predictor_name"]),
                effect_name=effect_name,
                subject_ids=list(effect_payload["subject_ids"]),
                inference=inference,
            )
        )

    return {
        "model_name": model_name,
        "n_effects": len(outputs),
        "effects": outputs,
        "threshold": _serialize_threshold(threshold),
    }


def _load_subject_effects(config: Mapping[str, Any], *, model_name: str) -> dict[str, dict[str, Any]]:
    model_root = resolve_out_dir(config) / "glm"
    if not model_root.exists():
        raise FileNotFoundError(f"No GLM outputs found under: {model_root}")

    collected: dict[str, dict[str, Any]] = {}
    reference_times: np.ndarray | None = None
    reference_channels: list[str] | None = None

    for subject_dir in sorted(path for path in model_root.iterdir() if path.is_dir()):
        result_dir = subject_dir / str(model_name)
        predictor_betas_path = result_dir / "predictor_betas.npz"
        if not predictor_betas_path.exists():
            continue

        summary_path = result_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if str(summary.get("status", "")).lower() != "ok":
                LOGGER.info("Skipping GLM group stats for %s model=%s because subject fit status is %s", subject_dir.name, model_name, summary.get("status"))
                continue

        times = np.asarray(np.load(result_dir / "times.npy"), dtype=float)
        channel_names = json.loads((result_dir / "channel_names.json").read_text(encoding="utf-8"))
        predictor_columns = _load_predictor_columns(result_dir / "predictor_columns.json")
        if reference_times is None:
            reference_times = times
        elif not np.allclose(reference_times, times):
            raise ValueError(f"All subject GLM time axes must match for model `{model_name}`.")

        if reference_channels is None:
            reference_channels = list(channel_names)
        elif list(reference_channels) != list(channel_names):
            raise ValueError(f"All subject GLM channel orders must match for model `{model_name}`.")

        predictor_betas = np.load(predictor_betas_path)
        try:
            for effect_name in predictor_betas.files:
                beta_map = np.asarray(predictor_betas[effect_name], dtype=float)
                if beta_map.ndim != 2:
                    raise ValueError(
                        f"Expected beta map `{effect_name}` for model `{model_name}` to have shape (channels, times)."
                    )
                if not np.isfinite(beta_map).all():
                    LOGGER.warning(
                        "Skipping subject %s effect=%s for model=%s because beta map contains non-finite values.",
                        subject_dir.name,
                        effect_name,
                        model_name,
                    )
                    continue
                predictor_name = predictor_columns.get(effect_name, effect_name)
                effect_store = collected.setdefault(
                    effect_name,
                    {
                        "predictor_name": predictor_name,
                        "subject_ids": [],
                        "beta_maps": [],
                    },
                )
                effect_store["subject_ids"].append(subject_dir.name)
                effect_store["beta_maps"].append(beta_map)
        finally:
            predictor_betas.close()

    if not collected:
        raise ValueError(f"No usable subject-level GLM beta maps found for model `{model_name}`.")

    normalized: dict[str, dict[str, Any]] = {}
    for effect_name, payload in collected.items():
        beta_maps = np.asarray(payload["beta_maps"], dtype=float)
        n_subjects = int(beta_maps.shape[0])
        if n_subjects < 2:
            LOGGER.warning(
                "Skipping group inference for model=%s effect=%s because fewer than two valid subjects remain.",
                model_name,
                effect_name,
            )
            continue
        normalized[effect_name] = {
            **payload,
            "beta_maps": beta_maps,
            "n_subjects": n_subjects,
        }

    if not normalized:
        raise ValueError(f"No GLM effects had at least two valid subject beta maps for model `{model_name}`.")
    return normalized


def _load_predictor_columns(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}
    effect_to_predictor: dict[str, str] = {}
    for predictor_name, effect_names in payload.items():
        if not isinstance(effect_names, list):
            continue
        for effect_name in effect_names:
            effect_to_predictor[str(effect_name)] = str(predictor_name)
    return effect_to_predictor


def _resolve_correction_name(stats_cfg: Mapping[str, Any]) -> str:
    method = str((stats_cfg.get("stats") or {}).get("method", "mne_cluster")).lower()
    if method in {"mne_cluster", "cluster"}:
        return "cluster"
    if method in {"mne_tfce", "tfce"}:
        return "tfce"
    if method == "maxstat":
        return "maxstat"
    raise ValueError(f"Unsupported GLM stats method `{method}`.")


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
    subject_effects: Mapping[str, Mapping[str, Any]],
):
    adjacency_cfg = (stats_cfg.get("stats") or {}).get("adjacency")
    if adjacency_cfg in (None, "none", "None"):
        return None
    if adjacency_cfg != "mne_default":
        raise ValueError(f"Unsupported GLM adjacency setting `{adjacency_cfg}`.")

    first_effect = next(iter(subject_effects.values()))
    first_map = np.asarray(first_effect["beta_maps"], dtype=float)
    n_channels = int(first_map.shape[1])
    channel_names = _load_reference_channel_names(config, n_channels=n_channels)

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


def _load_reference_channel_names(config: Mapping[str, Any], *, n_channels: int) -> list[str]:
    glm_root = resolve_out_dir(config) / "glm"
    for subject_dir in sorted(path for path in glm_root.iterdir() if path.is_dir()):
        for model_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            channel_path = model_dir / "channel_names.json"
            if channel_path.exists():
                channel_names = json.loads(channel_path.read_text(encoding="utf-8"))
                if len(channel_names) == n_channels:
                    return [str(name) for name in channel_names]
    raise ValueError("Unable to locate a subject-level GLM channel_names.json file for adjacency construction.")


def _resolve_preprocessing_montage_name(config: Mapping[str, Any]) -> str:
    preprocessing_cfg = dict(config.get("preprocessing") or {})
    montage_cfg = preprocessing_cfg.get("montage", preprocessing_cfg.get("montage_name"))
    if isinstance(montage_cfg, Mapping):
        return str(montage_cfg.get("name", "standard_1020"))
    return str(montage_cfg or "standard_1020")


def _write_inference_outputs(
    config: Mapping[str, Any],
    *,
    model_name: str,
    predictor_name: str,
    effect_name: str,
    subject_ids: list[str],
    inference,
) -> dict[str, Any]:
    out_dir = resolve_out_dir(config)
    effect_dir = out_dir / "stats" / "glm" / str(model_name) / _safe_token(effect_name)
    observed_path = effect_dir / "observed_statistic.npy"
    corrected_p_path = effect_dir / "corrected_p_values.npy"
    null_path = effect_dir / "null_distribution.npy"
    cluster_p_path = effect_dir / "cluster_p_values.npy"
    metadata_path = effect_dir / "summary.json"

    ensure_parent_dir(metadata_path)
    np.save(observed_path, np.asarray(inference.observed_statistic, dtype=float))
    np.save(corrected_p_path, np.asarray(inference.corrected_p_values, dtype=float))
    np.save(null_path, np.asarray(inference.null_distribution, dtype=float))
    np.save(cluster_p_path, np.asarray([] if inference.cluster_p_values is None else inference.cluster_p_values, dtype=float))

    corrected_p_values = np.asarray(inference.corrected_p_values, dtype=float)
    significant_mask = corrected_p_values < 0.05
    summary = {
        "status": "ok",
        "predictor": predictor_name,
        "effect": effect_name,
        "model_name": model_name,
        "n_subjects": len(subject_ids),
        "subject_ids": subject_ids,
        "correction": str(inference.backend_metadata.get("correction", "")),
        "backend_metadata": dict(inference.backend_metadata),
        "observed_shape": list(np.asarray(inference.observed_statistic).shape),
        "corrected_p_shape": list(corrected_p_values.shape),
        "n_significant_p_lt_0_05": int(np.count_nonzero(significant_mask)),
        "min_corrected_p": float(np.nanmin(corrected_p_values)),
        "observed_statistic": str(observed_path),
        "corrected_p_values": str(corrected_p_path),
        "null_distribution": str(null_path),
        "cluster_p_values": str(cluster_p_path),
    }
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _serialize_threshold(threshold):
    if isinstance(threshold, Mapping):
        return {str(key): float(value) for key, value in threshold.items()}
    if threshold is None:
        return None
    return float(threshold)


def _safe_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value) or "effect"
