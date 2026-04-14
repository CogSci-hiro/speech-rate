"""Decoding-specific configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from rate.config import ConfigValidationError, load_yaml_config

DEFAULT_DECODING_CONFIG_PATH = Path("config/decoding.yaml")


def load_decoding_config(
    decoding_config_path: Path | str,
    *,
    project_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and lightly validate the standalone decoding config."""
    payload = load_yaml_config(decoding_config_path)
    decoding_cfg = payload.get("decoding") or {}
    if not isinstance(decoding_cfg, Mapping):
        raise ConfigValidationError("`decoding` must be a mapping in the decoding config.")

    resolved = dict(decoding_cfg)
    resolved.setdefault("enabled", True)
    resolved.setdefault("analysis_name", "decoding")
    resolved.setdefault("output_dir", "decoding")
    resolved.setdefault("sfreq", 32.0)
    resolved.setdefault("random_state", int((project_config or {}).get("project", {}).get("random_seed", 42)))
    resolved.setdefault("controls", [])
    resolved.setdefault("input", {})
    resolved.setdefault("time_window", {})
    resolved.setdefault("channels", {})
    resolved.setdefault("targets", {})
    resolved.setdefault("eeg_residualisation", {})
    resolved.setdefault("target_residualisation", {})
    resolved.setdefault("cv", {})
    resolved.setdefault("nested_cv", {})
    resolved.setdefault("estimator", {})
    resolved.setdefault("scoring", {})
    resolved.setdefault("stats", {})
    resolved.setdefault("models", [])

    for key in ("input", "time_window", "channels", "targets", "eeg_residualisation", "target_residualisation", "cv", "nested_cv", "estimator", "scoring", "stats"):
        if not isinstance(resolved[key], Mapping):
            raise ConfigValidationError(f"`decoding.{key}` must be a mapping.")

    models = resolved["models"]
    if not isinstance(models, list) or not models:
        raise ConfigValidationError("`decoding.models` must contain at least one model definition.")

    input_cfg = dict(resolved["input"])
    input_cfg.setdefault("epochs_section", "epochs")
    input_cfg.setdefault("events_table", None)
    resolved["input"] = input_cfg

    channels_cfg = dict(resolved["channels"])
    channels_cfg.setdefault("picks", "eeg")
    channels_cfg.setdefault("include", [])
    channels_cfg.setdefault("exclude", [])
    resolved["channels"] = channels_cfg

    eeg_resid_cfg = dict(resolved["eeg_residualisation"])
    eeg_resid_cfg.setdefault("random_effect", "subject_id")
    eeg_resid_cfg.setdefault("controls", list(resolved["controls"]))
    resolved["eeg_residualisation"] = eeg_resid_cfg

    target_resid_cfg = dict(resolved["target_residualisation"])
    target_resid_cfg.setdefault("random_effect", "subject_id")
    target_resid_cfg.setdefault("controls", list(resolved["controls"]))
    resolved["target_residualisation"] = target_resid_cfg

    cv_cfg = dict(resolved["cv"])
    cv_cfg.setdefault("n_splits", 5)
    cv_cfg.setdefault("group_column", "subject_id")
    cv_cfg.setdefault("shuffle", True)
    cv_cfg.setdefault("random_state", int(resolved["random_state"]))
    resolved["cv"] = cv_cfg

    nested_cv_cfg = dict(resolved["nested_cv"])
    nested_cv_cfg.setdefault("enabled", True)
    nested_cv_cfg.setdefault("inner_n_splits", 4)
    nested_cv_cfg.setdefault("selection_metric", "mean_diagonal_pearson_r")
    nested_cv_cfg.setdefault("select_single_alpha_per_outer_fold", True)
    if str(nested_cv_cfg["selection_metric"]) != "mean_diagonal_pearson_r":
        raise ConfigValidationError(
            "Only `decoding.nested_cv.selection_metric: mean_diagonal_pearson_r` is currently supported."
        )
    if not bool(nested_cv_cfg["select_single_alpha_per_outer_fold"]):
        raise ConfigValidationError(
            "`decoding.nested_cv.select_single_alpha_per_outer_fold` must be true for the current decoding pipeline."
        )
    resolved["nested_cv"] = nested_cv_cfg

    estimator_cfg = dict(resolved["estimator"])
    estimator_cfg.setdefault("name", "ridge")
    estimator_cfg.setdefault("alpha", 1.0)
    estimator_cfg.setdefault("alpha_grid", None)
    estimator_cfg.setdefault("alphas", None)
    estimator_cfg.setdefault("save_alpha_scores", True)
    if estimator_cfg.get("alpha_grid") is not None:
        alpha_grid = estimator_cfg["alpha_grid"]
        if not isinstance(alpha_grid, list) or not alpha_grid:
            raise ConfigValidationError("`decoding.estimator.alpha_grid` must be a non-empty list when provided.")
        estimator_cfg["alpha_grid"] = [float(value) for value in alpha_grid]
    if estimator_cfg.get("alphas") is not None:
        alphas_cfg = estimator_cfg["alphas"]
        if not isinstance(alphas_cfg, Mapping):
            raise ConfigValidationError("`decoding.estimator.alphas` must be a mapping when provided.")
        normalized_alphas_cfg = dict(alphas_cfg)
        for required_key in ("start", "stop", "n_alphas"):
            if required_key not in normalized_alphas_cfg:
                raise ConfigValidationError(f"`decoding.estimator.alphas.{required_key}` is required when using an alpha grid.")
        estimator_cfg["alphas"] = normalized_alphas_cfg
    if estimator_cfg.get("alpha_grid") is not None and estimator_cfg.get("alphas") is not None:
        raise ConfigValidationError("Use either `decoding.estimator.alpha_grid` or `decoding.estimator.alphas`, not both.")
    resolved["estimator"] = estimator_cfg

    scoring_cfg = dict(resolved["scoring"])
    scoring_cfg.setdefault("name", "pearson_r")
    resolved["scoring"] = scoring_cfg

    stats_cfg = dict(resolved["stats"])
    stats_cfg.setdefault("enabled", True)
    stats_cfg.setdefault("n_permutations", 1000)
    stats_cfg.setdefault("random_state", int(resolved["random_state"]))
    stats_cfg.setdefault("permutation_scheme", "within_subject")
    stats_cfg.setdefault("n_jobs", 1)
    stats_cfg.setdefault("tfce", {})
    if not isinstance(stats_cfg["tfce"], Mapping):
        raise ConfigValidationError("`decoding.stats.tfce` must be a mapping.")
    tfce_cfg = dict(stats_cfg["tfce"])
    tfce_cfg.setdefault("enabled", True)
    tfce_cfg.setdefault("h_power", 2.0)
    tfce_cfg.setdefault("e_power", 0.5)
    tfce_cfg.setdefault("h_start", 0.0)
    tfce_cfg.setdefault("dh", 0.1)
    tfce_cfg.setdefault("tail", "positive")
    stats_cfg["tfce"] = tfce_cfg
    if str(stats_cfg["permutation_scheme"]) not in {"within_subject", "within_subject_within_run"}:
        raise ConfigValidationError("`decoding.stats.permutation_scheme` must be `within_subject` or `within_subject_within_run`.")
    if str(tfce_cfg["tail"]) != "positive":
        raise ConfigValidationError("Only `decoding.stats.tfce.tail: positive` is currently supported.")
    resolved["stats"] = stats_cfg

    target_defs = resolved["targets"]
    if target_defs:
        if not isinstance(target_defs, Mapping):
            raise ConfigValidationError("`decoding.targets` must be a mapping when provided.")
        resolved["targets"] = {str(name): dict(spec or {}) for name, spec in target_defs.items()}

    model_names: set[str] = set()
    normalized_models: list[dict[str, str]] = []
    for raw_model in models:
        if not isinstance(raw_model, Mapping):
            raise ConfigValidationError("Each decoding model definition must be a mapping.")
        model_cfg = dict(raw_model)
        for required_key in ("name", "train_target", "test_target"):
            value = model_cfg.get(required_key)
            if not value:
                raise ConfigValidationError(f"Decoding model is missing required key `{required_key}`.")
        model_name = str(model_cfg["name"])
        if model_name in model_names:
            raise ConfigValidationError(f"Duplicate decoding model name: {model_name}")
        model_names.add(model_name)
        normalized_models.append(
            {
                "name": model_name,
                "train_target": str(model_cfg["train_target"]),
                "test_target": str(model_cfg["test_target"]),
            }
        )
    resolved["models"] = normalized_models
    return resolved


def list_decoding_model_names(decoding_config: Mapping[str, Any]) -> list[str]:
    """Return configured decoding model names in order."""
    models = decoding_config.get("models") or []
    return [str(model["name"]) for model in models]
