"""Similarity and difference analysis for paired lmeEEG predictor maps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from rate.design import list_model_names, load_model_spec
from rate.lmeeeg.io import build_lmeeeg_model_output_paths
from rate.paths import ensure_parent_dir, resolve_out_dir
from rate.util import write_parquet_dataframe

WholeMapMetricName = Literal["pearson", "cosine"]
TimeResolvedMethodName = Literal["pearson"]
TailName = Literal["two-sided", "greater", "less"]


@dataclass(frozen=True)
class SimilarityInputSettings:
    """Resolved input settings for the map comparison analysis."""

    domain: str
    result_kind: str
    predictor_a: str
    predictor_b: str
    value_field: str
    use_unthresholded: bool


@dataclass(frozen=True)
class TimeResolvedSettings:
    """Resolved time-resolved topographic settings."""

    enabled: bool
    method: TimeResolvedMethodName


@dataclass(frozen=True)
class SimilarityInferentialSettings:
    """Resolved permutation-testing settings."""

    enabled: bool
    n_permutations: int
    random_seed: int
    null_method: str
    tail: TailName


@dataclass(frozen=True)
class SimilarityOutputSettings:
    """Resolved output toggles."""

    save_null_distribution: bool
    save_time_resolved_curve: bool
    save_summary_table: bool
    save_difference_map: bool


@dataclass(frozen=True)
class SimilaritySettings:
    """Resolved top-level config for the explicit similarity analysis."""

    enabled: bool
    analysis_name: str
    input_settings: SimilarityInputSettings
    whole_map_metrics: tuple[WholeMapMetricName, ...]
    time_resolved: TimeResolvedSettings
    inferential_test: SimilarityInferentialSettings
    outputs: SimilarityOutputSettings


@dataclass(frozen=True)
class SimilarityOutputPaths:
    """Canonical outputs for one similarity analysis."""

    root_dir: Path
    summary_output_path: Path
    summary_table_output_path: Path
    difference_map_output_path: Path
    time_resolved_curve_output_path: Path
    time_resolved_p_values_output_path: Path
    time_resolved_null_output_path: Path
    whole_map_null_output_path: Path
    channel_names_output_path: Path
    times_output_path: Path


@dataclass(frozen=True)
class PredictorMap:
    """Loaded map and metadata for one predictor-specific lmeEEG result."""

    predictor_name: str
    model_name: str
    effect_name: str
    value_field: str
    values: np.ndarray
    channel_names: list[str]
    times: np.ndarray


def resolve_similarity_settings(config: Mapping[str, Any]) -> SimilaritySettings:
    """Resolve the dedicated similarity config with small defaults."""
    section = dict(config.get("similarity") or {})
    input_section = dict(section.get("input") or {})
    metrics_section = dict(section.get("metrics") or {})
    inferential_section = dict(section.get("inferential_test") or {})
    outputs_section = dict(section.get("outputs") or {})
    time_resolved_section = dict(metrics_section.get("time_resolved_topography") or {})

    whole_map_metrics = tuple(
        _normalize_whole_map_metric(metric_name)
        for metric_name in (metrics_section.get("whole_map") or ["pearson", "cosine"])
    )
    if not whole_map_metrics:
        raise ValueError("`similarity.metrics.whole_map` must list at least one metric.")

    null_method = str(inferential_section.get("null_method", "sign_flip")).lower()
    if null_method != "sign_flip":
        raise ValueError(f"Unsupported similarity null method `{null_method}`.")

    tail = str(inferential_section.get("tail", "two-sided")).lower()
    if tail not in {"two-sided", "greater", "less"}:
        raise ValueError(f"Unsupported similarity tail `{tail}`.")

    return SimilaritySettings(
        enabled=bool(section.get("enabled", True)),
        analysis_name=str(section.get("analysis_name", "similarity")),
        input_settings=SimilarityInputSettings(
            domain=str(input_section.get("domain", "lmeeeg")),
            result_kind=str(input_section.get("result_kind", "group_stats")),
            predictor_a=str(input_section.get("predictor_a", "other_rate")),
            predictor_b=str(input_section.get("predictor_b", "self_rate")),
            value_field=str(input_section.get("value_field", "beta")).lower(),
            use_unthresholded=bool(input_section.get("use_unthresholded", True)),
        ),
        whole_map_metrics=whole_map_metrics,
        time_resolved=TimeResolvedSettings(
            enabled=bool(time_resolved_section.get("enabled", False)),
            method=_normalize_time_resolved_method(time_resolved_section.get("method", "pearson")),
        ),
        inferential_test=SimilarityInferentialSettings(
            enabled=bool(inferential_section.get("enabled", True)),
            n_permutations=max(1, int(inferential_section.get("n_permutations", 1024))),
            random_seed=int(inferential_section.get("random_seed", 0)),
            null_method=null_method,
            tail=tail,
        ),
        outputs=SimilarityOutputSettings(
            save_null_distribution=bool(outputs_section.get("save_null_distribution", True)),
            save_time_resolved_curve=bool(outputs_section.get("save_time_resolved_curve", True)),
            save_summary_table=bool(outputs_section.get("save_summary_table", True)),
            save_difference_map=bool(outputs_section.get("save_difference_map", True)),
        ),
    )


def build_similarity_output_paths(config: Mapping[str, Any]) -> SimilarityOutputPaths:
    """Build canonical output paths for the explicit similarity analysis."""
    settings = resolve_similarity_settings(config)
    root_dir = resolve_out_dir(config) / "similarity" / _sanitize_token(settings.analysis_name)
    return SimilarityOutputPaths(
        root_dir=root_dir,
        summary_output_path=root_dir / "summary.json",
        summary_table_output_path=root_dir / "summary_table.parquet",
        difference_map_output_path=root_dir / "difference_map.npy",
        time_resolved_curve_output_path=root_dir / "time_resolved_curve.npy",
        time_resolved_p_values_output_path=root_dir / "time_resolved_corrected_p_values.npy",
        time_resolved_null_output_path=root_dir / "time_resolved_null_max.npy",
        whole_map_null_output_path=root_dir / "whole_map_null_distributions.npz",
        channel_names_output_path=root_dir / "channel_names.json",
        times_output_path=root_dir / "times.npy",
    )


def run_similarity_analysis(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the explicit similarity analysis and write its outputs."""
    settings = resolve_similarity_settings(config)
    if not settings.enabled:
        raise ValueError("Similarity analysis is disabled in config.")

    map_a = load_predictor_map(config, predictor_name=settings.input_settings.predictor_a)
    map_b = load_predictor_map(config, predictor_name=settings.input_settings.predictor_b)
    _validate_predictor_maps(map_a, map_b)
    output_paths = build_similarity_output_paths(config)
    ensure_parent_dir(output_paths.summary_output_path)

    difference_map = np.asarray(map_a.values - map_b.values, dtype=float)
    summary_rows: list[dict[str, Any]] = []
    whole_map_summary: dict[str, Any] = {}
    whole_map_nulls: dict[str, np.ndarray] = {}

    for metric_name in settings.whole_map_metrics:
        observed_value, contributions = _compute_metric_with_contributions(map_a.values, map_b.values, metric_name=metric_name)
        metric_summary = {
            "scope": "whole_map",
            "metric": metric_name,
            "observed_value": None if not np.isfinite(observed_value) else float(observed_value),
            "n_features": int(contributions.shape[0]),
            "p_value": None,
        }
        if settings.inferential_test.enabled and contributions.size > 0:
            null_distribution = _sign_flip_null_distribution(
                contributions,
                n_permutations=settings.inferential_test.n_permutations,
                seed=settings.inferential_test.random_seed + len(whole_map_nulls),
                aggregator="sum" if metric_name == "cosine" else "mean",
            )
            whole_map_nulls[metric_name] = null_distribution
            metric_summary["p_value"] = float(
                _permutation_p_value(
                    observed=float(observed_value),
                    null_distribution=null_distribution,
                    tail=settings.inferential_test.tail,
                )
            )
        whole_map_summary[metric_name] = metric_summary
        summary_rows.append(metric_summary)

    difference_summary = _build_difference_summary(difference_map)
    summary_rows.extend(
        {
            "scope": "whole_map_difference",
            "metric": metric_name,
            "observed_value": float(metric_value),
            "n_features": int(np.count_nonzero(np.isfinite(difference_map))),
            "p_value": None,
        }
        for metric_name, metric_value in difference_summary.items()
    )

    time_resolved_summary: dict[str, Any] | None = None
    if settings.time_resolved.enabled:
        observed_curve, corrected_p_values, null_max_distribution = _compute_time_resolved_similarity(
            map_a.values,
            map_b.values,
            method=settings.time_resolved.method,
            inferential=settings.inferential_test,
        )
        time_resolved_summary = {
            "method": settings.time_resolved.method,
            "correction": "tfce",
            "curve_path": str(output_paths.time_resolved_curve_output_path) if settings.outputs.save_time_resolved_curve else None,
            "corrected_p_values_path": str(output_paths.time_resolved_p_values_output_path) if settings.outputs.save_time_resolved_curve else None,
            "null_distribution_path": str(output_paths.time_resolved_null_output_path) if settings.outputs.save_null_distribution else None,
            "n_timepoints": int(observed_curve.shape[0]),
            "n_significant_p_lt_0_05": int(np.count_nonzero(np.asarray(corrected_p_values, dtype=float) < 0.05)),
            "max_observed_value": None if not np.any(np.isfinite(observed_curve)) else float(np.nanmax(observed_curve)),
        }
        if settings.outputs.save_time_resolved_curve:
            np.save(output_paths.time_resolved_curve_output_path, observed_curve)
            np.save(output_paths.time_resolved_p_values_output_path, corrected_p_values)
        if settings.outputs.save_null_distribution:
            np.save(output_paths.time_resolved_null_output_path, null_max_distribution)

    if settings.outputs.save_difference_map:
        np.save(output_paths.difference_map_output_path, difference_map)
    if settings.outputs.save_null_distribution and whole_map_nulls:
        np.savez(output_paths.whole_map_null_output_path, **whole_map_nulls)
    if settings.outputs.save_summary_table:
        write_parquet_dataframe(pd.DataFrame(summary_rows), output_paths.summary_table_output_path)

    output_paths.channel_names_output_path.write_text(json.dumps(map_a.channel_names, indent=2) + "\n", encoding="utf-8")
    np.save(output_paths.times_output_path, np.asarray(map_a.times, dtype=float))

    summary = {
        "status": "ok",
        "analysis": "lmeeeg_similarity",
        "analysis_name": settings.analysis_name,
        "input": {
            "domain": settings.input_settings.domain,
            "result_kind": settings.input_settings.result_kind,
            "value_field": settings.input_settings.value_field,
            "use_unthresholded": settings.input_settings.use_unthresholded,
            "predictor_a": map_a.predictor_name,
            "predictor_b": map_b.predictor_name,
            "model_a": map_a.model_name,
            "model_b": map_b.model_name,
            "effect_a": map_a.effect_name,
            "effect_b": map_b.effect_name,
        },
        "whole_map_similarity": whole_map_summary,
        "whole_map_difference": difference_summary,
        "difference_map_path": str(output_paths.difference_map_output_path) if settings.outputs.save_difference_map else None,
        "summary_table_path": str(output_paths.summary_table_output_path) if settings.outputs.save_summary_table else None,
        "whole_map_null_path": str(output_paths.whole_map_null_output_path) if settings.outputs.save_null_distribution and whole_map_nulls else None,
        "time_resolved_topography": time_resolved_summary,
        "channel_names_path": str(output_paths.channel_names_output_path),
        "times_path": str(output_paths.times_output_path),
        "summary_path": str(output_paths.summary_output_path),
        "n_channels": len(map_a.channel_names),
        "n_times": int(map_a.times.shape[0]),
        "note": (
            "This component compares two existing unthresholded pooled lmeEEG fixed-effect maps. "
            "The current `result_kind=group_stats` implementation resolves those maps from the existing pooled "
            "lmeEEG fit outputs because the permutation inference outputs do not store unthresholded beta maps."
        ),
    }
    output_paths.summary_output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def load_predictor_map(config: Mapping[str, Any], *, predictor_name: str) -> PredictorMap:
    """Load one predictor-specific pooled lmeEEG map from existing outputs."""
    settings = resolve_similarity_settings(config)
    if settings.input_settings.domain != "lmeeeg":
        raise ValueError(f"Unsupported similarity input domain `{settings.input_settings.domain}`.")
    if settings.input_settings.result_kind != "group_stats":
        raise ValueError(f"Unsupported similarity result kind `{settings.input_settings.result_kind}`.")
    if not settings.input_settings.use_unthresholded:
        raise ValueError("Similarity analysis currently requires `use_unthresholded: true`.")
    if settings.input_settings.value_field not in {"beta", "t_value"}:
        raise ValueError(f"Unsupported similarity value field `{settings.input_settings.value_field}`.")

    model_name = _resolve_single_predictor_model_name(config, predictor_name=predictor_name, section=settings.input_settings.domain)
    model_paths = build_lmeeeg_model_output_paths(config, config_section=settings.input_settings.domain, model_name=model_name)
    if not model_paths.summary_output_path.exists():
        raise FileNotFoundError(f"lmeEEG pooled summary not found for model `{model_name}`: {model_paths.summary_output_path}")

    column_names = json.loads(model_paths.column_names_output_path.read_text(encoding="utf-8"))
    channel_names = json.loads(model_paths.channel_names_output_path.read_text(encoding="utf-8"))
    times = np.load(model_paths.times_output_path)
    effect_name = _resolve_effect_name(column_names, predictor_name=predictor_name)

    values = np.load(model_paths.betas_output_path if settings.input_settings.value_field == "beta" else model_paths.t_values_output_path)
    if values.ndim != 3:
        raise ValueError(
            f"Expected 3D predictor maps in `{settings.input_settings.value_field}` output for model `{model_name}`, "
            f"got shape {values.shape}."
        )
    effect_index = int(column_names.index(effect_name))
    return PredictorMap(
        predictor_name=predictor_name,
        model_name=model_name,
        effect_name=effect_name,
        value_field=settings.input_settings.value_field,
        values=np.asarray(values[effect_index], dtype=float),
        channel_names=[str(name) for name in channel_names],
        times=np.asarray(times, dtype=float),
    )


def _resolve_single_predictor_model_name(config: Mapping[str, Any], *, predictor_name: str, section: str) -> str:
    candidate_models: list[str] = []
    for model_name in list_model_names(config, section=section):
        model_spec = load_model_spec(config, section=section, model_name=model_name)
        if tuple(model_spec.test_predictors) == (predictor_name,):
            candidate_models.append(model_name)
    if len(candidate_models) == 1:
        return candidate_models[0]
    if not candidate_models:
        raise ValueError(f"No single-predictor lmeEEG model found for predictor `{predictor_name}` in section `{section}`.")
    joined = ", ".join(candidate_models)
    raise ValueError(f"Multiple single-predictor lmeEEG models match predictor `{predictor_name}`: {joined}")


def _resolve_effect_name(column_names: list[str], *, predictor_name: str) -> str:
    for column_name in column_names:
        if str(column_name) == predictor_name:
            return str(column_name)
    raise ValueError(f"Predictor `{predictor_name}` did not resolve to a saved fixed-effect column. Available columns: {column_names}")


def _validate_predictor_maps(map_a: PredictorMap, map_b: PredictorMap) -> None:
    if map_a.values.shape != map_b.values.shape:
        raise ValueError(f"Predictor maps have incompatible shapes: {map_a.values.shape} != {map_b.values.shape}")
    if list(map_a.channel_names) != list(map_b.channel_names):
        raise ValueError("Predictor maps do not share the same channel ordering.")
    if not np.array_equal(np.asarray(map_a.times), np.asarray(map_b.times)):
        raise ValueError("Predictor maps do not share the same time axis.")


def _compute_metric_with_contributions(
    map_a: np.ndarray,
    map_b: np.ndarray,
    *,
    metric_name: WholeMapMetricName,
) -> tuple[float, np.ndarray]:
    x, y = _paired_valid_vectors(map_a, map_b)
    if x.size == 0:
        return float("nan"), np.asarray([], dtype=float)
    if metric_name == "pearson":
        std_x = float(np.std(x))
        std_y = float(np.std(y))
        if std_x == 0.0 or std_y == 0.0:
            return float("nan"), np.asarray([], dtype=float)
        zx = (x - float(np.mean(x))) / std_x
        zy = (y - float(np.mean(y))) / std_y
        contributions = np.asarray(zx * zy, dtype=float)
        return float(np.mean(contributions)), contributions
    if metric_name == "cosine":
        norm_x = float(np.linalg.norm(x))
        norm_y = float(np.linalg.norm(y))
        if norm_x == 0.0 or norm_y == 0.0:
            return float("nan"), np.asarray([], dtype=float)
        contributions = np.asarray((x / norm_x) * (y / norm_y), dtype=float)
        return float(np.sum(contributions)), contributions
    raise ValueError(f"Unsupported whole-map metric `{metric_name}`.")


def _build_difference_summary(difference_map: np.ndarray) -> dict[str, float]:
    flat = difference_map[np.isfinite(difference_map)]
    if flat.size == 0:
        return {
            "mean_signed_difference": float("nan"),
            "mean_absolute_difference": float("nan"),
            "rms_difference": float("nan"),
        }
    return {
        "mean_signed_difference": float(np.mean(flat)),
        "mean_absolute_difference": float(np.mean(np.abs(flat))),
        "rms_difference": float(np.sqrt(np.mean(np.square(flat)))),
    }


def _compute_time_resolved_similarity(
    map_a: np.ndarray,
    map_b: np.ndarray,
    *,
    method: TimeResolvedMethodName,
    inferential: SimilarityInferentialSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method != "pearson":
        raise ValueError(f"Unsupported time-resolved similarity method `{method}`.")

    n_times = int(map_a.shape[1])
    observed = np.full(n_times, np.nan, dtype=float)
    per_time_contributions: list[np.ndarray] = []

    for time_index in range(n_times):
        observed_value, contributions = _compute_metric_with_contributions(
            map_a[:, time_index],
            map_b[:, time_index],
            metric_name="pearson",
        )
        observed[time_index] = float(observed_value)
        per_time_contributions.append(contributions)

    corrected_p = np.full(n_times, np.nan, dtype=float)
    observed_tfce = _compute_tfce_1d(np.abs(observed), h_power=2.0, e_power=0.5, h_start=0.0, dh=0.1)
    null_max = np.asarray([], dtype=float)
    if inferential.enabled:
        null_max_tfce = np.zeros(inferential.n_permutations, dtype=float)
        rng = np.random.default_rng(inferential.random_seed + 1000)
        for permutation_index in range(inferential.n_permutations):
            permuted_curve = np.full(n_times, np.nan, dtype=float)
            for time_index, contributions in enumerate(per_time_contributions):
                if contributions.size == 0:
                    continue
                signs = rng.choice(np.asarray([-1.0, 1.0], dtype=float), size=contributions.shape[0], replace=True)
                permuted_curve[time_index] = float(np.mean(contributions * signs))
            permuted_tfce = _compute_tfce_1d(np.abs(permuted_curve), h_power=2.0, e_power=0.5, h_start=0.0, dh=0.1)
            null_max_tfce[permutation_index] = float(np.nanmax(permuted_tfce)) if np.any(np.isfinite(permuted_tfce)) else 0.0

        if null_max_tfce.size > 0:
            null_max = null_max_tfce
            for time_index, observed_value in enumerate(observed):
                if not np.isfinite(observed_value):
                    continue
                corrected_p[time_index] = _permutation_p_value(
                    observed=float(observed_tfce[time_index]),
                    null_distribution=null_max,
                    tail="greater",
                )
    return observed, corrected_p, null_max


def _paired_valid_vectors(map_a: np.ndarray, map_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(map_a, dtype=float).reshape(-1)
    y = np.asarray(map_b, dtype=float).reshape(-1)
    valid_mask = np.isfinite(x) & np.isfinite(y)
    return x[valid_mask], y[valid_mask]


def _sign_flip_null_distribution(
    contributions: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
    aggregator: Literal["mean", "sum"],
) -> np.ndarray:
    if contributions.size == 0:
        return np.asarray([], dtype=float)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=float), size=(n_permutations, contributions.shape[0]), replace=True)
    signed = signs * contributions[np.newaxis, :]
    if aggregator == "sum":
        return np.asarray(np.sum(signed, axis=1), dtype=float)
    return np.asarray(np.mean(signed, axis=1), dtype=float)


def _permutation_p_value(*, observed: float, null_distribution: np.ndarray, tail: TailName) -> float:
    if null_distribution.size == 0 or not np.isfinite(observed):
        return float("nan")
    null_values = np.asarray(null_distribution, dtype=float)
    if tail == "two-sided":
        exceedances = np.count_nonzero(np.abs(null_values) >= abs(observed))
    elif tail == "greater":
        exceedances = np.count_nonzero(null_values >= observed)
    else:
        exceedances = np.count_nonzero(null_values <= observed)
    return float((exceedances + 1.0) / (null_values.shape[0] + 1.0))


def _compute_tfce_1d(
    score_vector: np.ndarray,
    *,
    h_power: float,
    e_power: float,
    h_start: float,
    dh: float,
) -> np.ndarray:
    """Compute a positive-tail TFCE transform on a 1D time series."""
    scores = np.maximum(np.asarray(score_vector, dtype=float), 0.0)
    tfce = np.zeros_like(scores, dtype=float)
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        return tfce
    max_height = float(np.nanmax(finite_scores))
    if np.isclose(max_height, 0.0):
        return tfce

    threshold = max(float(h_start), 0.0)
    if np.isclose(threshold, 0.0):
        threshold = float(dh)
    while threshold <= (max_height + (0.5 * dh)):
        supra_threshold = np.asarray(scores >= threshold, dtype=bool)
        for component_indices in _connected_components_1d(supra_threshold):
            extent = float(len(component_indices))
            increment = (threshold ** h_power) * (extent ** e_power) * dh
            tfce[np.asarray(component_indices, dtype=int)] += increment
        threshold += dh
    return tfce


def _connected_components_1d(mask: np.ndarray) -> list[list[int]]:
    """Return contiguous True components in a 1D boolean mask."""
    values = np.asarray(mask, dtype=bool).reshape(-1)
    components: list[list[int]] = []
    current: list[int] = []
    for index, is_active in enumerate(values):
        if is_active:
            current.append(index)
            continue
        if current:
            components.append(current)
            current = []
    if current:
        components.append(current)
    return components


def _normalize_whole_map_metric(metric_name: Any) -> WholeMapMetricName:
    normalized = str(metric_name).lower()
    if normalized not in {"pearson", "cosine"}:
        raise ValueError(f"Unsupported whole-map similarity metric `{normalized}`.")
    return normalized  # type: ignore[return-value]


def _normalize_time_resolved_method(method_name: Any) -> TimeResolvedMethodName:
    normalized = str(method_name).lower()
    if normalized != "pearson":
        raise ValueError(f"Unsupported time-resolved similarity method `{normalized}`.")
    return normalized  # type: ignore[return-value]


def _sanitize_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value)) or "analysis"
