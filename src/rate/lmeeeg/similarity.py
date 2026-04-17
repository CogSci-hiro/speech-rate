"""Similarity and difference analysis for paired lmeEEG predictor maps."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from rate.design import list_model_names, load_model_spec
from rate.lmeeeg.io import build_lmeeeg_model_output_paths
from rate.lmeeeg.pooling import PooledLMEEEGInputs, fit_pooled_lmeeeg_model, load_pooled_lmeeeg_inputs, select_lmeeeg_rows
from rate.paths import ensure_parent_dir, resolve_out_dir
from rate.util import write_parquet_dataframe

LOGGER = logging.getLogger(__name__)

WholeMapMetricName = Literal["pearson", "cosine"]
TimeResolvedMethodName = Literal["pearson"]
TailName = Literal["two-sided", "greater", "less"]
SimilarityInferenceMethod = Literal["none", "sign_flip", "refit_permutation"]
SimilarityPermutationKind = Literal["circular_shift_within_subject"]


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
class SimilarityTFCEConfig:
    """Resolved TFCE configuration for time-resolved similarity inference."""

    enabled: bool
    e_power: float
    h_power: float
    dh: float
    h_start: float


@dataclass(frozen=True)
class SimilarityPermutationScheme:
    """Resolved constrained permutation scheme for similarity inference."""

    kind: SimilarityPermutationKind
    group_by: tuple[str, ...]
    within_run: bool


@dataclass(frozen=True)
class SimilarityInferentialSettings:
    """Resolved inferential settings for similarity analysis."""

    method: SimilarityInferenceMethod
    n_permutations: int
    random_seed: int
    tail: TailName
    tfce: SimilarityTFCEConfig
    permutation_scheme: SimilarityPermutationScheme

    @property
    def enabled(self) -> bool:
        return self.method != "none"


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
    inference: SimilarityInferentialSettings
    outputs: SimilarityOutputSettings


@dataclass(frozen=True)
class SimilarityOutputPaths:
    """Canonical outputs for one similarity analysis."""

    root_dir: Path
    summary_output_path: Path
    metadata_output_path: Path
    summary_table_output_path: Path
    difference_map_output_path: Path
    time_resolved_curve_output_path: Path
    time_resolved_p_values_output_path: Path
    time_resolved_null_curves_output_path: Path
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


@dataclass(frozen=True)
class SimilarityPermutationGrouping:
    """Ordered block indices reused across many constrained permutations."""

    group_labels: tuple[str, ...]
    group_indices: tuple[np.ndarray, ...]


def resolve_similarity_settings(config: Mapping[str, Any]) -> SimilaritySettings:
    """Resolve the dedicated similarity config with explicit inference modes."""
    section = dict(config.get("similarity") or {})
    input_section = dict(section.get("input") or {})
    metrics_section = dict(section.get("metrics") or {})
    inference_section = dict(section.get("inference") or section.get("inferential_test") or {})
    outputs_section = dict(section.get("outputs") or {})
    time_resolved_section = dict(
        metrics_section.get("time_resolved_topography")
        or metrics_section.get("time_resolved")
        or {}
    )
    tfce_section = dict(inference_section.get("tfce") or {})
    permutation_section = dict(inference_section.get("permutation_scheme") or {})

    whole_map_metrics = tuple(
        _normalize_whole_map_metric(metric_name)
        for metric_name in (metrics_section.get("whole_map") or ["pearson", "cosine"])
    )
    if not whole_map_metrics:
        raise ValueError("`similarity.metrics.whole_map` must list at least one metric.")

    legacy_enabled = inference_section.get("enabled")
    explicit_method = inference_section.get("method")
    if explicit_method is None and legacy_enabled is False:
        explicit_method = "none"
    if explicit_method is None:
        explicit_method = inference_section.get("null_method", "sign_flip")
    method = _normalize_inference_method(explicit_method)

    tail = str(inference_section.get("tail", "two-sided")).lower()
    if tail not in {"two-sided", "greater", "less"}:
        raise ValueError(f"Unsupported similarity tail `{tail}`.")

    permutation_kind = str(permutation_section.get("kind", "circular_shift_within_subject")).lower()
    if permutation_kind != "circular_shift_within_subject":
        raise ValueError(f"Unsupported similarity permutation scheme `{permutation_kind}`.")

    group_by = tuple(str(column) for column in (permutation_section.get("group_by") or ["subject_id"]))
    if not group_by:
        raise ValueError("`similarity.inference.permutation_scheme.group_by` must list at least one grouping column.")

    return SimilaritySettings(
        enabled=bool(section.get("enabled", True)),
        analysis_name=str(section.get("analysis_name", "similarity")),
        input_settings=SimilarityInputSettings(
            domain=str(input_section.get("domain", "lmeeeg")),
            result_kind=str(input_section.get("result_kind", "group_stats")),
            predictor_a=str(input_section.get("predictor_a", section.get("predictor_a", "other_rate"))),
            predictor_b=str(input_section.get("predictor_b", section.get("predictor_b", "self_rate"))),
            value_field=str(input_section.get("value_field", section.get("map_type", "beta"))).lower(),
            use_unthresholded=bool(input_section.get("use_unthresholded", True)),
        ),
        whole_map_metrics=whole_map_metrics,
        time_resolved=TimeResolvedSettings(
            enabled=bool(time_resolved_section.get("enabled", False)),
            method=_normalize_time_resolved_method(time_resolved_section.get("method", "pearson")),
        ),
        inference=SimilarityInferentialSettings(
            method=method,
            n_permutations=max(1, int(inference_section.get("n_permutations", 1024))),
            random_seed=int(
                permutation_section.get(
                    "random_seed",
                    inference_section.get("random_seed", 0),
                )
            ),
            tail=tail,  # type: ignore[arg-type]
            tfce=SimilarityTFCEConfig(
                enabled=bool(tfce_section.get("enabled", True)),
                e_power=float(tfce_section.get("e_power", 0.5)),
                h_power=float(tfce_section.get("h_power", 2.0)),
                dh=float(tfce_section.get("dh", 0.1)),
                h_start=float(tfce_section.get("h_start", 0.0)),
            ),
            permutation_scheme=SimilarityPermutationScheme(
                kind="circular_shift_within_subject",
                group_by=group_by,
                within_run=bool(permutation_section.get("within_run", False)),
            ),
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
        metadata_output_path=root_dir / "metadata.json",
        summary_table_output_path=root_dir / "summary_table.parquet",
        difference_map_output_path=root_dir / "difference_map.npy",
        time_resolved_curve_output_path=root_dir / "time_resolved_curve.npy",
        time_resolved_p_values_output_path=root_dir / "time_resolved_corrected_p_values.npy",
        time_resolved_null_curves_output_path=root_dir / "time_resolved_null_curves.npy",
        time_resolved_null_output_path=root_dir / "time_resolved_null_max_tfce.npy",
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
    difference_summary = _build_difference_summary(difference_map)
    observed_time_curve = (
        _compute_time_resolved_curve(map_a.values, map_b.values, method=settings.time_resolved.method)
        if settings.time_resolved.enabled
        else None
    )

    whole_map_summary: dict[str, Any] = {}
    whole_map_nulls: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []

    refit_null_payload: dict[str, Any] | None = None
    if settings.inference.method == "refit_permutation":
        refit_null_payload = _run_refit_permutation_similarity(
            config,
            settings=settings,
            observed_map_a=map_a,
            observed_map_b=map_b,
        )

    for metric_name in settings.whole_map_metrics:
        observed_value = _compute_metric(map_a.values, map_b.values, metric_name=metric_name)
        null_distribution = _resolve_whole_map_null_distribution(
            metric_name=metric_name,
            map_a=map_a.values,
            map_b=map_b.values,
            settings=settings,
            refit_null_payload=refit_null_payload,
            metric_offset=len(whole_map_nulls),
        )
        if null_distribution is not None:
            whole_map_nulls[metric_name] = null_distribution
        metric_summary = {
            "scope": "whole_map",
            "metric": metric_name,
            "observed_value": None if not np.isfinite(observed_value) else float(observed_value),
            "n_features": int(np.count_nonzero(np.isfinite(map_a.values) & np.isfinite(map_b.values))),
            "p_value": None if null_distribution is None else float(
                _permutation_p_value(
                    observed=float(observed_value),
                    null_distribution=null_distribution,
                    tail=settings.inference.tail,
                )
            ),
            "inference_method": settings.inference.method,
        }
        whole_map_summary[metric_name] = metric_summary
        summary_rows.append(metric_summary)

    summary_rows.extend(
        {
            "scope": "whole_map_difference",
            "metric": metric_name,
            "observed_value": float(metric_value),
            "n_features": int(np.count_nonzero(np.isfinite(difference_map))),
            "p_value": None,
            "inference_method": settings.inference.method,
        }
        for metric_name, metric_value in difference_summary.items()
    )

    time_resolved_summary: dict[str, Any] | None = None
    corrected_p_values: np.ndarray | None = None
    null_curves: np.ndarray | None = None
    null_max_distribution: np.ndarray | None = None
    if observed_time_curve is not None:
        time_null_payload = _resolve_time_resolved_null_payload(
            map_a=map_a.values,
            map_b=map_b.values,
            observed_curve=observed_time_curve,
            settings=settings,
            refit_null_payload=refit_null_payload,
        )
        corrected_p_values = time_null_payload["corrected_p_values"]
        null_curves = time_null_payload["null_curves"]
        null_max_distribution = time_null_payload["null_max_distribution"]
        correction_name = "tfce" if settings.inference.enabled and settings.inference.tfce.enabled else None
        time_resolved_summary = {
            "method": settings.time_resolved.method,
            "correction": correction_name,
            "curve_path": str(output_paths.time_resolved_curve_output_path) if settings.outputs.save_time_resolved_curve else None,
            "corrected_p_values_path": (
                str(output_paths.time_resolved_p_values_output_path)
                if settings.outputs.save_time_resolved_curve and corrected_p_values is not None
                else None
            ),
            "null_curves_path": (
                str(output_paths.time_resolved_null_curves_output_path)
                if settings.outputs.save_null_distribution and null_curves is not None and null_curves.size > 0
                else None
            ),
            "null_distribution_path": (
                str(output_paths.time_resolved_null_output_path)
                if settings.outputs.save_null_distribution and null_max_distribution is not None and null_max_distribution.size > 0
                else None
            ),
            "n_timepoints": int(observed_time_curve.shape[0]),
            "n_significant_p_lt_0_05": (
                int(np.count_nonzero(np.asarray(corrected_p_values, dtype=float) < 0.05))
                if corrected_p_values is not None
                else 0
            ),
            "max_observed_value": None if not np.any(np.isfinite(observed_time_curve)) else float(np.nanmax(observed_time_curve)),
            "inference_method": settings.inference.method,
        }

    if settings.outputs.save_difference_map:
        np.save(output_paths.difference_map_output_path, difference_map)
    if settings.outputs.save_null_distribution and whole_map_nulls:
        np.savez(output_paths.whole_map_null_output_path, **whole_map_nulls)
    if settings.outputs.save_summary_table:
        write_parquet_dataframe(pd.DataFrame(summary_rows), output_paths.summary_table_output_path)

    output_paths.channel_names_output_path.write_text(json.dumps(map_a.channel_names, indent=2) + "\n", encoding="utf-8")
    np.save(output_paths.times_output_path, np.asarray(map_a.times, dtype=float))
    if settings.outputs.save_time_resolved_curve and observed_time_curve is not None:
        np.save(output_paths.time_resolved_curve_output_path, observed_time_curve)
        if corrected_p_values is not None:
            np.save(output_paths.time_resolved_p_values_output_path, corrected_p_values)
    if settings.outputs.save_null_distribution:
        if null_curves is not None and null_curves.size > 0:
            np.save(output_paths.time_resolved_null_curves_output_path, null_curves)
        if null_max_distribution is not None and null_max_distribution.size > 0:
            np.save(output_paths.time_resolved_null_output_path, null_max_distribution)

    metadata = _build_similarity_metadata(
        settings=settings,
        map_a=map_a,
        map_b=map_b,
        whole_map_nulls=whole_map_nulls,
        null_curves=null_curves,
        null_max_distribution=null_max_distribution,
    )
    output_paths.metadata_output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
        "inference": metadata["inference"],
        "whole_map_similarity": whole_map_summary,
        "whole_map_difference": difference_summary,
        "difference_map_path": str(output_paths.difference_map_output_path) if settings.outputs.save_difference_map else None,
        "summary_table_path": str(output_paths.summary_table_output_path) if settings.outputs.save_summary_table else None,
        "whole_map_null_path": str(output_paths.whole_map_null_output_path) if settings.outputs.save_null_distribution and whole_map_nulls else None,
        "time_resolved_topography": time_resolved_summary,
        "channel_names_path": str(output_paths.channel_names_output_path),
        "times_path": str(output_paths.times_output_path),
        "metadata_path": str(output_paths.metadata_output_path),
        "summary_path": str(output_paths.summary_output_path),
        "n_channels": len(map_a.channel_names),
        "n_times": int(map_a.times.shape[0]),
        "note": (
            "This analysis describes similarity between two existing pooled lmeEEG fixed-effect maps. "
            "Inferential similarity, when enabled, targets a null of disrupted predictor-to-EEG alignment after refitting "
            "the lmeEEG models; it does not test map equality or identical mechanisms."
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


def _run_refit_permutation_similarity(
    config: Mapping[str, Any],
    *,
    settings: SimilaritySettings,
    observed_map_a: PredictorMap,
    observed_map_b: PredictorMap,
) -> dict[str, Any]:
    if settings.input_settings.domain != "lmeeeg":
        raise ValueError("Refit-permutation similarity currently supports only `domain: lmeeeg`.")
    if settings.input_settings.result_kind != "group_stats":
        raise ValueError("Refit-permutation similarity currently supports only `result_kind: group_stats`.")
    if settings.input_settings.value_field != "beta":
        raise ValueError("Refit-permutation similarity currently supports only `value_field: beta`.")

    rows = select_lmeeeg_rows(config)
    pooled_inputs = load_pooled_lmeeeg_inputs(
        config,
        rows,
        config_section=settings.input_settings.domain,
        model_name=observed_map_a.model_name,
    )
    predictor_names = (observed_map_a.predictor_name, observed_map_b.predictor_name)
    grouping = _build_similarity_permutation_grouping(
        pooled_inputs.metadata_df,
        scheme=settings.inference.permutation_scheme,
    )

    n_permutations = settings.inference.n_permutations
    null_whole_map = {
        metric_name: np.full(n_permutations, np.nan, dtype=float)
        for metric_name in settings.whole_map_metrics
    }
    null_time_curves = (
        np.full((n_permutations, observed_map_a.values.shape[1]), np.nan, dtype=float)
        if settings.time_resolved.enabled
        else np.asarray([], dtype=float)
    )

    for permutation_index, permutation_indices in _iter_similarity_permutations(
        pooled_inputs.metadata_df,
        scheme=settings.inference.permutation_scheme,
        grouping=grouping,
        n_permutations=n_permutations,
        random_seed=settings.inference.random_seed,
    ):
        permuted_metadata = _permute_similarity_predictor(
            pooled_inputs.metadata_df,
            predictor_names=predictor_names,
            permutation_indices=permutation_indices,
        )
        base_provenance = pooled_inputs.metadata_provenance or {}
        permutation_meta = {
            "sources": base_provenance.get("sources"),
            "similarity_permutation": {
                "index": int(permutation_index),
                "scheme": settings.inference.permutation_scheme.kind,
                "predictors": list(predictor_names),
            },
        }

        null_map_a = _fit_similarity_predictor_map(
            config,
            pooled_inputs=pooled_inputs,
            metadata_df=permuted_metadata,
            metadata_provenance=permutation_meta,
            model_name=observed_map_a.model_name,
            predictor_name=observed_map_a.predictor_name,
            expected_map=observed_map_a,
        )
        null_map_b = _fit_similarity_predictor_map(
            config,
            pooled_inputs=pooled_inputs,
            metadata_df=permuted_metadata,
            metadata_provenance=permutation_meta,
            model_name=observed_map_b.model_name,
            predictor_name=observed_map_b.predictor_name,
            expected_map=observed_map_b,
        )

        for metric_name in settings.whole_map_metrics:
            null_whole_map[metric_name][permutation_index] = _compute_metric(
                null_map_a.values,
                null_map_b.values,
                metric_name=metric_name,
            )
        if settings.time_resolved.enabled:
            null_time_curves[permutation_index] = _compute_time_resolved_curve(
                null_map_a.values,
                null_map_b.values,
                method=settings.time_resolved.method,
            )

        if _should_log_similarity_progress(permutation_index, total=n_permutations):
            LOGGER.info(
                "Computed similarity refit permutation %d/%d for `%s`",
                permutation_index + 1,
                n_permutations,
                settings.analysis_name,
            )

    return {
        "whole_map_nulls": null_whole_map,
        "time_resolved_null_curves": null_time_curves,
    }


def _fit_similarity_predictor_map(
    config: Mapping[str, Any],
    *,
    pooled_inputs: PooledLMEEEGInputs,
    metadata_df: pd.DataFrame,
    metadata_provenance: dict[str, Any] | None,
    model_name: str,
    predictor_name: str,
    expected_map: PredictorMap,
) -> PredictorMap:
    result = fit_pooled_lmeeeg_model(
        config,
        pooled_inputs=pooled_inputs,
        config_section="lmeeeg",
        model_name=model_name,
        metadata_df=metadata_df,
        metadata_provenance=metadata_provenance,
    )
    null_map = _extract_predictor_map_from_result(
        result_betas=result.betas,
        column_names=result.column_names,
        predictor_name=predictor_name,
        model_name=model_name,
        value_field="beta",
        channel_names=result.channel_names,
        times=result.times,
    )
    _validate_predictor_maps(expected_map, null_map)
    return null_map


def _extract_predictor_map_from_result(
    *,
    result_betas: np.ndarray,
    column_names: list[str],
    predictor_name: str,
    model_name: str,
    value_field: str,
    channel_names: list[str],
    times: np.ndarray,
) -> PredictorMap:
    effect_name = _resolve_effect_name(column_names, predictor_name=predictor_name)
    effect_index = int(column_names.index(effect_name))
    return PredictorMap(
        predictor_name=predictor_name,
        model_name=model_name,
        effect_name=effect_name,
        value_field=value_field,
        values=np.asarray(result_betas[effect_index], dtype=float),
        channel_names=list(channel_names),
        times=np.asarray(times, dtype=float),
    )


def _resolve_whole_map_null_distribution(
    *,
    metric_name: WholeMapMetricName,
    map_a: np.ndarray,
    map_b: np.ndarray,
    settings: SimilaritySettings,
    refit_null_payload: dict[str, Any] | None,
    metric_offset: int,
) -> np.ndarray | None:
    if settings.inference.method == "none":
        return None
    if settings.inference.method == "sign_flip":
        observed_value, contributions = _compute_metric_with_contributions(map_a, map_b, metric_name=metric_name)
        if not np.isfinite(observed_value) or contributions.size == 0:
            return np.asarray([], dtype=float)
        return _sign_flip_null_distribution(
            contributions,
            n_permutations=settings.inference.n_permutations,
            seed=settings.inference.random_seed + metric_offset,
            aggregator="sum" if metric_name == "cosine" else "mean",
        )
    assert refit_null_payload is not None
    return np.asarray(refit_null_payload["whole_map_nulls"][metric_name], dtype=float)


def _resolve_time_resolved_null_payload(
    *,
    map_a: np.ndarray,
    map_b: np.ndarray,
    observed_curve: np.ndarray,
    settings: SimilaritySettings,
    refit_null_payload: dict[str, Any] | None,
) -> dict[str, np.ndarray | None]:
    if settings.inference.method == "none":
        return {
            "corrected_p_values": None,
            "null_curves": None,
            "null_max_distribution": None,
        }

    if settings.inference.method == "sign_flip":
        null_curves = _compute_sign_flip_time_resolved_null_curves(
            map_a,
            map_b,
            n_permutations=settings.inference.n_permutations,
            seed=settings.inference.random_seed + 1000,
        )
    else:
        assert refit_null_payload is not None
        null_curves = np.asarray(refit_null_payload["time_resolved_null_curves"], dtype=float)

    corrected_p_values, null_max_distribution = _compute_time_resolved_p_values(
        observed_curve=observed_curve,
        null_curves=null_curves,
        inferential=settings.inference,
    )
    return {
        "corrected_p_values": corrected_p_values,
        "null_curves": null_curves,
        "null_max_distribution": null_max_distribution,
    }


def _compute_time_resolved_curve(
    map_a: np.ndarray,
    map_b: np.ndarray,
    *,
    method: TimeResolvedMethodName,
) -> np.ndarray:
    if method != "pearson":
        raise ValueError(f"Unsupported time-resolved similarity method `{method}`.")
    n_times = int(map_a.shape[1])
    observed = np.full(n_times, np.nan, dtype=float)
    for time_index in range(n_times):
        observed[time_index] = _compute_metric(
            map_a[:, time_index],
            map_b[:, time_index],
            metric_name="pearson",
        )
    return observed


def _compute_time_resolved_p_values(
    *,
    observed_curve: np.ndarray,
    null_curves: np.ndarray,
    inferential: SimilarityInferentialSettings,
) -> tuple[np.ndarray, np.ndarray]:
    if not inferential.enabled:
        return np.full(observed_curve.shape, np.nan, dtype=float), np.asarray([], dtype=float)

    null_array = np.asarray(null_curves, dtype=float)
    if null_array.ndim != 2:
        raise ValueError(f"Expected 2D null time-resolved similarity curves, got shape {null_array.shape}.")

    corrected_p = np.full(observed_curve.shape, np.nan, dtype=float)
    if inferential.tfce.enabled:
        observed_tfce = _compute_tfce_1d(
            np.abs(observed_curve),
            h_power=inferential.tfce.h_power,
            e_power=inferential.tfce.e_power,
            h_start=inferential.tfce.h_start,
            dh=inferential.tfce.dh,
        )
        null_max = np.zeros(null_array.shape[0], dtype=float)
        for permutation_index in range(null_array.shape[0]):
            permuted_tfce = _compute_tfce_1d(
                np.abs(null_array[permutation_index]),
                h_power=inferential.tfce.h_power,
                e_power=inferential.tfce.e_power,
                h_start=inferential.tfce.h_start,
                dh=inferential.tfce.dh,
            )
            null_max[permutation_index] = float(np.nanmax(permuted_tfce)) if np.any(np.isfinite(permuted_tfce)) else 0.0
        for time_index, observed_value in enumerate(observed_tfce):
            if not np.isfinite(observed_value):
                continue
            corrected_p[time_index] = _permutation_p_value(
                observed=float(observed_value),
                null_distribution=null_max,
                tail="greater",
            )
        return corrected_p, null_max

    null_max = np.asarray([], dtype=float)
    for time_index, observed_value in enumerate(observed_curve):
        if not np.isfinite(observed_value):
            continue
        corrected_p[time_index] = _permutation_p_value(
            observed=float(observed_value),
            null_distribution=np.asarray(null_array[:, time_index], dtype=float),
            tail=inferential.tail,
        )
    return corrected_p, null_max


def _compute_sign_flip_time_resolved_null_curves(
    map_a: np.ndarray,
    map_b: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    n_times = int(map_a.shape[1])
    null_curves = np.full((n_permutations, n_times), np.nan, dtype=float)
    rng = np.random.default_rng(seed)
    for time_index in range(n_times):
        observed_value, contributions = _compute_metric_with_contributions(
            map_a[:, time_index],
            map_b[:, time_index],
            metric_name="pearson",
        )
        if not np.isfinite(observed_value) or contributions.size == 0:
            continue
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=float), size=(n_permutations, contributions.shape[0]), replace=True)
        null_curves[:, time_index] = np.mean(signs * contributions[np.newaxis, :], axis=1)
    return null_curves


def _build_similarity_permutation_grouping(
    metadata_df: pd.DataFrame,
    *,
    scheme: SimilarityPermutationScheme,
) -> SimilarityPermutationGrouping:
    if scheme.kind != "circular_shift_within_subject":
        raise ValueError(f"Unsupported similarity permutation scheme `{scheme.kind}`.")

    group_columns = list(scheme.group_by)
    if scheme.within_run and "run" not in group_columns:
        group_columns.append("run")
    missing_columns = [column_name for column_name in group_columns if column_name not in metadata_df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Similarity permutation grouping columns are missing from metadata: {missing_text}")
    if metadata_df.loc[:, group_columns].isna().any().any():
        raise ValueError("Similarity permutation grouping columns contain missing values.")

    labels = metadata_df.loc[:, group_columns].astype(str).agg("::".join, axis=1)
    label_values = labels.to_numpy(dtype=str, copy=False)
    unique_labels = tuple(str(label) for label in labels.drop_duplicates().tolist())
    group_indices = tuple(np.flatnonzero(label_values == label) for label in unique_labels)
    if any(group.size == 0 for group in group_indices):
        raise ValueError("Similarity permutation grouping produced an empty block, which is not supported.")
    return SimilarityPermutationGrouping(group_labels=unique_labels, group_indices=group_indices)


def _generate_similarity_permutation_indices(
    *,
    n_rows: int,
    grouping: SimilarityPermutationGrouping,
    rng: np.random.Generator,
) -> np.ndarray:
    permutation_indices = np.arange(n_rows, dtype=int)
    for group_indices in grouping.group_indices:
        if group_indices.size == 0:
            raise ValueError("Similarity permutation blocks must be non-empty.")
        if group_indices.size == 1:
            continue
        shift = int(rng.integers(1, group_indices.size))
        permutation_indices[group_indices] = np.roll(group_indices, shift)
    return permutation_indices


def _permute_similarity_predictor(
    metadata_df: pd.DataFrame,
    *,
    predictor_names: tuple[str, ...],
    permutation_indices: np.ndarray,
) -> pd.DataFrame:
    if permutation_indices.shape[0] != len(metadata_df):
        raise ValueError("Similarity permutation indices must match the metadata row count.")
    permuted = metadata_df.copy()
    for predictor_name in predictor_names:
        if predictor_name not in metadata_df.columns:
            raise ValueError(f"Similarity permutation predictor column `{predictor_name}` is missing from metadata.")
        permuted.loc[:, predictor_name] = metadata_df.loc[:, predictor_name].to_numpy(copy=True)[permutation_indices]
    return permuted


def _iter_similarity_permutations(
    metadata_df: pd.DataFrame,
    *,
    scheme: SimilarityPermutationScheme,
    grouping: SimilarityPermutationGrouping | None,
    n_permutations: int,
    random_seed: int,
):
    resolved_grouping = grouping if grouping is not None else _build_similarity_permutation_grouping(metadata_df, scheme=scheme)
    rng = np.random.default_rng(random_seed)
    for permutation_index in range(n_permutations):
        yield permutation_index, _generate_similarity_permutation_indices(
            n_rows=len(metadata_df),
            grouping=resolved_grouping,
            rng=rng,
        )


def _build_similarity_metadata(
    *,
    settings: SimilaritySettings,
    map_a: PredictorMap,
    map_b: PredictorMap,
    whole_map_nulls: Mapping[str, np.ndarray],
    null_curves: np.ndarray | None,
    null_max_distribution: np.ndarray | None,
) -> dict[str, Any]:
    return {
        "analysis_name": settings.analysis_name,
        "predictor_a": map_a.predictor_name,
        "predictor_b": map_b.predictor_name,
        "underlying_lmeeeg_models": {
            "predictor_a": map_a.model_name,
            "predictor_b": map_b.model_name,
        },
        "inference": {
            "descriptive_only": not settings.inference.enabled,
            "method": settings.inference.method,
            "p_value_source": None if not settings.inference.enabled else settings.inference.method,
            "n_permutations": int(settings.inference.n_permutations) if settings.inference.enabled else 0,
            "random_seed": int(settings.inference.random_seed),
            "tail": settings.inference.tail,
            "tfce": {
                "enabled": settings.inference.tfce.enabled,
                "e_power": settings.inference.tfce.e_power,
                "h_power": settings.inference.tfce.h_power,
                "dh": settings.inference.tfce.dh,
                "h_start": settings.inference.tfce.h_start,
            },
            "permutation_scheme": {
                "kind": settings.inference.permutation_scheme.kind,
                "group_by": list(settings.inference.permutation_scheme.group_by),
                "within_run": settings.inference.permutation_scheme.within_run,
            },
            "whole_map_null_metrics": sorted(whole_map_nulls),
            "time_resolved_null_shape": None if null_curves is None else list(np.asarray(null_curves).shape),
            "time_resolved_null_max_shape": None if null_max_distribution is None else list(np.asarray(null_max_distribution).shape),
            "null_hypothesis": (
                "Similarity is evaluated against null maps generated by disrupting predictor-to-EEG alignment "
                "and refitting the lmeEEG models. This does not test whether the observed maps are equal."
            ),
            "method_note": (
                "Similarity inference should be based on null maps generated by disrupting predictor-to-EEG alignment "
                "and refitting the lmeEEG models; post hoc sign-flip over map features is kept only as a legacy/fast approximation."
            ),
        },
    }


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


def _compute_metric(
    map_a: np.ndarray,
    map_b: np.ndarray,
    *,
    metric_name: WholeMapMetricName,
) -> float:
    observed_value, _contributions = _compute_metric_with_contributions(map_a, map_b, metric_name=metric_name)
    return float(observed_value)


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


def _normalize_inference_method(method_name: Any) -> SimilarityInferenceMethod:
    normalized = str(method_name).lower()
    if normalized not in {"none", "sign_flip", "refit_permutation"}:
        raise ValueError(f"Unsupported similarity inference method `{normalized}`.")
    return normalized  # type: ignore[return-value]


def _sanitize_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value)) or "analysis"


def _should_log_similarity_progress(permutation_index: int, *, total: int) -> bool:
    if total <= 10:
        return True
    stride = max(1, total // 10)
    return (permutation_index + 1) % stride == 0 or (permutation_index + 1) == total
