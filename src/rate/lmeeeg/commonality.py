"""Independent commonality analysis for pooled lmeEEG models."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from rate.design.formulas import ModelSpec, load_model_spec
from rate.design.matrices import build_design_matrix
from rate.lmeeeg.mass_model import LMEEEGResult, fit_lmeeeg_mass_model
from rate.lmeeeg.pooling import PooledLMEEEGInputs, load_pooled_lmeeeg_inputs
from rate.lmeeeg.trial_data import build_lmeeeg_trial_data_from_arrays, resolve_random_effect_group
from rate.paths import ensure_parent_dir, resolve_out_dir
from rate.util import KEY_PACKAGES, get_package_versions, get_python_version, resolve_random_seed

DEFAULT_LMEEEG_COMMONALITY_SECTION = "lmeeeg_commonality"


@dataclass(frozen=True)
class LMEEEGCommonalitySettings:
    """Resolved config for the commonality analysis."""

    source_section: str
    source_model: str
    output_dirname: str
    self_reduced_formula: str
    other_reduced_formula: str
    covariate_reduced_formula: str
    n_boot: int
    alpha: float
    seed: int
    adjusted_r2: bool
    roi_source: str
    roi_mask_path: Path | None
    roi_summary_path: Path | None
    calibration_permutations: int
    n_jobs: int
    chunk_size: int


@dataclass(frozen=True)
class LMEEEGCommonalityOutputPaths:
    """Canonical output paths for the commonality analysis."""

    root_dir: Path
    point_estimate_output_path: Path
    roi_mask_output_path: Path
    bca_lower_output_path: Path
    bca_upper_output_path: Path
    bootstrap_p_output_path: Path
    calibration_floor_output_path: Path
    map_wide_p_output_path: Path
    roi_table_output_path: Path
    bootstrap_roi_output_path: Path
    metadata_output_path: Path
    summary_output_path: Path


@dataclass(frozen=True)
class CommonalityObservedFit:
    """Observed-data fit pieces reused by bootstrap and summaries."""

    fit_result: LMEEEGResult
    trial_data: Any
    full_model_spec: ModelSpec
    column_groups: dict[str, np.ndarray]
    point_estimate_flat: np.ndarray
    design_matrix: np.ndarray
    group_values: np.ndarray


@dataclass(frozen=True)
class _CommonalityWorkerContext:
    """Shared worker context for bootstrap and jackknife refits."""

    config: dict[str, Any]
    pooled_eeg_data: np.ndarray
    pooled_metadata_df: pd.DataFrame
    channel_names: list[str]
    times: np.ndarray
    metadata_provenance: dict[str, Any] | None
    config_section: str
    model_name: str
    column_groups: dict[str, np.ndarray]
    adjusted_r2: bool
    roi_indices: np.ndarray
    roi_source: str
    group_column: str
    unique_groups: list[str]
    group_to_indices: dict[str, np.ndarray]


_COMMONALITY_WORKER_CONTEXT: _CommonalityWorkerContext | None = None


def resolve_lmeeeg_commonality_settings(
    config: Mapping[str, Any],
    *,
    config_section: str = DEFAULT_LMEEEG_COMMONALITY_SECTION,
) -> LMEEEGCommonalitySettings:
    """Resolve commonality-analysis settings from config."""
    section = dict(config.get(config_section) or {})
    formulas = dict(section.get("formulas") or {})
    bootstrap = dict(section.get("bootstrap") or {})
    roi = dict(section.get("roi") or {})
    calibration = dict(section.get("calibration") or {})

    self_reduced = _normalize_reduced_formula(formulas.get("self_reduced"))
    other_reduced = _normalize_reduced_formula(formulas.get("other_reduced"))
    covariate_reduced = _normalize_reduced_formula(formulas.get("covariate_reduced"))
    roi_source = str(roi.get("source", "self_rate_significant"))
    if roi_source not in {"self_rate_significant", "explicit_mask_path", "map_wide"}:
        raise ValueError(
            "`lmeeeg_commonality.roi.source` must be one of "
            "`self_rate_significant`, `explicit_mask_path`, or `map_wide`."
        )
    roi_mask_path = None if roi.get("mask_path") in (None, "") else Path(str(roi.get("mask_path")))
    roi_summary_path = None if roi.get("summary_path") in (None, "") else Path(str(roi.get("summary_path")))

    if roi_source == "explicit_mask_path" and roi_mask_path is None:
        raise ValueError("`lmeeeg_commonality.roi.mask_path` is required when `roi.source=explicit_mask_path`.")

    seed = int(bootstrap.get("seed", resolve_random_seed(config)))
    n_boot = int(bootstrap.get("n_boot", 5000))
    if n_boot < 1:
        raise ValueError("`lmeeeg_commonality.bootstrap.n_boot` must be >= 1.")
    alpha = float(bootstrap.get("alpha", 0.05))
    if not (0.0 < alpha < 1.0):
        raise ValueError("`lmeeeg_commonality.bootstrap.alpha` must be strictly between 0 and 1.")

    return LMEEEGCommonalitySettings(
        source_section=str(section.get("source_section", "lmeeeg")),
        source_model=str(section.get("source_model", "both_effect")),
        output_dirname=str(section.get("output_dirname", config_section)),
        self_reduced_formula=self_reduced,
        other_reduced_formula=other_reduced,
        covariate_reduced_formula=covariate_reduced,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
        adjusted_r2=bool(section.get("adjusted_r2", True)),
        roi_source=roi_source,
        roi_mask_path=roi_mask_path,
        roi_summary_path=roi_summary_path,
        calibration_permutations=int(calibration.get("n_permutations", 32)),
        n_jobs=max(1, int(section.get("n_jobs", 1))),
        chunk_size=max(1, int(section.get("chunk_size", 256))),
    )


def build_lmeeeg_commonality_output_paths(
    config: Mapping[str, Any],
    *,
    config_section: str = DEFAULT_LMEEEG_COMMONALITY_SECTION,
) -> LMEEEGCommonalityOutputPaths:
    """Build canonical output paths for the commonality analysis."""
    settings = resolve_lmeeeg_commonality_settings(config, config_section=config_section)
    root_dir = resolve_out_dir(config) / "stats" / settings.output_dirname
    return LMEEEGCommonalityOutputPaths(
        root_dir=root_dir,
        point_estimate_output_path=root_dir / "point_estimate_commonality.npy",
        roi_mask_output_path=root_dir / "roi_mask.npy",
        bca_lower_output_path=root_dir / "bca_lower.npy",
        bca_upper_output_path=root_dir / "bca_upper.npy",
        bootstrap_p_output_path=root_dir / "bootstrap_p.npy",
        calibration_floor_output_path=root_dir / "calibration_floor.npy",
        map_wide_p_output_path=root_dir / "map_wide_p.npy",
        roi_table_output_path=root_dir / "roi_feature_summary.parquet",
        bootstrap_roi_output_path=root_dir / "bootstrap_roi_distribution.npy",
        metadata_output_path=root_dir / "run_metadata.json",
        summary_output_path=resolve_out_dir(config) / "stats" / f"{config_section}_group_stats.json",
    )


def run_commonality_analysis(
    config: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    *,
    config_section: str = DEFAULT_LMEEEG_COMMONALITY_SECTION,
    trial_regressors_path: Path | None = None,
    n_jobs_override: int | None = None,
) -> dict[str, Any]:
    """Run the full commonality analysis and write outputs."""
    settings = resolve_lmeeeg_commonality_settings(config, config_section=config_section)
    if n_jobs_override is not None:
        settings = replace(settings, n_jobs=max(1, int(n_jobs_override)))
    output_paths = build_lmeeeg_commonality_output_paths(config, config_section=config_section)
    pooled_inputs = load_pooled_lmeeeg_inputs(
        config,
        rows,
        config_section=settings.source_section,
        model_name=settings.source_model,
        trial_regressors_path=trial_regressors_path,
    )
    observed = _fit_commonality_observed(config, pooled_inputs=pooled_inputs, settings=settings)
    point_map = observed.point_estimate_flat.reshape(len(observed.trial_data.channel_names), len(observed.trial_data.times))
    roi_mask = _resolve_roi_mask(config, settings=settings, observed=observed, point_map=point_map)
    roi_indices = np.flatnonzero(roi_mask.reshape(-1))
    if roi_indices.size == 0:
        raise ValueError("Resolved ROI mask contains no selected features.")

    rng = np.random.default_rng(settings.seed)
    bootstrap_roi, max_null, distinct_groups = _run_subject_bootstrap(
        config,
        pooled_inputs=pooled_inputs,
        observed=observed,
        roi_indices=roi_indices,
        settings=settings,
        rng=rng,
    )
    jackknife_roi = _run_subject_jackknife(
        config,
        pooled_inputs=pooled_inputs,
        observed=observed,
        roi_indices=roi_indices,
        settings=settings,
    )
    roi_point = observed.point_estimate_flat[roi_indices]
    bca_lower_roi, bca_upper_roi = _compute_bca_bounds(
        observed=roi_point,
        bootstrap=bootstrap_roi,
        jackknife=jackknife_roi,
        alpha=settings.alpha,
    )
    bootstrap_p_roi = np.mean(bootstrap_roi <= 0.0, axis=0)
    calibration_floor_flat = _estimate_calibration_floor(
        fit_result=observed.fit_result,
        column_groups=observed.column_groups,
        group_values=observed.group_values,
        adjusted_r2=settings.adjusted_r2,
        n_permutations=settings.calibration_permutations,
        rng=rng,
    )
    design_companion = _compute_design_companion(
        design_matrix=observed.design_matrix,
        group_values=observed.group_values,
        self_columns=observed.column_groups["self"],
        other_columns=observed.column_groups["other"],
        covariate_columns=observed.column_groups["covariate"],
        alpha=settings.alpha,
        seed=settings.seed,
    )

    shape = point_map.shape
    lower_map = np.full(shape, np.nan, dtype=float)
    upper_map = np.full(shape, np.nan, dtype=float)
    bootstrap_p_map = np.full(shape, np.nan, dtype=float)
    calibration_floor_map = calibration_floor_flat.reshape(shape)
    lower_map.reshape(-1)[roi_indices] = bca_lower_roi
    upper_map.reshape(-1)[roi_indices] = bca_upper_roi
    bootstrap_p_map.reshape(-1)[roi_indices] = bootstrap_p_roi

    map_wide_p_map = np.full(shape, np.nan, dtype=float)
    if settings.roi_source == "map_wide":
        map_wide_p_map = _compute_map_wide_p(point_flat=observed.point_estimate_flat, max_null=max_null).reshape(shape)

    _write_commonality_outputs(
        output_paths=output_paths,
        observed=observed,
        roi_mask=roi_mask,
        lower_map=lower_map,
        upper_map=upper_map,
        bootstrap_p_map=bootstrap_p_map,
        calibration_floor_map=calibration_floor_map,
        map_wide_p_map=map_wide_p_map,
        bootstrap_roi=bootstrap_roi,
        settings=settings,
        design_companion=design_companion,
        rows=rows,
        distinct_groups=distinct_groups,
        roi_indices=roi_indices,
    )
    return json.loads(output_paths.summary_output_path.read_text(encoding="utf-8"))


def _fit_commonality_observed(
    config: Mapping[str, Any],
    *,
    pooled_inputs: PooledLMEEEGInputs,
    settings: LMEEEGCommonalitySettings,
) -> CommonalityObservedFit:
    full_model_spec = load_model_spec(config, section=settings.source_section, model_name=settings.source_model)
    trial_data = build_lmeeeg_trial_data_from_arrays(
        eeg_data=pooled_inputs.eeg_data,
        metadata_df=pooled_inputs.metadata_df,
        channel_names=pooled_inputs.channel_names,
        times=pooled_inputs.times,
        config=config,
        config_section=settings.source_section,
        model_name=settings.source_model,
        metadata_provenance=pooled_inputs.metadata_provenance,
    )
    fit_result = fit_lmeeeg_mass_model(
        trial_data,
        config,
        config_section=settings.source_section,
        model_name=settings.source_model,
    )
    column_groups = _resolve_column_groups(
        metadata_df=trial_data.trial_metadata,
        full_design_result=trial_data.design_result,
        full_model_spec=full_model_spec,
        self_reduced_formula=settings.self_reduced_formula,
        other_reduced_formula=settings.other_reduced_formula,
        covariate_reduced_formula=settings.covariate_reduced_formula,
    )
    point_estimate_flat = _compute_commonality_from_fit_result(
        fit_result=fit_result,
        column_groups=column_groups,
        adjusted_r2=settings.adjusted_r2,
    )
    group_values = trial_data.trial_metadata[trial_data.group_column].to_numpy(copy=True)
    return CommonalityObservedFit(
        fit_result=fit_result,
        trial_data=trial_data,
        full_model_spec=full_model_spec,
        column_groups=column_groups,
        point_estimate_flat=point_estimate_flat,
        design_matrix=np.asarray(trial_data.design_result.design_matrix, dtype=float),
        group_values=group_values,
    )


def _resolve_column_groups(
    *,
    metadata_df: pd.DataFrame,
    full_design_result,
    full_model_spec: ModelSpec,
    self_reduced_formula: str,
    other_reduced_formula: str,
    covariate_reduced_formula: str,
) -> dict[str, np.ndarray]:
    full_columns = list(full_design_result.column_names)
    full_non_intercept = [name for name in full_columns if name != "Intercept"]
    groups = {
        "self": _resolve_block_columns(
            metadata_df=metadata_df,
            full_columns=full_columns,
            full_non_intercept=full_non_intercept,
            full_model_spec=full_model_spec,
            reduced_formula=self_reduced_formula,
            block_name="self",
        ),
        "other": _resolve_block_columns(
            metadata_df=metadata_df,
            full_columns=full_columns,
            full_non_intercept=full_non_intercept,
            full_model_spec=full_model_spec,
            reduced_formula=other_reduced_formula,
            block_name="other",
        ),
        "covariate": _resolve_block_columns(
            metadata_df=metadata_df,
            full_columns=full_columns,
            full_non_intercept=full_non_intercept,
            full_model_spec=full_model_spec,
            reduced_formula=covariate_reduced_formula,
            block_name="covariate",
        ),
    }
    used = np.concatenate([groups["self"], groups["other"], groups["covariate"]])
    if len(np.unique(used)) != len(full_non_intercept):
        missing = sorted(set(range(len(full_columns))) - {0} - set(used.tolist()))
        overlap_count = int(len(used) - len(np.unique(used)))
        raise ValueError(
            "Commonality block decomposition must partition the non-intercept full design exactly. "
            f"Missing columns={missing}; overlap_count={overlap_count}."
        )
    return groups


def _resolve_block_columns(
    *,
    metadata_df: pd.DataFrame,
    full_columns: list[str],
    full_non_intercept: list[str],
    full_model_spec: ModelSpec,
    reduced_formula: str,
    block_name: str,
) -> np.ndarray:
    reduced_spec = replace(full_model_spec, formula=reduced_formula, outcome=None, test_predictors=())
    reduced_design = build_design_matrix(metadata_df, model_spec=reduced_spec)
    if len(reduced_design.row_index) != len(metadata_df):
        raise ValueError(
            f"Reduced formula for commonality block `{block_name}` changed row filtering. "
            "Blocks must be defined on the already-filtered full-model trial table."
        )
    reduced_columns = set(reduced_design.column_names)
    if "Intercept" in full_columns and "Intercept" not in reduced_columns:
        raise ValueError(
            f"Reduced formula for commonality block `{block_name}` removed the intercept. "
            "Reduced formulas must preserve the full-model intercept convention."
        )
    reduced_non_intercept = {name for name in reduced_columns if name != "Intercept"}
    missing_from_full = sorted(reduced_non_intercept - set(full_non_intercept))
    if missing_from_full:
        raise ValueError(
            f"Reduced formula for commonality block `{block_name}` generated columns not present in the full design: "
            + ", ".join(missing_from_full)
        )
    block_columns = [full_columns.index(name) for name in full_non_intercept if name not in reduced_non_intercept]
    if not block_columns:
        raise ValueError(f"Commonality block `{block_name}` resolved to zero columns.")
    return np.asarray(block_columns, dtype=int)


def _normalize_reduced_formula(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Commonality reduced formulas must be non-empty strings.")
    text = str(value).strip()
    if "~" not in text:
        return f"~ {text}"
    lhs, rhs = text.split("~", maxsplit=1)
    if lhs.strip():
        return f"~ {rhs.strip()}"
    return f"~ {rhs.strip()}"


def _compute_commonality_from_fit_result(
    *,
    fit_result: LMEEEGResult,
    column_groups: Mapping[str, np.ndarray],
    adjusted_r2: bool,
) -> np.ndarray:
    design_matrix = np.asarray(fit_result.design_result.design_matrix, dtype=float)
    y_flat = np.asarray(fit_result.residualised_data, dtype=float).reshape(design_matrix.shape[0], -1)
    include_intercept = [fit_result.column_names.index("Intercept")] if "Intercept" in fit_result.column_names else []
    covariate_columns = np.concatenate([np.asarray(include_intercept, dtype=int), column_groups["covariate"]])
    self_cov_columns = np.concatenate([np.asarray(include_intercept, dtype=int), column_groups["self"], column_groups["covariate"]])
    other_cov_columns = np.concatenate([np.asarray(include_intercept, dtype=int), column_groups["other"], column_groups["covariate"]])
    full_columns = np.arange(design_matrix.shape[1], dtype=int)

    r2_cov = _compute_mass_univariate_r2(design_matrix[:, covariate_columns], y_flat, adjusted=adjusted_r2)
    r2_self_cov = _compute_mass_univariate_r2(design_matrix[:, self_cov_columns], y_flat, adjusted=adjusted_r2)
    r2_other_cov = _compute_mass_univariate_r2(design_matrix[:, other_cov_columns], y_flat, adjusted=adjusted_r2)
    r2_full = _compute_mass_univariate_r2(design_matrix[:, full_columns], y_flat, adjusted=adjusted_r2)
    return r2_self_cov + r2_other_cov - r2_full - r2_cov


def _compute_mass_univariate_r2(
    design_matrix: np.ndarray,
    y_flat: np.ndarray,
    *,
    adjusted: bool,
) -> np.ndarray:
    x = np.asarray(design_matrix, dtype=float)
    y = np.asarray(y_flat, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("OLS inputs must be 2D and aligned on rows.")
    rank = int(np.linalg.matrix_rank(x))
    if rank < x.shape[1]:
        raise ValueError(f"Rank-deficient OLS design detected in commonality analysis: rank={rank}, columns={x.shape[1]}.")
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residual = y - fitted
    sse = np.sum(residual * residual, axis=0)
    has_intercept = bool(np.any(np.all(np.isclose(x, 1.0), axis=0)))
    if has_intercept:
        centered = y - np.mean(y, axis=0, keepdims=True)
        tss = np.sum(centered * centered, axis=0)
        dof_total = x.shape[0] - 1
    else:
        tss = np.sum(y * y, axis=0)
        dof_total = x.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - (sse / tss)
    if not adjusted:
        return r2
    dof_resid = x.shape[0] - rank
    if dof_resid <= 0 or dof_total <= 0:
        return np.full(y.shape[1], np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        adjusted_r2 = 1.0 - ((sse / dof_resid) / (tss / dof_total))
    return adjusted_r2


def _resolve_roi_mask(
    config: Mapping[str, Any],
    *,
    settings: LMEEEGCommonalitySettings,
    observed: CommonalityObservedFit,
    point_map: np.ndarray,
) -> np.ndarray:
    if settings.roi_source == "map_wide":
        return np.ones(point_map.shape, dtype=bool)
    if settings.roi_source == "explicit_mask_path":
        mask = np.asarray(np.load(settings.roi_mask_path), dtype=bool)
        if mask.shape != point_map.shape:
            raise ValueError(f"Explicit ROI mask shape {mask.shape} does not match point-estimate shape {point_map.shape}.")
        return mask

    summary_path = settings.roi_summary_path or (resolve_out_dir(config) / "stats" / "lmeeeg_group_stats.json")
    if not summary_path.exists():
        nested_path = resolve_out_dir(config) / "stats" / "lmeeeg_nested_group_stats.json"
        if nested_path.exists():
            summary_path = nested_path
    if not summary_path.exists():
        raise FileNotFoundError(f"Unable to locate ROI summary for `self_rate_significant`: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    corrected_p_path = _resolve_self_rate_corrected_p_path(payload)
    corrected_p = np.asarray(np.load(corrected_p_path), dtype=float)
    if corrected_p.shape != point_map.shape:
        raise ValueError(
            f"Resolved self-rate corrected-p map shape {corrected_p.shape} does not match commonality map shape {point_map.shape}."
        )
    return corrected_p < settings.alpha


def _resolve_self_rate_corrected_p_path(summary_payload: Mapping[str, Any]) -> Path:
    analyses = summary_payload.get("analyses")
    if not isinstance(analyses, Sequence):
        raise ValueError("ROI summary payload does not contain an `analyses` sequence.")
    for analysis in analyses:
        if not isinstance(analysis, Mapping):
            continue
        effects = analysis.get("effects")
        if isinstance(effects, Sequence):
            for effect in effects:
                if isinstance(effect, Mapping) and str(effect.get("predictor")) == "self_rate":
                    return Path(str(effect["corrected_p_values"]))
        nested_tests = analysis.get("nested_tests")
        if isinstance(nested_tests, Sequence):
            for item in nested_tests:
                if isinstance(item, Mapping) and str(item.get("tested_predictor")) == "self_rate":
                    return Path(str(item["corrected_p_values"]))
    raise ValueError("Unable to resolve a self-rate significance map from the configured ROI summary.")


def _run_subject_bootstrap(
    config: Mapping[str, Any],
    *,
    pooled_inputs: PooledLMEEEGInputs,
    observed: CommonalityObservedFit,
    roi_indices: np.ndarray,
    settings: LMEEEGCommonalitySettings,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    metadata_df = pooled_inputs.metadata_df.reset_index(drop=True).copy()
    group_column = resolve_random_effect_group(config, config_section=settings.source_section, model_name=settings.source_model)
    unique_groups = metadata_df[group_column].astype(str).tolist()
    unique_groups = list(dict.fromkeys(unique_groups))
    group_to_indices = {
        group_value: np.flatnonzero(metadata_df[group_column].astype(str).to_numpy() == group_value)
        for group_value in unique_groups
    }
    bootstrap_roi = np.full((settings.n_boot, roi_indices.size), np.nan, dtype=float)
    max_null = np.full(settings.n_boot, np.nan, dtype=float)
    distinct_group_counts: list[int] = []
    draw_matrix = rng.integers(0, len(unique_groups), size=(settings.n_boot, len(unique_groups)))
    worker_context = _build_worker_context(
        config=config,
        pooled_inputs=pooled_inputs,
        observed=observed,
        roi_indices=roi_indices,
        settings=settings,
        group_column=group_column,
        unique_groups=unique_groups,
        group_to_indices=group_to_indices,
    )
    if settings.n_jobs == 1:
        _set_commonality_worker_context(worker_context)
        results = [_run_bootstrap_iteration((boot_index, draw_matrix[boot_index])) for boot_index in range(settings.n_boot)]
    else:
        with ProcessPoolExecutor(
            max_workers=settings.n_jobs,
            initializer=_set_commonality_worker_context,
            initargs=(worker_context,),
        ) as executor:
            results = list(
                executor.map(
                    _run_bootstrap_iteration,
                    ((boot_index, draw_matrix[boot_index]) for boot_index in range(settings.n_boot)),
                    chunksize=settings.chunk_size,
                )
            )
    for boot_index, roi_values, max_value, distinct_count in results:
        bootstrap_roi[int(boot_index), :] = roi_values
        max_null[int(boot_index)] = max_value
        distinct_group_counts.append(int(distinct_count))
    return bootstrap_roi, max_null, distinct_group_counts


def _run_subject_jackknife(
    config: Mapping[str, Any],
    *,
    pooled_inputs: PooledLMEEEGInputs,
    observed: CommonalityObservedFit,
    roi_indices: np.ndarray,
    settings: LMEEEGCommonalitySettings,
) -> np.ndarray:
    metadata_df = pooled_inputs.metadata_df.reset_index(drop=True).copy()
    group_column = resolve_random_effect_group(config, config_section=settings.source_section, model_name=settings.source_model)
    unique_groups = list(dict.fromkeys(metadata_df[group_column].astype(str).tolist()))
    jackknife = np.full((len(unique_groups), roi_indices.size), np.nan, dtype=float)
    group_to_indices = {
        group_value: np.flatnonzero(metadata_df[group_column].astype(str).to_numpy() == group_value)
        for group_value in unique_groups
    }
    worker_context = _build_worker_context(
        config=config,
        pooled_inputs=pooled_inputs,
        observed=observed,
        roi_indices=roi_indices,
        settings=settings,
        group_column=group_column,
        unique_groups=unique_groups,
        group_to_indices=group_to_indices,
    )
    if settings.n_jobs == 1:
        _set_commonality_worker_context(worker_context)
        results = [_run_jackknife_iteration((group_index, dropped_group)) for group_index, dropped_group in enumerate(unique_groups)]
    else:
        with ProcessPoolExecutor(
            max_workers=settings.n_jobs,
            initializer=_set_commonality_worker_context,
            initargs=(worker_context,),
        ) as executor:
            results = list(
                executor.map(
                    _run_jackknife_iteration,
                    ((group_index, dropped_group) for group_index, dropped_group in enumerate(unique_groups)),
                    chunksize=1,
                )
            )
    for group_index, roi_values in results:
        jackknife[int(group_index), :] = roi_values
    return jackknife


def _build_worker_context(
    *,
    config: Mapping[str, Any],
    pooled_inputs: PooledLMEEEGInputs,
    observed: CommonalityObservedFit,
    roi_indices: np.ndarray,
    settings: LMEEEGCommonalitySettings,
    group_column: str,
    unique_groups: list[str],
    group_to_indices: dict[str, np.ndarray],
) -> _CommonalityWorkerContext:
    return _CommonalityWorkerContext(
        config=dict(config),
        pooled_eeg_data=np.asarray(pooled_inputs.eeg_data, dtype=float),
        pooled_metadata_df=pooled_inputs.metadata_df.reset_index(drop=True).copy(),
        channel_names=list(pooled_inputs.channel_names),
        times=np.asarray(pooled_inputs.times, dtype=float),
        metadata_provenance=pooled_inputs.metadata_provenance,
        config_section=settings.source_section,
        model_name=settings.source_model,
        column_groups={name: np.asarray(value, dtype=int) for name, value in observed.column_groups.items()},
        adjusted_r2=bool(settings.adjusted_r2),
        roi_indices=np.asarray(roi_indices, dtype=int),
        roi_source=str(settings.roi_source),
        group_column=str(group_column),
        unique_groups=list(unique_groups),
        group_to_indices={str(key): np.asarray(value, dtype=int) for key, value in group_to_indices.items()},
    )


def _set_commonality_worker_context(context: _CommonalityWorkerContext) -> None:
    global _COMMONALITY_WORKER_CONTEXT
    _COMMONALITY_WORKER_CONTEXT = context


def _run_bootstrap_iteration(task: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, float, int]:
    context = _require_commonality_worker_context()
    boot_index, drawn = task
    eeg_chunks: list[np.ndarray] = []
    metadata_chunks: list[pd.DataFrame] = []
    for draw_position, group_idx in enumerate(np.asarray(drawn, dtype=int)):
        original_group = context.unique_groups[int(group_idx)]
        row_indices = context.group_to_indices[original_group]
        metadata_chunk = context.pooled_metadata_df.iloc[row_indices].copy().reset_index(drop=True)
        metadata_chunk.loc[:, context.group_column] = f"{original_group}__boot{int(boot_index):05d}_{draw_position:04d}"
        eeg_chunks.append(np.asarray(context.pooled_eeg_data[row_indices, :, :], dtype=float))
        metadata_chunks.append(metadata_chunk)
    trial_data = build_lmeeeg_trial_data_from_arrays(
        eeg_data=np.concatenate(eeg_chunks, axis=0),
        metadata_df=pd.concat(metadata_chunks, axis=0, ignore_index=True),
        channel_names=context.channel_names,
        times=context.times,
        config=context.config,
        config_section=context.config_section,
        model_name=context.model_name,
        metadata_provenance=context.metadata_provenance,
    )
    fit_result = fit_lmeeeg_mass_model(
        trial_data,
        context.config,
        config_section=context.config_section,
        model_name=context.model_name,
    )
    c_flat = _compute_commonality_from_fit_result(
        fit_result=fit_result,
        column_groups=context.column_groups,
        adjusted_r2=context.adjusted_r2,
    )
    max_value = float(np.nanmax(c_flat)) if context.roi_source == "map_wide" else np.nan
    return int(boot_index), c_flat[context.roi_indices], max_value, int(len(metadata_chunks))


def _run_jackknife_iteration(task: tuple[int, str]) -> tuple[int, np.ndarray]:
    context = _require_commonality_worker_context()
    group_index, dropped_group = task
    keep_mask = context.pooled_metadata_df[context.group_column].astype(str).to_numpy() != str(dropped_group)
    trial_data = build_lmeeeg_trial_data_from_arrays(
        eeg_data=np.asarray(context.pooled_eeg_data[keep_mask, :, :], dtype=float),
        metadata_df=context.pooled_metadata_df.loc[keep_mask].reset_index(drop=True),
        channel_names=context.channel_names,
        times=context.times,
        config=context.config,
        config_section=context.config_section,
        model_name=context.model_name,
        metadata_provenance=context.metadata_provenance,
    )
    fit_result = fit_lmeeeg_mass_model(
        trial_data,
        context.config,
        config_section=context.config_section,
        model_name=context.model_name,
    )
    c_flat = _compute_commonality_from_fit_result(
        fit_result=fit_result,
        column_groups=context.column_groups,
        adjusted_r2=context.adjusted_r2,
    )
    return int(group_index), c_flat[context.roi_indices]


def _require_commonality_worker_context() -> _CommonalityWorkerContext:
    if _COMMONALITY_WORKER_CONTEXT is None:
        raise RuntimeError("Commonality worker context is not initialized.")
    return _COMMONALITY_WORKER_CONTEXT


def _compute_bca_bounds(
    *,
    observed: np.ndarray,
    bootstrap: np.ndarray,
    jackknife: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1.0 / (2.0 * max(1, bootstrap.shape[0]))
    prop_less = np.mean(bootstrap < observed[None, :], axis=0)
    prop_less = np.clip(prop_less, eps, 1.0 - eps)
    z0 = norm.ppf(prop_less)
    valid_counts = np.sum(np.isfinite(jackknife), axis=0)
    jack_sums = np.nansum(jackknife, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        jack_mean = jack_sums / valid_counts
    diffs = jack_mean[None, :] - jackknife
    numerator = np.nansum(diffs**3, axis=0)
    denominator = 6.0 * np.power(np.nansum(diffs**2, axis=0), 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        acceleration = numerator / denominator
    acceleration = np.where(np.isfinite(acceleration), acceleration, 0.0)
    lower = _bca_percentile(bootstrap=bootstrap, z0=z0, acceleration=acceleration, probability=alpha)
    upper = _bca_percentile(bootstrap=bootstrap, z0=z0, acceleration=acceleration, probability=1.0 - alpha)
    return lower, upper


def _bca_percentile(
    *,
    bootstrap: np.ndarray,
    z0: np.ndarray,
    acceleration: np.ndarray,
    probability: float,
) -> np.ndarray:
    z_alpha = norm.ppf(probability)
    denom = 1.0 - (acceleration * (z0 + z_alpha))
    with np.errstate(divide="ignore", invalid="ignore"):
        adjusted = norm.cdf(z0 + ((z0 + z_alpha) / denom))
    adjusted = np.clip(adjusted, 0.0, 1.0)
    quantiles = np.full(bootstrap.shape[1], np.nan, dtype=float)
    for feature_index, quantile in enumerate(adjusted):
        if not np.isfinite(quantile):
            quantile = probability
        quantiles[feature_index] = np.nanquantile(bootstrap[:, feature_index], quantile)
    return quantiles


def _estimate_calibration_floor(
    *,
    fit_result: LMEEEGResult,
    column_groups: Mapping[str, np.ndarray],
    group_values: np.ndarray,
    adjusted_r2: bool,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.asarray(fit_result.design_result.design_matrix, dtype=float).copy()
    y_flat = np.asarray(fit_result.residualised_data, dtype=float).reshape(x.shape[0], -1)
    other_columns = np.asarray(column_groups["other"], dtype=int)
    unique_groups = list(dict.fromkeys(group_values.astype(str).tolist()))
    estimates = np.full((max(1, n_permutations), y_flat.shape[1]), np.nan, dtype=float)
    for perm_index in range(max(1, n_permutations)):
        permuted_x = x.copy()
        for group_value in unique_groups:
            group_mask = group_values.astype(str) == str(group_value)
            group_rows = np.flatnonzero(group_mask)
            if group_rows.size <= 1:
                continue
            order = rng.permutation(group_rows.size)
            permuted_x[np.ix_(group_rows, other_columns)] = permuted_x[np.ix_(group_rows[order], other_columns)]
        estimates[perm_index, :] = _compute_commonality_from_design_and_response(
            design_matrix=permuted_x,
            y_flat=y_flat,
            column_groups=column_groups,
            adjusted_r2=adjusted_r2,
        )
    return np.nanmean(estimates, axis=0)


def _compute_commonality_from_design_and_response(
    *,
    design_matrix: np.ndarray,
    y_flat: np.ndarray,
    column_groups: Mapping[str, np.ndarray],
    adjusted_r2: bool,
) -> np.ndarray:
    intercept_indices = np.flatnonzero(np.all(np.isclose(design_matrix, 1.0), axis=0))
    include_intercept = intercept_indices[:1].astype(int)
    covariate_columns = np.concatenate([np.asarray(include_intercept, dtype=int), column_groups["covariate"]])
    self_cov_columns = np.concatenate([np.asarray(include_intercept, dtype=int), column_groups["self"], column_groups["covariate"]])
    other_cov_columns = np.concatenate([np.asarray(include_intercept, dtype=int), column_groups["other"], column_groups["covariate"]])
    full_columns = np.arange(design_matrix.shape[1], dtype=int)
    r2_cov = _compute_mass_univariate_r2(design_matrix[:, covariate_columns], y_flat, adjusted=adjusted_r2)
    r2_self_cov = _compute_mass_univariate_r2(design_matrix[:, self_cov_columns], y_flat, adjusted=adjusted_r2)
    r2_other_cov = _compute_mass_univariate_r2(design_matrix[:, other_cov_columns], y_flat, adjusted=adjusted_r2)
    r2_full = _compute_mass_univariate_r2(design_matrix[:, full_columns], y_flat, adjusted=adjusted_r2)
    return r2_self_cov + r2_other_cov - r2_full - r2_cov


def _compute_design_companion(
    *,
    design_matrix: np.ndarray,
    group_values: np.ndarray,
    self_columns: np.ndarray,
    other_columns: np.ndarray,
    covariate_columns: np.ndarray,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    if len(self_columns) != 1 or len(other_columns) != 1:
        raise ValueError("Design-side partial correlation currently requires scalar self and other predictor blocks.")
    observed = _compute_partial_correlation(
        design_matrix=design_matrix,
        group_values=group_values,
        self_column=int(self_columns[0]),
        other_column=int(other_columns[0]),
        covariate_columns=np.asarray(covariate_columns, dtype=int),
    )
    rng = np.random.default_rng(seed)
    unique_groups = list(dict.fromkeys(group_values.astype(str).tolist()))
    bootstrap = np.full(256, np.nan, dtype=float)
    for boot_index in range(256):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        row_indices: list[int] = []
        for group_idx in sampled:
            group_value = unique_groups[int(group_idx)]
            row_indices.extend(np.flatnonzero(group_values.astype(str) == str(group_value)).tolist())
        bootstrap[boot_index] = _compute_partial_correlation(
            design_matrix=design_matrix[np.asarray(row_indices, dtype=int), :],
            group_values=group_values[np.asarray(row_indices, dtype=int)],
            self_column=int(self_columns[0]),
            other_column=int(other_columns[0]),
            covariate_columns=np.asarray(covariate_columns, dtype=int),
        )
    lower = float(np.nanquantile(bootstrap, alpha / 2.0))
    upper = float(np.nanquantile(bootstrap, 1.0 - (alpha / 2.0)))
    return {
        "estimate": float(observed),
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_method": "subject_percentile",
        "n_boot": int(len(bootstrap)),
    }


def _compute_partial_correlation(
    *,
    design_matrix: np.ndarray,
    group_values: np.ndarray,
    self_column: int,
    other_column: int,
    covariate_columns: np.ndarray,
) -> float:
    x = np.asarray(design_matrix[:, self_column], dtype=float)
    y = np.asarray(design_matrix[:, other_column], dtype=float)
    cov = np.asarray(design_matrix[:, covariate_columns], dtype=float) if covariate_columns.size else np.empty((len(x), 0))
    centered_x = x.copy()
    centered_y = y.copy()
    centered_cov = cov.copy()
    for group_value in dict.fromkeys(group_values.astype(str).tolist()):
        group_mask = group_values.astype(str) == str(group_value)
        centered_x[group_mask] -= centered_x[group_mask].mean()
        centered_y[group_mask] -= centered_y[group_mask].mean()
        if centered_cov.size:
            centered_cov[group_mask, :] -= centered_cov[group_mask, :].mean(axis=0, keepdims=True)
    resid_x = _residualise_vector(centered_x, centered_cov)
    resid_y = _residualise_vector(centered_y, centered_cov)
    if np.isclose(np.std(resid_x), 0.0) or np.isclose(np.std(resid_y), 0.0):
        return np.nan
    return float(np.corrcoef(resid_x, resid_y)[0, 1])


def _residualise_vector(vector: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    if covariates.size == 0:
        return vector
    design = np.column_stack([np.ones(len(vector), dtype=float), covariates])
    beta, *_ = np.linalg.lstsq(design, vector, rcond=None)
    return vector - (design @ beta)


def _compute_map_wide_p(*, point_flat: np.ndarray, max_null: np.ndarray) -> np.ndarray:
    p_values = np.full(point_flat.shape, np.nan, dtype=float)
    for feature_index, observed_value in enumerate(point_flat):
        p_values[feature_index] = float(np.mean(max_null >= observed_value))
    return p_values


def _write_commonality_outputs(
    *,
    output_paths: LMEEEGCommonalityOutputPaths,
    observed: CommonalityObservedFit,
    roi_mask: np.ndarray,
    lower_map: np.ndarray,
    upper_map: np.ndarray,
    bootstrap_p_map: np.ndarray,
    calibration_floor_map: np.ndarray,
    map_wide_p_map: np.ndarray,
    bootstrap_roi: np.ndarray,
    settings: LMEEEGCommonalitySettings,
    design_companion: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    distinct_groups: list[int],
    roi_indices: np.ndarray,
) -> None:
    ensure_parent_dir(output_paths.summary_output_path)
    ensure_parent_dir(output_paths.point_estimate_output_path)
    point_map = observed.point_estimate_flat.reshape(len(observed.trial_data.channel_names), len(observed.trial_data.times))
    np.save(output_paths.point_estimate_output_path, point_map)
    np.save(output_paths.roi_mask_output_path, roi_mask.astype(bool))
    np.save(output_paths.bca_lower_output_path, lower_map)
    np.save(output_paths.bca_upper_output_path, upper_map)
    np.save(output_paths.bootstrap_p_output_path, bootstrap_p_map)
    np.save(output_paths.calibration_floor_output_path, calibration_floor_map)
    np.save(output_paths.map_wide_p_output_path, map_wide_p_map)
    np.save(output_paths.bootstrap_roi_output_path, bootstrap_roi)

    roi_table = pd.DataFrame(
        {
            "channel": [observed.trial_data.channel_names[int(index // len(observed.trial_data.times))] for index in roi_indices],
            "time": [float(observed.trial_data.times[int(index % len(observed.trial_data.times))]) for index in roi_indices],
            "channel_index": [int(index // len(observed.trial_data.times)) for index in roi_indices],
            "time_index": [int(index % len(observed.trial_data.times)) for index in roi_indices],
            "commonality": point_map.reshape(-1)[roi_indices],
            "bca_lower": lower_map.reshape(-1)[roi_indices],
            "bca_upper": upper_map.reshape(-1)[roi_indices],
            "bootstrap_p": bootstrap_p_map.reshape(-1)[roi_indices],
            "calibration_floor": calibration_floor_map.reshape(-1)[roi_indices],
            "reliably_positive": lower_map.reshape(-1)[roi_indices] > 0.0,
        }
    )
    roi_table.to_parquet(output_paths.roi_table_output_path, index=False)

    metadata = {
        "seed": int(settings.seed),
        "n_boot": int(settings.n_boot),
        "alpha": float(settings.alpha),
        "config_hash": hashlib.sha256(json.dumps({"settings": settings.__dict__}, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        "python_version": get_python_version(),
        "package_versions": get_package_versions(KEY_PACKAGES),
        "n_input_rows": len(rows),
        "n_trials_used": int(observed.fit_result.n_trials_used),
        "n_subjects": int(len(dict.fromkeys(observed.group_values.astype(str).tolist()))),
        "bootstrap_distinct_groups": distinct_groups,
        "lmm_failed_targets": int(observed.fit_result.n_lmm_failed),
        "lmm_failed_target_indices": [int(index) for index in observed.fit_result.failed_target_indices],
    }
    output_paths.metadata_output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "status": "ok",
        "analysis": "lmeeeg_commonality",
        "config_section": DEFAULT_LMEEEG_COMMONALITY_SECTION,
        "source_section": settings.source_section,
        "source_model": settings.source_model,
        "roi_source": settings.roi_source,
        "n_roi_features": int(np.count_nonzero(roi_mask)),
        "point_estimate": str(output_paths.point_estimate_output_path),
        "roi_mask": str(output_paths.roi_mask_output_path),
        "bca_lower": str(output_paths.bca_lower_output_path),
        "bca_upper": str(output_paths.bca_upper_output_path),
        "bootstrap_p": str(output_paths.bootstrap_p_output_path),
        "calibration_floor": str(output_paths.calibration_floor_output_path),
        "map_wide_p": str(output_paths.map_wide_p_output_path),
        "roi_table": str(output_paths.roi_table_output_path),
        "bootstrap_roi_distribution": str(output_paths.bootstrap_roi_output_path),
        "run_metadata": str(output_paths.metadata_output_path),
        "design_companion": dict(design_companion),
        "summary_path": str(output_paths.summary_output_path),
        "note": (
            "Observed point estimates use adjusted-R2 commonality on backend-marginalized Y*. "
            "Bootstrap resamples refit the random-intercept lmeEEG model at the subject cluster level."
        ),
    }
    output_paths.summary_output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
