"""Variance-partition helpers for pooled lmeEEG analyses."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from rate.lmeeeg.pooling import load_pooled_lmeeeg_trial_data
from rate.paths import ensure_parent_dir, resolve_out_dir
from rate.util import read_parquet_dataframe, write_parquet_dataframe

COMPONENT_NAMES = ("total", "unique_self", "unique_other", "shared")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariancePartitionSettings:
    """Resolved config for the explicit variance-partition analysis."""

    source_section: str
    source_model: str
    self_predictor: str
    other_predictor: str
    output_dirname: str


@dataclass(frozen=True)
class VariancePartitionFitPaths:
    """Canonical pooled-fit outputs for the variance-partition analysis."""

    root_dir: Path
    table_output_path: Path
    channel_names_output_path: Path
    times_output_path: Path
    summary_output_path: Path


@dataclass(frozen=True)
class VariancePartitionGroupPaths:
    """Canonical group-summary outputs for the variance-partition analysis."""

    root_dir: Path
    group_dir: Path
    table_output_path: Path
    channel_names_output_path: Path
    times_output_path: Path
    total_map_output_path: Path
    unique_self_map_output_path: Path
    unique_other_map_output_path: Path
    shared_map_output_path: Path
    summary_output_path: Path


@dataclass(frozen=True)
class FeatureModelFit:
    """Per-feature fit summary for one fixed-effects specification."""

    marginal_r2: float
    converged: bool
    singular: bool
    boundary_warning: bool
    random_variance: float
    residual_variance: float
    message: str


def resolve_variance_partition_settings(config: Mapping[str, Any]) -> VariancePartitionSettings:
    """Resolve the dedicated variance-partition config with small defaults."""
    section = dict(config.get("lmeeeg_variance_partition") or {})
    predictors = dict(section.get("predictors") or {})
    return VariancePartitionSettings(
        source_section=str(section.get("source_section", "lmeeeg")),
        source_model=str(section.get("source_model", "both_effect")),
        self_predictor=str(predictors.get("self", "self_rate")),
        other_predictor=str(predictors.get("other", "other_rate")),
        output_dirname=str(section.get("output_dirname", "lmeeeg_variance_partition")),
    )


def build_variance_partition_fit_output_paths(config: Mapping[str, Any]) -> VariancePartitionFitPaths:
    """Build canonical pooled-fit output paths."""
    settings = resolve_variance_partition_settings(config)
    root_dir = resolve_out_dir(config) / settings.output_dirname
    return VariancePartitionFitPaths(
        root_dir=root_dir,
        table_output_path=root_dir / "variance_partition.parquet",
        channel_names_output_path=root_dir / "channel_names.json",
        times_output_path=root_dir / "times.npy",
        summary_output_path=root_dir / "summary.json",
    )


def build_variance_partition_group_output_paths(config: Mapping[str, Any]) -> VariancePartitionGroupPaths:
    """Build canonical descriptive group-summary output paths."""
    settings = resolve_variance_partition_settings(config)
    root_dir = resolve_out_dir(config) / settings.output_dirname
    group_dir = root_dir / "group"
    return VariancePartitionGroupPaths(
        root_dir=root_dir,
        group_dir=group_dir,
        table_output_path=group_dir / "descriptive_maps.parquet",
        channel_names_output_path=group_dir / "channel_names.json",
        times_output_path=group_dir / "times.npy",
        total_map_output_path=group_dir / "total.npy",
        unique_self_map_output_path=group_dir / "unique_self.npy",
        unique_other_map_output_path=group_dir / "unique_other.npy",
        shared_map_output_path=group_dir / "shared.npy",
        summary_output_path=group_dir / "summary.json",
    )


def compute_variance_partition_components(
    *,
    r2_null: float,
    r2_self: float,
    r2_other: float,
    r2_full: float,
) -> dict[str, float]:
    """Compute the four-way descriptive variance partition."""
    return {
        "total": float(r2_full - r2_null),
        "unique_self": float(r2_full - r2_other),
        "unique_other": float(r2_full - r2_self),
        "shared": float(r2_self + r2_other - r2_full - r2_null),
    }


def fit_variance_partition_table(
    config: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    *,
    trial_regressors_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the pooled variance-partition analysis across all selected rows."""
    settings = resolve_variance_partition_settings(config)
    trial_data = load_pooled_lmeeeg_trial_data(
        config,
        rows,
        config_section=settings.source_section,
        model_name=settings.source_model,
        trial_regressors_path=trial_regressors_path,
    )

    control_terms = _resolve_control_terms(
        trial_data.design_result.term_name_slices,
        self_predictor=settings.self_predictor,
        other_predictor=settings.other_predictor,
    )
    group_column = str(trial_data.group_column)
    eeg_data = np.asarray(trial_data.eeg_data, dtype=float)
    metadata_df = trial_data.trial_metadata.reset_index(drop=True).copy()
    n_obs = int(eeg_data.shape[0])
    n_subjects = int(metadata_df[group_column].nunique(dropna=True))

    formulas = {
        "null": _make_fixed_formula(control_terms),
        "self": _make_fixed_formula([settings.self_predictor, *control_terms]),
        "other": _make_fixed_formula([settings.other_predictor, *control_terms]),
        "full": _make_fixed_formula([settings.self_predictor, settings.other_predictor, *control_terms]),
    }
    total_features = int(len(trial_data.channel_names) * len(trial_data.times))
    progress_interval = max(1, len(trial_data.channel_names) // 8)
    LOGGER.info(
        "Starting lmeEEG variance partition for %d channel x time features (%d channels x %d times)",
        total_features,
        len(trial_data.channel_names),
        len(trial_data.times),
    )
    LOGGER.info(
        "Variance partition formulas: null=`%s`, self=`%s`, other=`%s`, full=`%s`",
        formulas["null"],
        formulas["self"],
        formulas["other"],
        formulas["full"],
    )

    results: list[dict[str, Any]] = []
    for channel_index, channel_name in enumerate(trial_data.channel_names):
        if channel_index == 0 or channel_index == len(trial_data.channel_names) - 1 or ((channel_index + 1) % progress_interval == 0):
            completed_features = int(channel_index * len(trial_data.times))
            LOGGER.info(
                "Variance partition progress: channel %d/%d (%s), completed %d/%d features",
                channel_index + 1,
                len(trial_data.channel_names),
                channel_name,
                completed_features,
                total_features,
            )
        for time_index, time_value in enumerate(trial_data.times):
            feature_df = metadata_df.copy(deep=False)
            feature_df["y"] = eeg_data[:, channel_index, time_index]

            null_fit = _fit_feature_model(feature_df, formula=formulas["null"], group_column=group_column)
            self_fit = _fit_feature_model(feature_df, formula=formulas["self"], group_column=group_column)
            other_fit = _fit_feature_model(feature_df, formula=formulas["other"], group_column=group_column)
            full_fit = _fit_feature_model(feature_df, formula=formulas["full"], group_column=group_column)

            row = {
                "channel": str(channel_name),
                "time": float(time_value),
                "channel_index": int(channel_index),
                "time_index": int(time_index),
                "r2_null": null_fit.marginal_r2,
                "r2_self": self_fit.marginal_r2,
                "r2_other": other_fit.marginal_r2,
                "r2_full": full_fit.marginal_r2,
                "n_obs": n_obs,
                "n_subjects": n_subjects,
                "converged_null": bool(null_fit.converged),
                "converged_self": bool(self_fit.converged),
                "converged_other": bool(other_fit.converged),
                "converged_full": bool(full_fit.converged),
                "singular_null": bool(null_fit.singular),
                "singular_self": bool(self_fit.singular),
                "singular_other": bool(other_fit.singular),
                "singular_full": bool(full_fit.singular),
                "boundary_warning_null": bool(null_fit.boundary_warning),
                "boundary_warning_self": bool(self_fit.boundary_warning),
                "boundary_warning_other": bool(other_fit.boundary_warning),
                "boundary_warning_full": bool(full_fit.boundary_warning),
                "message_null": null_fit.message,
                "message_self": self_fit.message,
                "message_other": other_fit.message,
                "message_full": full_fit.message,
            }
            if np.all(np.isfinite([row["r2_null"], row["r2_self"], row["r2_other"], row["r2_full"]])):
                row.update(
                    compute_variance_partition_components(
                        r2_null=float(row["r2_null"]),
                        r2_self=float(row["r2_self"]),
                        r2_other=float(row["r2_other"]),
                        r2_full=float(row["r2_full"]),
                    )
                )
            else:
                row.update({component_name: np.nan for component_name in COMPONENT_NAMES})
            results.append(row)

    result_df = pd.DataFrame(results)
    valid_features = int(result_df["r2_full"].notna().sum()) if "r2_full" in result_df.columns else 0
    LOGGER.info(
        "Finished lmeEEG variance partition: wrote %d rows with valid full-model R2 for %d/%d features",
        len(result_df),
        valid_features,
        total_features,
    )
    summary = {
        "status": "ok",
        "analysis": "lmeeeg_variance_partition",
        "fit_scope": "pooled",
        "source_section": settings.source_section,
        "source_model": settings.source_model,
        "self_predictor": settings.self_predictor,
        "other_predictor": settings.other_predictor,
        "group_column": group_column,
        "control_terms": control_terms,
        "fixed_formulas": formulas,
        "n_input_rows": len(rows),
        "n_trials_used": n_obs,
        "n_subjects": n_subjects,
        "n_channels": len(trial_data.channel_names),
        "n_times": int(len(trial_data.times)),
        "channel_names": list(trial_data.channel_names),
        "times_path": None,
        "table_path": None,
        "note": (
            "Variance partitioning is descriptive and is fit once across the pooled lmeEEG-ready table. "
            "This preserves the existing random-intercept-by-subject structure used by the current lmeEEG pipeline."
        ),
    }
    return result_df, summary | {"times": np.asarray(trial_data.times, dtype=float)}


def save_variance_partition_fit(
    *,
    config: Mapping[str, Any],
    table: pd.DataFrame,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Write pooled variance-partition outputs to disk."""
    paths = build_variance_partition_fit_output_paths(config)
    ensure_parent_dir(paths.summary_output_path)
    write_parquet_dataframe(table.reset_index(drop=True), paths.table_output_path)

    channel_names = [str(value) for value in summary["channel_names"]]
    times = np.asarray(summary["times"], dtype=float)
    paths.channel_names_output_path.write_text(json.dumps(channel_names, indent=2) + "\n", encoding="utf-8")
    np.save(paths.times_output_path, times)

    summary_payload = dict(summary)
    summary_payload["table_path"] = str(paths.table_output_path)
    summary_payload["times_path"] = str(paths.times_output_path)
    summary_payload["channel_names_path"] = str(paths.channel_names_output_path)
    summary_payload["summary_path"] = str(paths.summary_output_path)
    summary_payload.pop("times", None)
    paths.summary_output_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_payload


def build_group_descriptive_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read pooled variance-partition outputs and write descriptive component maps."""
    fit_paths = build_variance_partition_fit_output_paths(config)
    group_paths = build_variance_partition_group_output_paths(config)
    if not fit_paths.table_output_path.exists():
        raise FileNotFoundError(f"Variance-partition fit table not found: {fit_paths.table_output_path}")
    if not fit_paths.summary_output_path.exists():
        raise FileNotFoundError(f"Variance-partition fit summary not found: {fit_paths.summary_output_path}")

    fit_summary = json.loads(fit_paths.summary_output_path.read_text(encoding="utf-8"))
    table = read_parquet_dataframe(fit_paths.table_output_path).copy()
    channel_names = json.loads(fit_paths.channel_names_output_path.read_text(encoding="utf-8"))
    times = np.load(fit_paths.times_output_path)

    descriptive_table = table.loc[
        :,
        [
            "channel",
            "time",
            "channel_index",
            "time_index",
            "r2_null",
            "r2_self",
            "r2_other",
            "r2_full",
            *COMPONENT_NAMES,
            "n_obs",
            "n_subjects",
            "converged_null",
            "converged_self",
            "converged_other",
            "converged_full",
            "singular_null",
            "singular_self",
            "singular_other",
            "singular_full",
        ],
    ].copy()

    ensure_parent_dir(group_paths.summary_output_path)
    write_parquet_dataframe(descriptive_table.reset_index(drop=True), group_paths.table_output_path)
    group_paths.channel_names_output_path.write_text(json.dumps(channel_names, indent=2) + "\n", encoding="utf-8")
    np.save(group_paths.times_output_path, np.asarray(times, dtype=float))

    component_map_paths = {
        "total": group_paths.total_map_output_path,
        "unique_self": group_paths.unique_self_map_output_path,
        "unique_other": group_paths.unique_other_map_output_path,
        "shared": group_paths.shared_map_output_path,
    }
    for component_name, output_path in component_map_paths.items():
        np.save(output_path, build_component_map(descriptive_table, channel_names=channel_names, times=times, value_column=component_name))

    summary = {
        "status": "ok",
        "analysis": "lmeeeg_variance_partition_group_stats",
        "descriptive_only": True,
        "fit_scope": fit_summary.get("fit_scope", "pooled"),
        "source_section": fit_summary.get("source_section"),
        "source_model": fit_summary.get("source_model"),
        "self_predictor": fit_summary.get("self_predictor"),
        "other_predictor": fit_summary.get("other_predictor"),
        "n_rows": int(len(descriptive_table)),
        "n_channels": len(channel_names),
        "n_times": int(len(times)),
        "table_path": str(group_paths.table_output_path),
        "channel_names_path": str(group_paths.channel_names_output_path),
        "times_path": str(group_paths.times_output_path),
        "component_maps": {name: str(path) for name, path in component_map_paths.items()},
        "note": (
            "These maps are descriptive summaries of the pooled mixed-model marginal R^2 decomposition. "
            "They are not inferential statistics."
        ),
    }
    group_paths.summary_output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_component_map(
    table: pd.DataFrame,
    *,
    channel_names: list[str],
    times: np.ndarray,
    value_column: str,
) -> np.ndarray:
    """Convert a tidy variance-partition table into a channel x time map."""
    component_map = np.full((len(channel_names), int(len(times))), np.nan, dtype=float)
    for row in table.itertuples(index=False):
        component_map[int(row.channel_index), int(row.time_index)] = float(getattr(row, value_column))
    return component_map


def _resolve_control_terms(
    term_name_slices: Mapping[str, slice],
    *,
    self_predictor: str,
    other_predictor: str,
) -> list[str]:
    term_names = [str(term_name) for term_name in term_name_slices if str(term_name) != "Intercept"]
    if self_predictor not in term_names or other_predictor not in term_names:
        raise ValueError(
            "Variance-partition source model must include both predictors "
            f"`{self_predictor}` and `{other_predictor}`."
        )
    return [term_name for term_name in term_names if term_name not in {self_predictor, other_predictor}]


def _make_fixed_formula(terms: list[str]) -> str:
    if not terms:
        return "y ~ 1"
    return "y ~ " + " + ".join(str(term_name) for term_name in terms)


def _fit_feature_model(feature_df: pd.DataFrame, *, formula: str, group_column: str) -> FeatureModelFit:
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            mixed_model = smf.mixedlm(formula, data=feature_df, groups=feature_df[group_column], re_formula="1")
            mixed_result = mixed_model.fit(reml=True, method="lbfgs", disp=False)
        warning_messages = [str(item.message) for item in caught_warnings]
        warning_text = " | ".join(message for message in warning_messages if message)
        random_variance = _extract_random_variance(mixed_result)
        residual_variance = float(mixed_result.scale)
        fitted_fixed = np.asarray(mixed_result.model.exog @ np.asarray(mixed_result.fe_params, dtype=float), dtype=float)
        fixed_variance = float(np.var(fitted_fixed, ddof=0))
        denominator = fixed_variance + random_variance + residual_variance
        marginal_r2 = float(fixed_variance / denominator) if denominator > 0.0 else np.nan
        return FeatureModelFit(
            marginal_r2=marginal_r2,
            converged=bool(getattr(mixed_result, "converged", False)),
            singular=bool(random_variance <= 1e-8),
            boundary_warning=any("boundary" in message.lower() for message in warning_messages),
            random_variance=random_variance,
            residual_variance=residual_variance,
            message=warning_text,
        )
    except Exception as exc:
        return FeatureModelFit(
            marginal_r2=np.nan,
            converged=False,
            singular=False,
            boundary_warning=False,
            random_variance=np.nan,
            residual_variance=np.nan,
            message=str(exc),
        )


def _extract_random_variance(mixed_result) -> float:
    cov_re = getattr(mixed_result, "cov_re", None)
    if cov_re is None:
        return 0.0
    if hasattr(cov_re, "to_numpy"):
        values = cov_re.to_numpy(dtype=float)
    else:
        values = np.asarray(cov_re, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.squeeze(values))
