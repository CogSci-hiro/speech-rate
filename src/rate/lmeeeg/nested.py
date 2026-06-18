"""Configuration and path helpers for nested-block lmeEEG inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rate.paths import resolve_out_dir

DEFAULT_LMEEEG_NESTED_SECTION = "lmeeeg_nested"


@dataclass(frozen=True)
class LMEEEGNestedAnalysisSpec:
    """One nested-block comparison against a full pooled lmeEEG model."""

    name: str
    reduced_formula: str | list[str]
    full_model: str
    tested_predictor: str | None = None


@dataclass(frozen=True)
class LMEEEGNestedStatsSettings:
    """Resolved settings for nested-block pooled lmeEEG inference."""

    source_section: str
    output_dirname: str
    correction: str
    n_permutations: int
    seed: int
    tail: int
    threshold: float | dict[str, float] | None
    adjacency: str | None
    tfce_h_power: float | None
    tfce_e_power: float | None
    analyses: tuple[LMEEEGNestedAnalysisSpec, ...]


@dataclass(frozen=True)
class LMEEEGNestedGroupPaths:
    """Resolved root and summary outputs for nested-block lmeEEG inference."""

    root_dir: Path
    summary_output_path: Path


def build_lmeeeg_nested_group_output_paths(
    config: Mapping[str, Any],
    *,
    config_section: str = DEFAULT_LMEEEG_NESTED_SECTION,
) -> LMEEEGNestedGroupPaths:
    """Build canonical output paths for nested-block lmeEEG inference."""
    settings = resolve_lmeeeg_nested_settings(config, config_section=config_section)
    out_dir = resolve_out_dir(config)
    return LMEEEGNestedGroupPaths(
        root_dir=out_dir / "stats" / settings.output_dirname,
        summary_output_path=out_dir / "stats" / f"{config_section}_group_stats.json",
    )


def resolve_lmeeeg_nested_settings(
    config: Mapping[str, Any],
    *,
    config_section: str = DEFAULT_LMEEEG_NESTED_SECTION,
) -> LMEEEGNestedStatsSettings:
    """Resolve nested-block lmeEEG settings from config."""
    section = dict(config.get(config_section) or {})
    source_section = str(section.get("source_section", "lmeeeg"))
    output_dirname = str(section.get("output_dirname", config_section))
    analyses = _resolve_nested_analyses(section, source_section=source_section)

    stats = dict(section.get("stats") or {})
    correction = str(stats.get("correction", "cluster")).lower()
    if correction not in {"cluster", "tfce"}:
        raise ValueError(f"Unsupported nested lmeEEG correction `{correction}`.")

    threshold = _resolve_nested_threshold(stats, correction=correction)
    adjacency = stats.get("adjacency")
    adjacency_name = None if adjacency in (None, "none", "None") else str(adjacency)

    return LMEEEGNestedStatsSettings(
        source_section=source_section,
        output_dirname=output_dirname,
        correction=correction,
        n_permutations=int(stats.get("n_permutations", 1024)),
        seed=int(stats.get("seed", 0)),
        tail=int(stats.get("tail", 1)),
        threshold=threshold,
        adjacency=adjacency_name,
        tfce_h_power=_optional_float(stats.get("tfce_h_power")),
        tfce_e_power=_optional_float(stats.get("tfce_e_power")),
        analyses=analyses,
    )


def _resolve_nested_analyses(
    section: Mapping[str, Any],
    *,
    source_section: str,
) -> tuple[LMEEEGNestedAnalysisSpec, ...]:
    analyses_cfg = section.get("analyses")
    if not isinstance(analyses_cfg, Mapping) or not analyses_cfg:
        raise ValueError("`lmeeeg_nested.analyses` must be a non-empty mapping.")

    default_full_model = section.get("full_model")
    if not isinstance(default_full_model, str) or not default_full_model.strip():
        raise ValueError("`lmeeeg_nested.full_model` must be configured as the pooled full-model name.")

    analyses: list[LMEEEGNestedAnalysisSpec] = []
    for analysis_name, analysis_cfg in analyses_cfg.items():
        if not isinstance(analysis_cfg, Mapping):
            raise ValueError(f"`lmeeeg_nested.analyses.{analysis_name}` must be a mapping.")
        reduced_formula = analysis_cfg.get("reduced_formula")
        if not isinstance(reduced_formula, (str, list)):
            raise ValueError(
                f"`lmeeeg_nested.analyses.{analysis_name}.reduced_formula` must be a string or list of strings."
            )
        if isinstance(reduced_formula, list) and not all(isinstance(value, str) and value.strip() for value in reduced_formula):
            raise ValueError(
                f"`lmeeeg_nested.analyses.{analysis_name}.reduced_formula` list values must be non-empty strings."
            )
        full_model = str(analysis_cfg.get("full_model", default_full_model))
        tested_predictor = analysis_cfg.get("tested_predictor")
        analyses.append(
            LMEEEGNestedAnalysisSpec(
                name=str(analysis_name),
                reduced_formula=reduced_formula if isinstance(reduced_formula, list) else str(reduced_formula),
                full_model=full_model,
                tested_predictor=None if tested_predictor is None else str(tested_predictor),
            )
        )

    return tuple(analyses)


def _resolve_nested_threshold(stats: Mapping[str, Any], *, correction: str) -> float | dict[str, float] | None:
    if correction == "cluster":
        threshold = stats.get("threshold", stats.get("cluster_threshold"))
        if threshold is None:
            raise ValueError(
                "Nested-block cluster inference requires an explicit numeric `lmeeeg_nested.stats.threshold` "
                "(or `cluster_threshold`) because F-map alpha-to-threshold conversion is not inferred here."
            )
        return float(threshold)
    threshold = stats.get("tfce_threshold")
    if threshold is None:
        return None
    if not isinstance(threshold, Mapping):
        raise ValueError("`lmeeeg_nested.stats.tfce_threshold` must be a mapping like {start: ..., step: ...}.")
    return {str(key): float(value) for key, value in threshold.items()}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
