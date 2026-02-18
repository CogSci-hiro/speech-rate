"""Dataset index construction and parquet output."""

import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from rate.config import ConfigValidationError, apply_config_defaults
from rate.dataset.discover import (
    ANNOTATION_SUFFIXES,
    AUDIO_SUFFIXES,
    EEG_SUFFIXES,
    discover_dyads,
    find_unit_path,
    normalize_dyad_id,
    normalize_run,
    normalize_task,
    unique_preserve,
)
from rate.dataset.validate import REQUIRED_INDEX_COLUMNS, validate_dataset_index
from rate.paths import ensure_parent_dir
from rate.util import write_parquet_dataframe

LOGGER = logging.getLogger(__name__)

DEFAULT_MISSING_REASON = "listed in dataset.missing_runs"
DEFAULT_EXCLUDED_REASON = "listed in dataset.exclude_runs"
INDEX_RELATIVE_PATH = Path("dataset") / "index.parquet"


class DatasetIndexError(RuntimeError):
    """Raised when dataset index generation fails."""


def _build_override_maps(dataset_cfg: Mapping[str, Any]) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """Build missing/excluded run maps keyed by ``(dyad_id, run)``."""
    missing_map: dict[tuple[str, str], str] = {}
    excluded_map: dict[tuple[str, str], str] = {}

    missing_runs = dataset_cfg.get("missing_runs", []) or []
    exclude_runs = dataset_cfg.get("exclude_runs", []) or []

    if not isinstance(missing_runs, list):
        raise DatasetIndexError("dataset.missing_runs must be a list of mappings.")
    if not isinstance(exclude_runs, list):
        raise DatasetIndexError("dataset.exclude_runs must be a list of mappings.")

    try:
        for item in missing_runs:
            if not isinstance(item, Mapping):
                raise DatasetIndexError("Each dataset.missing_runs entry must be a mapping.")
            key = (normalize_dyad_id(item.get("dyad_id")), normalize_run(item.get("run")))
            missing_map[key] = DEFAULT_MISSING_REASON

        for item in exclude_runs:
            if not isinstance(item, Mapping):
                raise DatasetIndexError("Each dataset.exclude_runs entry must be a mapping.")
            key = (normalize_dyad_id(item.get("dyad_id")), normalize_run(item.get("run")))
            reason = str(item.get("reason") or DEFAULT_EXCLUDED_REASON)
            excluded_map[key] = reason
    except ValueError as exc:
        raise DatasetIndexError(str(exc)) from exc

    return missing_map, excluded_map


def _extract_paths_config(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    """Read required path locations from config."""
    paths_cfg = config.get("paths")
    if not isinstance(paths_cfg, Mapping):
        raise ConfigValidationError("Config missing required object: paths")

    try:
        bids_dir = Path(str(paths_cfg["bids_dir"]))
        eeg_dir = Path(str(paths_cfg["eeg_dir"]))
        annotations_dir = Path(str(paths_cfg["annotations_dir"]))
    except KeyError as exc:
        raise ConfigValidationError(
            "Config paths must include bids_dir, eeg_dir, and annotations_dir for dataset indexing."
        ) from exc

    return bids_dir, eeg_dir, annotations_dir


def _collect_missing_reasons(row: Mapping[str, Any]) -> str:
    """Compose a reason string for missing files in one row."""
    missing_fields = [
        field
        for field in ("eeg_path", "audio_path", "annotations_path")
        if not row.get(field)
    ]
    return f"missing files: {', '.join(missing_fields)}"


def _resolve_dataset_axes(
    dataset_cfg: Mapping[str, Any],
    discovered_dyads: list[str],
    missing_map: Mapping[tuple[str, str], str],
    excluded_map: Mapping[tuple[str, str], str],
) -> tuple[list[str], list[str], list[str]]:
    """Resolve dyad/task/run axes for index expansion."""
    raw_tasks = dataset_cfg.get("tasks", [])
    raw_runs = dataset_cfg.get("runs", [])

    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise DatasetIndexError("dataset.tasks must be a non-empty list.")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise DatasetIndexError("dataset.runs must be a non-empty list.")

    try:
        tasks = unique_preserve(normalize_task(task) for task in raw_tasks)
        runs = unique_preserve(normalize_run(run) for run in raw_runs)
    except ValueError as exc:
        raise DatasetIndexError(str(exc)) from exc

    override_keys = list(missing_map.keys()) + list(excluded_map.keys())
    override_dyads = [key[0] for key in override_keys]
    override_runs = [key[1] for key in override_keys]

    dyads = unique_preserve([*unique_preserve(discovered_dyads), *override_dyads])
    runs = unique_preserve([*runs, *override_runs])
    return dyads, tasks, runs


def build_dataset_index(config: Mapping[str, Any]) -> pd.DataFrame:
    """Build canonical dataset index table."""
    bids_dir, eeg_dir, annotations_dir = _extract_paths_config(config)
    dataset_cfg = config.get("dataset")
    if not isinstance(dataset_cfg, Mapping):
        raise DatasetIndexError("Config missing required object: dataset")

    missing_map, excluded_map = _build_override_maps(dataset_cfg)
    discovered_dyads = discover_dyads(annotations_dir=annotations_dir, bids_dir=bids_dir)
    dyads, tasks, runs = _resolve_dataset_axes(dataset_cfg, discovered_dyads, missing_map, excluded_map)

    if not dyads:
        LOGGER.warning("No dyads discovered and no overrides provided; writing empty dataset index.")
        return pd.DataFrame(columns=REQUIRED_INDEX_COLUMNS)

    rows: list[dict[str, Any]] = []
    for dyad_id in dyads:
        for task in tasks:
            for run in runs:
                row: dict[str, Any] = {
                    "dyad_id": dyad_id,
                    "subject_id": None,
                    "role": None,
                    "task": task,
                    "run": run,
                    "eeg_path": find_unit_path(eeg_dir, dyad_id=dyad_id, task=task, run=run, suffixes=EEG_SUFFIXES),
                    "audio_path": find_unit_path(
                        bids_dir,
                        dyad_id=dyad_id,
                        task=task,
                        run=run,
                        suffixes=AUDIO_SUFFIXES,
                    ),
                    "annotations_path": find_unit_path(
                        annotations_dir,
                        dyad_id=dyad_id,
                        task=task,
                        run=run,
                        suffixes=ANNOTATION_SUFFIXES,
                    ),
                    "status": "ok",
                    "reason": "",
                }

                key = (dyad_id, run)
                if not (row["eeg_path"] and row["audio_path"] and row["annotations_path"]):
                    row["status"] = "missing"
                    row["reason"] = _collect_missing_reasons(row)
                if key in missing_map:
                    row["status"] = "missing"
                    row["reason"] = missing_map[key]
                if key in excluded_map:
                    row["status"] = "excluded"
                    row["reason"] = excluded_map[key]

                rows.append(row)

    index_df = pd.DataFrame(rows, columns=REQUIRED_INDEX_COLUMNS)
    try:
        validate_dataset_index(index_df)
    except ValueError as exc:
        raise DatasetIndexError(str(exc)) from exc

    return index_df


def write_dataset_index(config: Mapping[str, Any], output_path: Path | None = None) -> Path:
    """Write dataset index parquet file."""
    resolved_config = apply_config_defaults(config)
    paths_cfg = resolved_config.get("paths")
    if not isinstance(paths_cfg, Mapping):
        raise ConfigValidationError("Config missing required object: paths")

    out_dir = paths_cfg.get("out_dir")
    if not out_dir:
        raise ConfigValidationError("Config missing required key: paths.out_dir")

    target_path = output_path or Path(str(out_dir)) / INDEX_RELATIVE_PATH
    ensure_parent_dir(target_path)

    index_df = build_dataset_index(resolved_config)
    try:
        write_parquet_dataframe(index_df, target_path)
    except RuntimeError as exc:
        raise DatasetIndexError(str(exc)) from exc
    LOGGER.info("Wrote dataset index to %s (%d rows).", target_path, len(index_df))
    return target_path
