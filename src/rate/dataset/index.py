"""Dataset index construction and parquet output."""

import logging
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from rate.config import ConfigValidationError, apply_config_defaults
from rate.debug import limit_dataframe_by_debug_max_files
from rate.dataset.discover import (
    AUDIO_SUFFIXES,
    EEG_SUFFIXES,
    scan_directory,
    discover_dyads,
    discover_subjects,
    dyad_from_subject_id,
    find_annotation_path,
    find_unit_path,
    normalize_dyad_id,
    normalize_run,
    normalize_subject_id,
    normalize_task,
    unique_preserve,
)
from rate.dataset.validate import REQUIRED_INDEX_COLUMNS, validate_dataset_index
from rate.paths import ensure_parent_dir
from rate.preprocessing.io import resolve_preprocessed_eeg_path
from rate.util import write_parquet_dataframe

LOGGER = logging.getLogger(__name__)

DEFAULT_MISSING_REASON = "listed in dataset.missing_runs"
DEFAULT_EXCLUDED_REASON = "listed in dataset.exclude_runs"
INDEX_RELATIVE_PATH = Path("dataset") / "index.parquet"
DYAD_NUM_PATTERN = re.compile(r"(\d+)")


class DatasetIndexError(RuntimeError):
    """Raised when dataset index generation fails."""


def _build_missing_map(dataset_cfg: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    """Build missing run map keyed by ``(dyad_id, run)``."""
    missing_map: dict[tuple[str, str], str] = {}

    missing_runs = dataset_cfg.get("missing_runs", []) or []
    if not isinstance(missing_runs, list):
        raise DatasetIndexError("dataset.missing_runs must be a list of mappings.")

    try:
        for item in missing_runs:
            if not isinstance(item, Mapping):
                raise DatasetIndexError("Each dataset.missing_runs entry must be a mapping.")
            key = (normalize_dyad_id(item.get("dyad_id")), normalize_run(item.get("run")))
            missing_map[key] = DEFAULT_MISSING_REASON
    except ValueError as exc:
        raise DatasetIndexError(str(exc)) from exc

    return missing_map


def _append_reason(mapping: dict[Any, list[str]], key: Any, reason: str) -> None:
    """Append an auditable reason string to a map entry."""
    if key not in mapping:
        mapping[key] = []
    if reason not in mapping[key]:
        mapping[key].append(reason)


def _format_exclude_reason(block_name: str, entity: str, reason: str) -> str:
    """Create a deterministic audit reason string for exclusion matches."""
    normalized_reason = reason.strip() or DEFAULT_EXCLUDED_REASON
    return f"{block_name}:{entity}:{normalized_reason}"


def _build_exclusion_maps(
    dataset_cfg: Mapping[str, Any],
) -> tuple[dict[str, list[str]], dict[tuple[str, str], list[str]], dict[str, list[str]], dict[tuple[str, str], list[str]]]:
    """Build exclusion maps from `dataset.exclude` and legacy `exclude_runs`."""
    dyad_map: dict[str, list[str]] = {}
    dyad_run_map: dict[tuple[str, str], list[str]] = {}
    subject_map: dict[str, list[str]] = {}
    subject_run_map: dict[tuple[str, str], list[str]] = {}

    exclude_cfg = dataset_cfg.get("exclude", {})
    if exclude_cfg is None:
        exclude_cfg = {}
    if not isinstance(exclude_cfg, Mapping):
        raise DatasetIndexError("dataset.exclude must be a mapping.")

    subjects = exclude_cfg.get("subjects", []) or []
    subject_runs = exclude_cfg.get("subject_runs", []) or []
    dyads = exclude_cfg.get("dyads", []) or []
    dyad_runs = exclude_cfg.get("dyad_runs", []) or []

    if not isinstance(subjects, list):
        raise DatasetIndexError("dataset.exclude.subjects must be a list of mappings.")
    if not isinstance(subject_runs, list):
        raise DatasetIndexError("dataset.exclude.subject_runs must be a list of mappings.")
    if not isinstance(dyads, list):
        raise DatasetIndexError("dataset.exclude.dyads must be a list of mappings.")
    if not isinstance(dyad_runs, list):
        raise DatasetIndexError("dataset.exclude.dyad_runs must be a list of mappings.")

    try:
        for item in subjects:
            if not isinstance(item, Mapping):
                raise DatasetIndexError("Each dataset.exclude.subjects entry must be a mapping.")
            subject_id = normalize_subject_id(item.get("subject_id"))
            reason = _format_exclude_reason("exclude.subjects", subject_id, str(item.get("reason") or DEFAULT_EXCLUDED_REASON))
            _append_reason(subject_map, subject_id, reason)

        for item in subject_runs:
            if not isinstance(item, Mapping):
                raise DatasetIndexError("Each dataset.exclude.subject_runs entry must be a mapping.")
            subject_id = normalize_subject_id(item.get("subject_id"))
            run = normalize_run(item.get("run"))
            reason = _format_exclude_reason(
                "exclude.subject_runs",
                f"{subject_id}/run-{run}",
                str(item.get("reason") or DEFAULT_EXCLUDED_REASON),
            )
            _append_reason(subject_run_map, (subject_id, run), reason)

        for item in dyads:
            if not isinstance(item, Mapping):
                raise DatasetIndexError("Each dataset.exclude.dyads entry must be a mapping.")
            dyad_id = normalize_dyad_id(item.get("dyad_id"))
            reason = _format_exclude_reason("exclude.dyads", dyad_id, str(item.get("reason") or DEFAULT_EXCLUDED_REASON))
            _append_reason(dyad_map, dyad_id, reason)

        for item in dyad_runs:
            if not isinstance(item, Mapping):
                raise DatasetIndexError("Each dataset.exclude.dyad_runs entry must be a mapping.")
            dyad_id = normalize_dyad_id(item.get("dyad_id"))
            run = normalize_run(item.get("run"))
            reason = _format_exclude_reason(
                "exclude.dyad_runs",
                f"{dyad_id}/run-{run}",
                str(item.get("reason") or DEFAULT_EXCLUDED_REASON),
            )
            _append_reason(dyad_run_map, (dyad_id, run), reason)
    except ValueError as exc:
        raise DatasetIndexError(str(exc)) from exc

    # Legacy compatibility for SG1 `exclude_runs`.
    legacy_exclude_runs = dataset_cfg.get("exclude_runs", []) or []
    if not isinstance(legacy_exclude_runs, list):
        raise DatasetIndexError("dataset.exclude_runs must be a list of mappings.")
    try:
        for item in legacy_exclude_runs:
            if not isinstance(item, Mapping):
                raise DatasetIndexError("Each dataset.exclude_runs entry must be a mapping.")
            dyad_id = normalize_dyad_id(item.get("dyad_id"))
            run = normalize_run(item.get("run"))
            reason = _format_exclude_reason(
                "exclude_runs",
                f"{dyad_id}/run-{run}",
                str(item.get("reason") or DEFAULT_EXCLUDED_REASON),
            )
            _append_reason(dyad_run_map, (dyad_id, run), reason)
    except ValueError as exc:
        raise DatasetIndexError(str(exc)) from exc

    return dyad_map, dyad_run_map, subject_map, subject_run_map


def _subject_ids_for_dyad(dyad_id: str) -> list[str]:
    """Derive the two paired subject ids from a normalized dyad id."""
    match = DYAD_NUM_PATTERN.search(dyad_id)
    if not match:
        return []
    dyad_num = int(match.group(1))
    odd_subject = (2 * dyad_num) - 1
    even_subject = 2 * dyad_num
    return [f"sub-{odd_subject:03d}", f"sub-{even_subject:03d}"]


def _subject_roles_for_dyad(dyad_id: str) -> list[tuple[str, str]]:
    """Return (subject_id, role) pairs for a dyad."""
    subject_ids = _subject_ids_for_dyad(dyad_id)
    if not subject_ids:
        return []
    return [(subject_ids[0], "odd"), (subject_ids[1], "even")]


def _resolve_exclusion_reason(
    dyad_id: str,
    run: str,
    subject_id: str | None,
    dyad_map: Mapping[str, list[str]],
    dyad_run_map: Mapping[tuple[str, str], list[str]],
    subject_map: Mapping[str, list[str]],
    subject_run_map: Mapping[tuple[str, str], list[str]],
) -> str | None:
    """Resolve all exclusion reasons that apply to a dyad/run/subject unit."""
    reasons: list[str] = []

    # Most-specific first for deterministic audit output.
    if subject_id is not None:
        reasons.extend(subject_run_map.get((subject_id, run), []))
    reasons.extend(dyad_run_map.get((dyad_id, run), []))
    if subject_id is not None:
        reasons.extend(subject_map.get(subject_id, []))
    reasons.extend(dyad_map.get(dyad_id, []))

    if not reasons:
        return None
    return " | ".join(unique_preserve(reasons))


def _extract_paths_config(config: Mapping[str, Any]) -> tuple[Path, Path]:
    """Read required path locations from config."""
    paths_cfg = config.get("paths")
    if not isinstance(paths_cfg, Mapping):
        raise ConfigValidationError("Config missing required object: paths")

    try:
        bids_dir = Path(str(paths_cfg["bids_dir"]))
        annotations_dir = Path(str(paths_cfg["annotations_dir"]))
    except KeyError as exc:
        raise ConfigValidationError(
            "Config paths must include bids_dir and annotations_dir for dataset indexing."
        ) from exc

    return bids_dir, annotations_dir


def _collect_missing_reasons(row: Mapping[str, Any]) -> str:
    """Compose a reason string for missing files in one row."""
    missing_fields = [
        field
        for field in ("raw_eeg_path", "audio_path", "annotations_path")
        if not row.get(field)
    ]
    return f"missing files: {', '.join(missing_fields)}"


def _resolve_dataset_axes(
    dataset_cfg: Mapping[str, Any],
    discovered_dyads: list[str],
    missing_map: Mapping[tuple[str, str], str],
    dyad_run_map: Mapping[tuple[str, str], list[str]],
    dyad_map: Mapping[str, list[str]],
    subject_map: Mapping[str, list[str]],
    subject_run_map: Mapping[tuple[str, str], list[str]],
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

    override_keys = list(missing_map.keys()) + list(dyad_run_map.keys())
    override_dyads = [key[0] for key in override_keys]
    override_runs = [key[1] for key in override_keys]
    override_dyads.extend(dyad_map.keys())
    override_dyads.extend(dyad_from_subject_id(subject_id) for subject_id in subject_map.keys())
    for subject_id, run in subject_run_map.keys():
        override_dyads.append(dyad_from_subject_id(subject_id))
        override_runs.append(run)

    dyads = unique_preserve([*unique_preserve(discovered_dyads), *override_dyads])
    runs = unique_preserve([*runs, *override_runs])
    return dyads, tasks, runs


def build_dataset_index(config: Mapping[str, Any]) -> pd.DataFrame:
    """Build canonical dataset index table."""
    bids_dir, annotations_dir = _extract_paths_config(config)
    bids_snapshot = scan_directory(bids_dir)
    annotations_snapshot = scan_directory(annotations_dir)
    dataset_cfg = config.get("dataset")
    if not isinstance(dataset_cfg, Mapping):
        raise DatasetIndexError("Config missing required object: dataset")

    missing_map = _build_missing_map(dataset_cfg)
    dyad_map, dyad_run_map, subject_map, subject_run_map = _build_exclusion_maps(dataset_cfg)
    discovered_subjects = discover_subjects(annotations_dir=annotations_dir, bids_dir=bids_dir)
    if subject_map:
        discovered_subjects = [
            subject_id for subject_id in discovered_subjects if subject_id not in subject_map
        ]
    discovered_dyads = [dyad_from_subject_id(subject_id) for subject_id in discovered_subjects]
    if not discovered_dyads:
        discovered_dyads = discover_dyads(annotations_dir=annotations_dir, bids_dir=bids_dir)
    dyads, tasks, runs = _resolve_dataset_axes(
        dataset_cfg=dataset_cfg,
        discovered_dyads=discovered_dyads,
        missing_map=missing_map,
        dyad_run_map=dyad_run_map,
        dyad_map=dyad_map,
        subject_map=subject_map,
        subject_run_map=subject_run_map,
    )

    if not dyads:
        LOGGER.warning("No dyads discovered and no overrides provided; writing empty dataset index.")
        return pd.DataFrame(columns=REQUIRED_INDEX_COLUMNS)

    rows: list[dict[str, Any]] = []
    for dyad_id in dyads:
        subject_roles = _subject_roles_for_dyad(dyad_id) or [(None, None)]
        for task in tasks:
            for run in runs:
                for subject_id, role in subject_roles:
                    raw_eeg_path = find_unit_path(
                        bids_snapshot,
                        dyad_id=dyad_id,
                        task=task,
                        run=run,
                        suffixes=EEG_SUFFIXES,
                        subject_id=subject_id,
                    )
                    row: dict[str, Any] = {
                        "dyad_id": dyad_id,
                        "subject_id": subject_id,
                        "role": role,
                        "task": task,
                        "run": run,
                        "raw_eeg_path": raw_eeg_path,
                        "eeg_path": "",
                        "audio_path": find_unit_path(
                            bids_snapshot,
                            dyad_id=dyad_id,
                            task=task,
                            run=run,
                            suffixes=AUDIO_SUFFIXES,
                            subject_id=subject_id,
                        ),
                        "annotations_path": find_annotation_path(
                            annotations_snapshot,
                            dyad_id=dyad_id,
                            task=task,
                            run=run,
                            subject_id=subject_id,
                        ),
                        "status": "ok",
                        "reason": "",
                    }
                    row["eeg_path"] = str(resolve_preprocessed_eeg_path(config, row))

                    key = (dyad_id, run)
                    if not (row["raw_eeg_path"] and row["audio_path"] and row["annotations_path"]):
                        row["status"] = "missing"
                        row["reason"] = _collect_missing_reasons(row)
                    if key in missing_map:
                        row["status"] = "missing"
                        row["reason"] = missing_map[key]
                    exclusion_reason = _resolve_exclusion_reason(
                        dyad_id=dyad_id,
                        run=run,
                        subject_id=subject_id,
                        dyad_map=dyad_map,
                        dyad_run_map=dyad_run_map,
                        subject_map=subject_map,
                        subject_run_map=subject_run_map,
                    )
                    if exclusion_reason:
                        row["status"] = "excluded"
                        row["reason"] = exclusion_reason

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
    index_df = limit_dataframe_by_debug_max_files(
        index_df,
        resolved_config,
        logger=LOGGER,
        label="dataset index",
    )
    try:
        write_parquet_dataframe(index_df, target_path)
    except RuntimeError as exc:
        raise DatasetIndexError(str(exc)) from exc
    LOGGER.info("Wrote dataset index to %s (%d rows).", target_path, len(index_df))
    return target_path
