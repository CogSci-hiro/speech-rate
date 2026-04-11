"""Build canonical annotation-derived event tables."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from rate.events.features import SUPPORTED_MATCHING_TIME_LOCKS, SUPPORTED_TIME_METHODS, add_event_rate_features
from rate.events.io import EventsTableError, load_ipu_annotations
from rate.preprocessing.events import find_run_anchor_event_from_file
from rate.events.questionnaires import (
    QUESTIONNAIRE_OUTPUT_COLUMNS,
    QuestionnaireError,
    build_subject_questionnaire_table,
    log_questionnaire_coverage,
    summarize_questionnaire_coverage,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_IPU_MIN_DURATION_S = 0.30
DEFAULT_IPU_MIN_SYLLABLES = 3
DEFAULT_ABSDIFF_WINDOW_S = 1.0
DEFAULT_MATCHING_TIME_LOCK = "onset"
DEFAULT_TIME_WITHIN_RUN_NORMALIZE = True
DEFAULT_TIME_WITHIN_RUN_METHOD = "run_duration"

BASE_EVENTS_COLUMNS = [
    "dyad_id",
    "subject_id",
    "run",
    "event_id",
    "t_onset",
    "t_offset",
    "speaker",
    "listener",
    "self_rate",
    "other_rate",
    "latency",
    "proximity",
    "convergence",
    "mean_rate",
    "self_rate_sq",
    "other_rate_sq",
    "proximity_sq",
    "convergence_sq",
    "time_within_run",
    "task",
    "duration_s",
    "syllables",
    "self_n_syllables",
    "other_n_syllables",
    "self_ipu_duration",
    "other_ipu_duration",
    "annotations_path",
]
QUESTIONNAIRE_COLUMNS = [column for column in QUESTIONNAIRE_OUTPUT_COLUMNS if column != "subject_id"]
EVENTS_OUTPUT_COLUMNS = [*BASE_EVENTS_COLUMNS, *QUESTIONNAIRE_COLUMNS]


def _extract_events_config(config: Mapping[str, Any]) -> dict[str, Any]:
    events_cfg = config.get("events", {})
    if not isinstance(events_cfg, Mapping):
        raise EventsTableError("Config key `events` must be an object.")

    time_cfg = events_cfg.get("time_within_run", {})
    if not isinstance(time_cfg, Mapping):
        raise EventsTableError("Config key `events.time_within_run` must be an object.")

    matching_cfg = events_cfg.get("matching", {})
    if not isinstance(matching_cfg, Mapping):
        raise EventsTableError("Config key `events.matching` must be an object.")

    method = str(time_cfg.get("method", DEFAULT_TIME_WITHIN_RUN_METHOD))
    if method not in SUPPORTED_TIME_METHODS:
        raise EventsTableError(
            f"Unsupported events.time_within_run.method={method!r}; expected one of {sorted(SUPPORTED_TIME_METHODS)}."
        )

    return {
        "ipu_min_duration_s": float(events_cfg.get("ipu_min_duration_s", DEFAULT_IPU_MIN_DURATION_S)),
        "ipu_min_syllables": int(events_cfg.get("ipu_min_syllables", DEFAULT_IPU_MIN_SYLLABLES)),
        "normalize_time_within_run": bool(time_cfg.get("normalize", DEFAULT_TIME_WITHIN_RUN_NORMALIZE)),
        "time_within_run_method": method,
        "absdiff_window_s": float(matching_cfg.get("absdiff_window_s", DEFAULT_ABSDIFF_WINDOW_S)),
        "matching_time_lock": str(matching_cfg.get("time_lock", DEFAULT_MATCHING_TIME_LOCK)),
    }


def _validate_matching_config(events_cfg: Mapping[str, Any]) -> None:
    time_lock = str(events_cfg["matching_time_lock"])
    if time_lock not in SUPPORTED_MATCHING_TIME_LOCKS:
        raise EventsTableError(
            "Unsupported events.matching.time_lock="
            f"{time_lock!r}; expected one of {sorted(SUPPORTED_MATCHING_TIME_LOCKS)}."
        )


def _empty_events_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BASE_EVENTS_COLUMNS)


def _resolve_run_anchor_time_s(dataset_row: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    raw_eeg_path_value = dataset_row.get("raw_eeg_path")
    if not raw_eeg_path_value:
        LOGGER.warning(
            "Dataset row is missing raw_eeg_path; leaving annotation times unshifted for subject_id=%s run=%s task=%s",
            dataset_row.get("subject_id"),
            dataset_row.get("run"),
            dataset_row.get("task"),
        )
        return 0.0

    raw_path = Path(str(raw_eeg_path_value))
    if not raw_path.exists():
        raise EventsTableError(f"Raw EEG file does not exist for anchor extraction: {raw_path}")

    try:
        _, anchor_time_s = find_run_anchor_event_from_file(raw_path, config)
    except ValueError as exc:
        raise EventsTableError(
            f"Unable to extract raw anchor event for subject_id={dataset_row.get('subject_id')} "
            f"run={dataset_row.get('run')} task={dataset_row.get('task')}: {exc}"
        ) from exc
    return float(anchor_time_s)


def _build_events_for_unit(dataset_row: Mapping[str, Any], events_cfg: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    annotations_path_value = dataset_row.get("annotations_path")
    if not annotations_path_value:
        return _empty_events_df()

    annotations_path = Path(str(annotations_path_value))
    if not annotations_path.exists():
        LOGGER.warning("Skipping missing annotations file: %s", annotations_path)
        return _empty_events_df()

    unit_df = load_ipu_annotations(annotations_path)
    if unit_df.empty:
        return _empty_events_df()

    unit_df = unit_df.copy()
    anchor_time_s = _resolve_run_anchor_time_s(dataset_row, config)
    unit_df["t_onset"] = unit_df["t_onset"].astype(float) + anchor_time_s
    unit_df["t_offset"] = unit_df["t_offset"].astype(float) + anchor_time_s
    unit_df["duration_s"] = unit_df["t_offset"] - unit_df["t_onset"]
    unit_df = unit_df[unit_df["t_offset"] > unit_df["t_onset"]]
    unit_df = unit_df[unit_df["duration_s"] >= float(events_cfg["ipu_min_duration_s"])]
    unit_df = unit_df[unit_df["syllables"] >= int(events_cfg["ipu_min_syllables"])]
    if unit_df.empty:
        return _empty_events_df()

    unit_df["dyad_id"] = dataset_row.get("dyad_id")
    subject_id = dataset_row.get("subject_id")
    unit_df["subject_id"] = subject_id
    if subject_id is not None:
        unit_df["speaker"] = str(subject_id)
    unit_df["run"] = str(dataset_row.get("run"))
    unit_df["task"] = dataset_row.get("task")
    unit_df["annotations_path"] = str(annotations_path)
    unit_df["event_id"] = ""
    unit_df["listener"] = pd.NA
    unit_df["self_rate"] = pd.NA
    unit_df["other_rate"] = pd.NA
    unit_df["latency"] = pd.NA
    unit_df["proximity"] = pd.NA
    unit_df["convergence"] = pd.NA
    unit_df["mean_rate"] = pd.NA
    unit_df["self_rate_sq"] = pd.NA
    unit_df["other_rate_sq"] = pd.NA
    unit_df["proximity_sq"] = pd.NA
    unit_df["convergence_sq"] = pd.NA
    unit_df["time_within_run"] = pd.NA
    unit_df["self_n_syllables"] = pd.NA
    unit_df["other_n_syllables"] = pd.NA
    unit_df["self_ipu_duration"] = pd.NA
    unit_df["other_ipu_duration"] = pd.NA

    return unit_df[BASE_EVENTS_COLUMNS]


def _add_features_by_group(events_df: pd.DataFrame, events_cfg: Mapping[str, Any]) -> pd.DataFrame:
    grouped_frames: list[pd.DataFrame] = []
    for _, group_df in events_df.groupby(["dyad_id", "run", "task"], dropna=False, sort=False):
        try:
            enriched = add_event_rate_features(
                group_df.reset_index(drop=True),
                absdiff_window_s=float(events_cfg["absdiff_window_s"]),
                normalize=bool(events_cfg["normalize_time_within_run"]),
                method=str(events_cfg["time_within_run_method"]),
                matching_time_lock=str(events_cfg["matching_time_lock"]),
            )
        except ValueError as exc:
            raise EventsTableError(str(exc)) from exc
        grouped_frames.append(enriched)

    if not grouped_frames:
        return _empty_events_df()
    return pd.concat(grouped_frames, ignore_index=True)


def _assign_deterministic_event_ids(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return events_df.copy()

    sorted_df = events_df.sort_values(
        by=["dyad_id", "run", "task", "t_onset", "t_offset", "speaker", "annotations_path"],
        kind="mergesort",
    ).reset_index(drop=True)

    event_ids: list[str] = []
    counters: dict[tuple[str, str, str], int] = {}
    for row in sorted_df.itertuples(index=False):
        key = (str(row.dyad_id), str(row.run), str(row.task))
        counters[key] = counters.get(key, 0) + 1
        event_ids.append(f"{key[0]}_{key[1]}_{key[2]}_ev{counters[key]:05d}")

    sorted_df["event_id"] = event_ids
    return sorted_df


def _merge_questionnaires(events_df: pd.DataFrame, config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        questionnaire_df = build_subject_questionnaire_table(config)
    except QuestionnaireError as exc:
        raise EventsTableError(str(exc)) from exc

    merged = events_df.merge(
        questionnaire_df,
        how="left",
        on="subject_id",
        validate="many_to_one",
    )
    if len(merged) != len(events_df):
        raise EventsTableError("Questionnaire merge changed the number of event rows.")

    coverage_summary = summarize_questionnaire_coverage(
        events_subject_ids=set(events_df["subject_id"].dropna().astype(str)),
        questionnaire_subject_ids=set(questionnaire_df["subject_id"].dropna().astype(str)),
    )
    log_questionnaire_coverage(coverage_summary)
    missing_value_counts = {
        column: int(merged[column].isna().sum())
        for column in QUESTIONNAIRE_COLUMNS
    }
    summary = {
        "event_row_count": int(len(merged)),
        "questionnaire_columns": QUESTIONNAIRE_COLUMNS,
        "missing_value_counts": missing_value_counts,
        **coverage_summary,
    }
    return merged, summary


def build_events_table(config: Mapping[str, Any], dataset_index_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build canonical events table from dataset index rows, then append questionnaires."""
    events_cfg = _extract_events_config(config)
    _validate_matching_config(events_cfg)
    if dataset_index_df.empty:
        return pd.DataFrame(columns=EVENTS_OUTPUT_COLUMNS), {"event_row_count": 0, "questionnaire_columns": QUESTIONNAIRE_COLUMNS}

    rows = dataset_index_df.copy()
    rows = rows[rows["status"] == "ok"].reset_index(drop=True)
    if rows.empty:
        return pd.DataFrame(columns=EVENTS_OUTPUT_COLUMNS), {"event_row_count": 0, "questionnaire_columns": QUESTIONNAIRE_COLUMNS}

    built_frames: list[pd.DataFrame] = []
    for dataset_row in rows.to_dict(orient="records"):
        built_df = _build_events_for_unit(dataset_row, events_cfg, config)
        if not built_df.empty:
            built_frames.append(built_df)

    if not built_frames:
        base_events_df = _empty_events_df()
    else:
        base_events_df = pd.concat(built_frames, ignore_index=True)
        base_events_df = _add_features_by_group(base_events_df, events_cfg)
        base_events_df = _assign_deterministic_event_ids(base_events_df)
        base_events_df = base_events_df[BASE_EVENTS_COLUMNS]

    merged_df, summary = _merge_questionnaires(base_events_df, config)
    return merged_df[EVENTS_OUTPUT_COLUMNS], summary
