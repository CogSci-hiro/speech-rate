"""Build canonical event tables from dataset index rows."""

import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from rate.events.features import SUPPORTED_MATCHING_TIME_LOCKS, SUPPORTED_TIME_METHODS, add_event_rate_features
from rate.events.io import EventsTableError, load_ipu_annotations

LOGGER = logging.getLogger(__name__)

DEFAULT_IPU_MIN_DURATION_S = 0.30
DEFAULT_IPU_MIN_SYLLABLES = 3
DEFAULT_ABSDIFF_WINDOW_S = 1.0
DEFAULT_MATCHING_TIME_LOCK = "onset"
DEFAULT_TIME_WITHIN_RUN_NORMALIZE = True
DEFAULT_TIME_WITHIN_RUN_METHOD = "run_duration"

INDEX_RELATIVE_PATH = Path("dataset") / "index.parquet"

REQUIRED_EVENTS_COLUMNS = [
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
    "alignment",
    "convergence",
    "mean_rate",
    "self_rate_sq",
    "other_rate_sq",
    "alignment_sq",
    "convergence_sq",
    "time_within_run",
    "task",
    "duration_s",
    "syllables",
    "annotations_path",
]


def _extract_events_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract events configuration with defaults and validation."""
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
    """Validate matching sub-configuration values."""
    time_lock = str(events_cfg["matching_time_lock"])
    if time_lock not in SUPPORTED_MATCHING_TIME_LOCKS:
        raise EventsTableError(
            "Unsupported events.matching.time_lock="
            f"{time_lock!r}; expected one of {sorted(SUPPORTED_MATCHING_TIME_LOCKS)}."
        )


def _build_events_for_unit(dataset_row: Mapping[str, Any], events_cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Build filtered events for one dyad/run/task dataset row."""
    annotations_path_value = dataset_row.get("annotations_path")
    if not annotations_path_value:
        return pd.DataFrame(columns=REQUIRED_EVENTS_COLUMNS)

    annotations_path = Path(str(annotations_path_value))
    if not annotations_path.exists():
        LOGGER.warning("Skipping missing annotations file: %s", annotations_path)
        return pd.DataFrame(columns=REQUIRED_EVENTS_COLUMNS)

    unit_df = load_ipu_annotations(annotations_path)
    if unit_df.empty:
        return pd.DataFrame(columns=REQUIRED_EVENTS_COLUMNS)

    unit_df = unit_df.copy()
    unit_df["duration_s"] = unit_df["t_offset"] - unit_df["t_onset"]
    unit_df = unit_df[unit_df["t_offset"] > unit_df["t_onset"]]

    unit_df = unit_df[unit_df["duration_s"] >= float(events_cfg["ipu_min_duration_s"])]
    unit_df = unit_df[unit_df["syllables"] >= int(events_cfg["ipu_min_syllables"])]
    if unit_df.empty:
        return pd.DataFrame(columns=REQUIRED_EVENTS_COLUMNS)

    unit_df["dyad_id"] = dataset_row.get("dyad_id")
    subject_id = dataset_row.get("subject_id")
    unit_df["subject_id"] = subject_id
    # Subject-scoped annotation files often have a placeholder speaker label.
    # Replace with subject_id so cross-subject pairing can be computed later.
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
    unit_df["alignment"] = pd.NA
    unit_df["convergence"] = pd.NA
    unit_df["mean_rate"] = pd.NA
    unit_df["self_rate_sq"] = pd.NA
    unit_df["other_rate_sq"] = pd.NA
    unit_df["alignment_sq"] = pd.NA
    unit_df["convergence_sq"] = pd.NA
    unit_df["time_within_run"] = pd.NA

    return unit_df[REQUIRED_EVENTS_COLUMNS]


def _add_features_by_group(events_df: pd.DataFrame, events_cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Compute behavioural/time features jointly per dyad/run/task group."""
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
        return pd.DataFrame(columns=REQUIRED_EVENTS_COLUMNS)
    return pd.concat(grouped_frames, ignore_index=True)


def _assign_deterministic_event_ids(events_df: pd.DataFrame) -> pd.DataFrame:
    """Assign stable event ids grouped by dyad/run/task in sorted order."""
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


def build_events_table(config: Mapping[str, Any], dataset_index_df: pd.DataFrame) -> pd.DataFrame:
    """Build canonical events table from dataset index rows."""
    events_cfg = _extract_events_config(config)
    _validate_matching_config(events_cfg)
    if dataset_index_df.empty:
        return pd.DataFrame(columns=REQUIRED_EVENTS_COLUMNS)

    rows = dataset_index_df.copy()
    rows = rows[rows["status"] == "ok"].reset_index(drop=True)
    if rows.empty:
        return pd.DataFrame(columns=REQUIRED_EVENTS_COLUMNS)

    built_frames: list[pd.DataFrame] = []
    for dataset_row in rows.to_dict(orient="records"):
        built_df = _build_events_for_unit(dataset_row, events_cfg)
        if not built_df.empty:
            built_frames.append(built_df)

    if not built_frames:
        return pd.DataFrame(columns=REQUIRED_EVENTS_COLUMNS)

    events_df = pd.concat(built_frames, ignore_index=True)
    events_df = _add_features_by_group(events_df, events_cfg)
    events_df = _assign_deterministic_event_ids(events_df)
    return events_df[REQUIRED_EVENTS_COLUMNS]
