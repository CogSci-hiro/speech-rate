"""Questionnaire loading and normalization helpers for events metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from rate.dataset.discover import normalize_subject_id

LOGGER = logging.getLogger(__name__)

LIKEABILITY_COLUMNS = ("subject_id", "Score")
PERSONALITY_COLUMNS = (
    "subject_id",
    "extraversion",
    "agreeableness",
    "conscienciousness",
    "neuroticism",
    "openness",
)
QUESTIONNAIRE_OUTPUT_COLUMNS = (
    "subject_id",
    "likeability",
    "extraversion",
    "agreeableness",
    "conscienciousness",
    "neuroticism",
    "openness",
)


class QuestionnaireError(RuntimeError):
    """Raised when questionnaire inputs are malformed."""


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise QuestionnaireError(f"Questionnaire file does not exist: {path}") from exc
    except Exception as exc:  # pragma: no cover - pandas exception taxonomy is broad
        raise QuestionnaireError(f"Failed to read questionnaire file: {path}") from exc


def _require_columns(frame: pd.DataFrame, *, path: Path, required_columns: tuple[str, ...]) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise QuestionnaireError(f"Questionnaire file {path} is missing required columns: {missing_text}")


def _normalize_questionnaire_subject_ids(frame: pd.DataFrame, *, path: Path) -> pd.DataFrame:
    normalized = frame.copy()
    try:
        normalized["subject_id"] = normalized["subject_id"].map(normalize_subject_id)
    except ValueError as exc:
        raise QuestionnaireError(f"Invalid questionnaire subject_id values in {path}: {exc}") from exc

    duplicate_mask = normalized.duplicated(subset=["subject_id"], keep=False)
    if duplicate_mask.any():
        duplicates = sorted(set(normalized.loc[duplicate_mask, "subject_id"].astype(str)))
        duplicate_text = ", ".join(duplicates)
        raise QuestionnaireError(f"Duplicate questionnaire subject_id rows found in {path}: {duplicate_text}")
    return normalized


def load_likeability_table(path: Path) -> pd.DataFrame:
    """Load and validate likeability questionnaire data.

    Parameters
    ----------
    path
        Path to ``likeability.csv``.

    Returns
    -------
    pandas.DataFrame
        Subject-level table with ``subject_id`` and ``likeability``.
    """
    frame = _load_csv(path)
    _require_columns(frame, path=path, required_columns=LIKEABILITY_COLUMNS)
    normalized = _normalize_questionnaire_subject_ids(frame.loc[:, LIKEABILITY_COLUMNS], path=path)
    normalized = normalized.rename(columns={"Score": "likeability"})
    return normalized


def load_personality_table(path: Path) -> pd.DataFrame:
    """Load and validate personality questionnaire data.

    Parameters
    ----------
    path
        Path to ``personality.csv``.

    Returns
    -------
    pandas.DataFrame
        Subject-level table with normalized personality columns.
    """
    frame = _load_csv(path)
    _require_columns(frame, path=path, required_columns=PERSONALITY_COLUMNS)
    normalized = _normalize_questionnaire_subject_ids(frame.loc[:, PERSONALITY_COLUMNS], path=path)
    return normalized


def resolve_questionnaire_paths(config: Mapping[str, Any]) -> tuple[Path, Path]:
    """Resolve questionnaire file paths from config.

    Parameters
    ----------
    config
        Resolved repository config.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Likeability and personality CSV paths.
    """
    paths_cfg = config.get("paths")
    if not isinstance(paths_cfg, Mapping):
        raise QuestionnaireError("Config is missing required object: paths")

    questionnaire_dir_value = paths_cfg.get("questionnaire_dir")
    if not questionnaire_dir_value:
        raise QuestionnaireError("Config paths must include questionnaire_dir for questionnaire loading.")
    questionnaire_dir = Path(str(questionnaire_dir_value)).expanduser()

    dataset_cfg = config.get("dataset")
    if dataset_cfg is None:
        dataset_cfg = {}
    if not isinstance(dataset_cfg, Mapping):
        raise QuestionnaireError("Config object dataset must be a mapping.")

    questionnaires_cfg = dataset_cfg.get("questionnaires")
    if questionnaires_cfg is None:
        questionnaires_cfg = {}
    if not isinstance(questionnaires_cfg, Mapping):
        raise QuestionnaireError("Config object dataset.questionnaires must be a mapping.")

    likeability_name = str(questionnaires_cfg.get("likeability_filename", "likeability.csv"))
    personality_name = str(questionnaires_cfg.get("personality_filename", "personality.csv"))
    return questionnaire_dir / likeability_name, questionnaire_dir / personality_name


def build_subject_questionnaire_table(config: Mapping[str, Any]) -> pd.DataFrame:
    """Build the merged subject-level questionnaire metadata table.

    Parameters
    ----------
    config
        Resolved repository config.

    Returns
    -------
    pandas.DataFrame
        One row per subject with the questionnaire-derived variables required
        by the events-definition stage.
    """
    likeability_path, personality_path = resolve_questionnaire_paths(config)
    likeability_df = load_likeability_table(likeability_path)
    personality_df = load_personality_table(personality_path)

    merged = likeability_df.merge(
        personality_df,
        how="outer",
        on="subject_id",
        validate="one_to_one",
    )
    return merged.loc[:, QUESTIONNAIRE_OUTPUT_COLUMNS]


def summarize_questionnaire_coverage(
    *,
    events_subject_ids: set[str],
    questionnaire_subject_ids: set[str],
) -> dict[str, Any]:
    """Summarize questionnaire/event subject matching for reporting."""
    missing_in_questionnaires = sorted(events_subject_ids - questionnaire_subject_ids)
    unused_questionnaires = sorted(questionnaire_subject_ids - events_subject_ids)
    return {
        "event_subject_count": len(events_subject_ids),
        "questionnaire_subject_count": len(questionnaire_subject_ids),
        "matched_subject_count": len(events_subject_ids & questionnaire_subject_ids),
        "events_without_questionnaires": missing_in_questionnaires,
        "questionnaires_without_events": unused_questionnaires,
    }


def log_questionnaire_coverage(summary: Mapping[str, Any]) -> None:
    """Log non-fatal questionnaire coverage mismatches."""
    if summary.get("events_without_questionnaires"):
        LOGGER.warning(
            "Questionnaire data missing for %d event subjects: %s",
            len(summary["events_without_questionnaires"]),
            ", ".join(summary["events_without_questionnaires"]),
        )
    if summary.get("questionnaires_without_events"):
        LOGGER.warning(
            "Questionnaire rows found for %d subjects not present in events: %s",
            len(summary["questionnaires_without_events"]),
            ", ".join(summary["questionnaires_without_events"]),
        )
