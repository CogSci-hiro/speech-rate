"""I/O helpers for canonical event tables and QC outputs."""

import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from rate.config import ConfigValidationError, apply_config_defaults
from rate.debug import limit_dataframe_by_debug_max_files
from rate.paths import ensure_parent_dir
from rate.util import read_parquet_dataframe, write_parquet_dataframe

LOGGER = logging.getLogger(__name__)

INDEX_RELATIVE_PATH = Path("dataset") / "index.parquet"
EVENTS_RELATIVE_PATH = Path("events") / "events.parquet"
QC_RELATIVE_PATH = Path("events") / "qc_ipu_summary.csv"

ANNOTATION_ALIASES = {
    "t_onset": ["t_onset", "onset", "start", "start_s"],
    "t_offset": ["t_offset", "offset", "end", "end_s"],
    "syllables": ["syllables", "n_syllables", "syllable_count"],
    "speaker": ["speaker", "talker", "who"],
}


class EventsTableError(RuntimeError):
    """Raised when canonical events generation fails."""


def _find_alias(df: pd.DataFrame, key: str) -> str | None:
    """Find the first matching column alias for a canonical field."""
    for candidate in ANNOTATION_ALIASES[key]:
        if candidate in df.columns:
            return candidate
    return None


def load_ipu_annotations(annotations_path: Path) -> pd.DataFrame:
    """Load one annotation file into canonical IPU columns."""
    suffix = annotations_path.suffix.lower()
    if suffix == ".csv":
        raw_df = pd.read_csv(annotations_path)
    elif suffix == ".tsv":
        raw_df = pd.read_csv(annotations_path, sep="\t")
    elif suffix == ".json":
        raw_df = pd.read_json(annotations_path)
    else:
        raise EventsTableError(f"Unsupported annotation file format: {annotations_path}")

    onset_col = _find_alias(raw_df, "t_onset")
    offset_col = _find_alias(raw_df, "t_offset")
    syll_col = _find_alias(raw_df, "syllables")
    speaker_col = _find_alias(raw_df, "speaker")

    missing_required: list[str] = []
    if not onset_col:
        missing_required.append("t_onset")
    if not offset_col:
        missing_required.append("t_offset")
    if not syll_col:
        missing_required.append("syllables")
    if missing_required:
        raise EventsTableError(
            f"Annotation file {annotations_path} missing required columns: {', '.join(missing_required)}"
        )

    canonical_df = pd.DataFrame(
        {
            "t_onset": pd.to_numeric(raw_df[onset_col], errors="coerce"),
            "t_offset": pd.to_numeric(raw_df[offset_col], errors="coerce"),
            "syllables": pd.to_numeric(raw_df[syll_col], errors="coerce"),
            "speaker": raw_df[speaker_col].astype(str) if speaker_col else "speaker_1",
        }
    )
    canonical_df = canonical_df.dropna(subset=["t_onset", "t_offset", "syllables"]).reset_index(drop=True)
    return canonical_df


def write_events_outputs(
    config: Mapping[str, Any],
    dataset_index_path: Path | None = None,
    events_output_path: Path | None = None,
    qc_output_path: Path | None = None,
) -> tuple[Path, Path]:
    """Write events parquet and QC CSV outputs."""
    from rate.events.build import build_events_table
    from rate.events.qc import build_ipu_qc_summary

    resolved_config = apply_config_defaults(config)
    paths_cfg = resolved_config.get("paths")
    if not isinstance(paths_cfg, Mapping):
        raise ConfigValidationError("Config missing required object: paths")

    out_dir = paths_cfg.get("out_dir")
    if not out_dir:
        raise ConfigValidationError("Config missing required key: paths.out_dir")

    out_dir_path = Path(str(out_dir))
    dataset_path = dataset_index_path or out_dir_path / INDEX_RELATIVE_PATH
    events_path = events_output_path or out_dir_path / EVENTS_RELATIVE_PATH
    qc_path = qc_output_path or out_dir_path / QC_RELATIVE_PATH

    if not dataset_path.exists():
        raise EventsTableError(f"Dataset index parquet does not exist: {dataset_path}")

    try:
        dataset_index_df = read_parquet_dataframe(dataset_path)
    except RuntimeError as exc:
        raise EventsTableError(str(exc)) from exc
    dataset_index_df = limit_dataframe_by_debug_max_files(
        dataset_index_df,
        resolved_config,
        logger=LOGGER,
        label="events dataset index",
    )

    events_df = build_events_table(resolved_config, dataset_index_df)
    qc_df = build_ipu_qc_summary(events_df)

    ensure_parent_dir(events_path)
    ensure_parent_dir(qc_path)
    try:
        write_parquet_dataframe(events_df, events_path)
    except RuntimeError as exc:
        raise EventsTableError(str(exc)) from exc
    qc_df.to_csv(qc_path, index=False)

    LOGGER.info("Wrote events table: %s (%d rows).", events_path, len(events_df))
    LOGGER.info("Wrote events QC summary: %s (%d rows).", qc_path, len(qc_df))
    return events_path, qc_path
