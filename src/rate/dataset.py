"""SG1 dataset index discovery and parquet writing."""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from rate.config import ConfigValidationError, apply_config_defaults
from rate.util import ensure_parent_dir

LOGGER = logging.getLogger(__name__)

DYAD_PATTERN = re.compile(r"dyad[-_]?(\d+)", re.IGNORECASE)
SUBJECT_PATTERN = re.compile(r"sub[-_]?(\d+)", re.IGNORECASE)
RUN_PATTERN = re.compile(r"run[-_]?(\d+)", re.IGNORECASE)

DEFAULT_MISSING_REASON = "listed in dataset.missing_runs"
DEFAULT_EXCLUDED_REASON = "listed in dataset.exclude_runs"
INDEX_RELATIVE_PATH = Path("dataset") / "index.parquet"

REQUIRED_INDEX_COLUMNS = [
    "dyad_id",
    "subject_id",
    "role",
    "task",
    "run",
    "eeg_path",
    "audio_path",
    "annotations_path",
    "status",
    "reason",
]

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
EEG_SUFFIXES = {".fif", ".edf", ".bdf", ".set"}
ANNOTATION_SUFFIXES = {".csv", ".tsv", ".json", ".txt", ".yaml", ".yml", ".textgrid"}
VALID_STATUSES = {"ok", "missing", "excluded"}
DARWIN_BROTLI_LIB = Path("/opt/anaconda3/lib/libbrotlicommon.1.1.0.dylib")


class DatasetIndexError(RuntimeError):
    """Raised when dataset index generation fails."""


def _unique_preserve(values: Iterable[str]) -> list[str]:
    """Keep first-seen order while removing duplicates."""
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _normalize_dyad_id(value: Any) -> str:
    """Normalize dyad identifier to ``dyad-XX`` format when possible."""
    if value is None:
        raise DatasetIndexError("dyad_id cannot be null.")
    text = str(value).strip()
    if not text:
        raise DatasetIndexError("Empty dyad_id is not allowed.")

    dyad_match = DYAD_PATTERN.search(text)
    if dyad_match:
        return f"dyad-{int(dyad_match.group(1)):02d}"

    if text.isdigit():
        return f"dyad-{int(text):02d}"

    return text.lower()


def _normalize_task(value: Any) -> str:
    """Normalize task values to non-empty strings."""
    text = str(value).strip()
    if not text:
        raise DatasetIndexError("dataset.tasks cannot include empty values.")
    return text


def _normalize_run(value: Any) -> str:
    """Normalize run values to two-digit strings when numeric."""
    if value is None:
        raise DatasetIndexError("run cannot be null.")
    text = str(value).strip()
    if not text:
        raise DatasetIndexError("dataset.runs cannot include empty values.")

    run_match = RUN_PATTERN.search(text)
    if run_match:
        return f"{int(run_match.group(1)):02d}"

    if text.isdigit():
        return f"{int(text):02d}"

    return text


def _parse_dyad_from_path(path: Path) -> str | None:
    """Extract normalized dyad id from a path."""
    dyad_match = DYAD_PATTERN.search(path.as_posix())
    if not dyad_match:
        return None
    return f"dyad-{int(dyad_match.group(1)):02d}"


def discover_dyads_from_annotations(annotations_dir: Path) -> list[str]:
    """Discover dyad ids from annotation file/directory names.

    Parameters
    ----------
    annotations_dir
        Root directory for annotation files.

    Returns
    -------
    list[str]
        Sorted dyad ids.
    """
    if not annotations_dir.exists():
        LOGGER.warning("annotations_dir does not exist: %s", annotations_dir)
        return []

    discovered: set[str] = set()
    for path in sorted(annotations_dir.rglob("*")):
        dyad_id = _parse_dyad_from_path(path)
        if dyad_id:
            discovered.add(dyad_id)

    ordered = sorted(discovered)
    LOGGER.info("Discovered %d dyads from annotations.", len(ordered))
    return ordered


def _dyad_from_subject(subject_id: int) -> str:
    """Map BIDS subject id to dyad id using odd/even pairing."""
    dyad_num = (subject_id + 1) // 2
    return f"dyad-{dyad_num:02d}"


def discover_dyads_from_bids(bids_dir: Path) -> list[str]:
    """Discover dyad ids from BIDS layout using explicit dyad names or sub-* ids."""
    if not bids_dir.exists():
        LOGGER.warning("bids_dir does not exist: %s", bids_dir)
        return []

    discovered: set[str] = set()
    subject_ids: set[int] = set()

    for path in sorted(bids_dir.rglob("*")):
        explicit_dyad = _parse_dyad_from_path(path)
        if explicit_dyad:
            discovered.add(explicit_dyad)
        for match in SUBJECT_PATTERN.finditer(path.as_posix()):
            subject_ids.add(int(match.group(1)))

    for subject_id in sorted(subject_ids):
        discovered.add(_dyad_from_subject(subject_id))

    ordered = sorted(discovered)
    LOGGER.info("Discovered %d dyads from BIDS.", len(ordered))
    return ordered


def discover_dyads(annotations_dir: Path, bids_dir: Path) -> list[str]:
    """Discover dyads using annotations first, then BIDS fallback."""
    from_annotations = discover_dyads_from_annotations(annotations_dir)
    if from_annotations:
        LOGGER.info("Using annotations-based dyad discovery.")
        return from_annotations

    LOGGER.info("No dyads discovered from annotations; falling back to BIDS.")
    return discover_dyads_from_bids(bids_dir)


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

    for item in missing_runs:
        if not isinstance(item, Mapping):
            raise DatasetIndexError("Each dataset.missing_runs entry must be a mapping.")
        key = (_normalize_dyad_id(item.get("dyad_id")), _normalize_run(item.get("run")))
        missing_map[key] = DEFAULT_MISSING_REASON

    for item in exclude_runs:
        if not isinstance(item, Mapping):
            raise DatasetIndexError("Each dataset.exclude_runs entry must be a mapping.")
        key = (_normalize_dyad_id(item.get("dyad_id")), _normalize_run(item.get("run")))
        reason = str(item.get("reason") or DEFAULT_EXCLUDED_REASON)
        excluded_map[key] = reason

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


def _subject_tokens_for_dyad(dyad_id: str) -> list[str]:
    """Build subject token candidates for odd/even paired dyads."""
    match = DYAD_PATTERN.search(dyad_id)
    if not match:
        return []
    dyad_num = int(match.group(1))
    odd_subject = (2 * dyad_num) - 1
    even_subject = 2 * dyad_num

    tokens: list[str] = []
    for width in (2, 3):
        tokens.extend(
            [
                f"sub-{odd_subject:0{width}d}",
                f"sub-{even_subject:0{width}d}",
            ]
        )
    return tokens


def _find_unit_path(root_dir: Path, dyad_id: str, task: str, run: str, suffixes: set[str]) -> str | None:
    """Find best-matching file path for one dyad/task/run unit."""
    if not root_dir.exists():
        return None

    dyad_tokens = _unique_preserve([dyad_id.lower(), dyad_id.replace("-", "").lower(), *_subject_tokens_for_dyad(dyad_id)])
    run_tokens = _unique_preserve([f"run-{run}".lower(), f"run_{run}".lower(), f"run{run}".lower()])
    task_tokens = _unique_preserve([task.lower(), f"task-{task}".lower(), f"task_{task}".lower()])

    candidates: list[tuple[int, str]] = []
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        if suffixes and path.suffix.lower() not in suffixes:
            continue

        path_text = path.as_posix().lower()
        dyad_match = any(token in path_text for token in dyad_tokens)
        run_match = any(token in path_text for token in run_tokens)
        task_match = any(token in path_text for token in task_tokens)
        if not (dyad_match and run_match):
            continue

        score = 2 if task_match else 1
        candidates.append((score, str(path)))

    if not candidates:
        return None

    # Prefer exact task matches; then lexical order for deterministic selection.
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


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

    tasks = _unique_preserve(_normalize_task(task) for task in raw_tasks)
    runs = _unique_preserve(_normalize_run(run) for run in raw_runs)

    override_keys = list(missing_map.keys()) + list(excluded_map.keys())
    override_dyads = [_normalize_dyad_id(key[0]) for key in override_keys]
    override_runs = [_normalize_run(key[1]) for key in override_keys]

    dyads = _unique_preserve([*_unique_preserve(discovered_dyads), *override_dyads])
    runs = _unique_preserve([*runs, *override_runs])
    return dyads, tasks, runs


def build_dataset_index(config: Mapping[str, Any]) -> pd.DataFrame:
    """Build canonical dataset index table.

    Parameters
    ----------
    config
        Resolved project config dictionary.

    Returns
    -------
    pandas.DataFrame
        Canonical index with stable SG1 columns.
    """
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
                    "eeg_path": _find_unit_path(eeg_dir, dyad_id=dyad_id, task=task, run=run, suffixes=EEG_SUFFIXES),
                    "audio_path": _find_unit_path(
                        bids_dir,
                        dyad_id=dyad_id,
                        task=task,
                        run=run,
                        suffixes=AUDIO_SUFFIXES,
                    ),
                    "annotations_path": _find_unit_path(
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
    duplicate_mask = index_df.duplicated(subset=["dyad_id", "run", "task"], keep=False)
    if duplicate_mask.any():
        duplicates = index_df.loc[duplicate_mask, ["dyad_id", "run", "task"]].drop_duplicates()
        duplicate_records = duplicates.to_dict(orient="records")
        raise DatasetIndexError(f"Duplicate rows detected for dyad/run/task keys: {duplicate_records}")

    bad_statuses = sorted(set(index_df["status"]) - VALID_STATUSES)
    if bad_statuses:
        raise DatasetIndexError(f"Invalid status values generated: {bad_statuses}")

    return index_df


def write_dataset_index(config: Mapping[str, Any], output_path: Path | None = None) -> Path:
    """Write dataset index parquet file.

    Parameters
    ----------
    config
        Config dictionary.
    output_path
        Optional explicit output path.

    Returns
    -------
    pathlib.Path
        Written parquet path.
    """
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
    _write_parquet(index_df, target_path)
    LOGGER.info("Wrote dataset index to %s (%d rows).", target_path, len(index_df))
    return target_path


def _write_parquet(index_df: pd.DataFrame, output_path: Path) -> None:
    """Write parquet with a platform-specific fallback for broken pyarrow installs."""
    try:
        index_df.to_parquet(output_path, index=False)
        return
    except ImportError as exc:
        if platform.system() != "Darwin" or not DARWIN_BROTLI_LIB.exists():
            raise DatasetIndexError(f"Unable to write parquet file: {exc}") from exc
        LOGGER.warning("Parquet engine import failed; retrying in subprocess with DYLD_INSERT_LIBRARIES fix.")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp_file:
        json.dump(index_df.to_dict(orient="records"), tmp_file)
        temp_json_path = Path(tmp_file.name)

    script = (
        "import json,sys,pandas as pd;"
        "rows=json.load(open(sys.argv[1],encoding='utf-8'));"
        "pd.DataFrame(rows).to_parquet(sys.argv[2],index=False)"
    )
    env = os.environ.copy()
    env["DYLD_INSERT_LIBRARIES"] = str(DARWIN_BROTLI_LIB)
    try:
        subprocess.run(
            [sys.executable, "-c", script, str(temp_json_path), str(output_path)],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise DatasetIndexError(f"Failed to write parquet via fallback subprocess: {exc.stderr}") from exc
    finally:
        temp_json_path.unlink(missing_ok=True)
