"""Dataset discovery utilities (dyads/runs/tasks/file location)."""

import logging
import re
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)

DYAD_PATTERN = re.compile(r"dyad[-_]?(\d+)", re.IGNORECASE)
SUBJECT_PATTERN = re.compile(r"sub[-_]?(\d+)", re.IGNORECASE)
RUN_PATTERN = re.compile(r"run[-_]?(\d+)", re.IGNORECASE)

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
EEG_SUFFIXES = {".fif", ".edf", ".bdf", ".set"}
ANNOTATION_SUFFIXES = {".csv", ".tsv", ".json", ".txt", ".yaml", ".yml", ".textgrid"}


def unique_preserve(values: Iterable[str]) -> list[str]:
    """Keep first-seen order while removing duplicates."""
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def normalize_dyad_id(value: Any) -> str:
    """Normalize dyad identifier to ``dyad-XXX`` format when possible."""
    if value is None:
        raise ValueError("dyad_id cannot be null.")
    text = str(value).strip()
    if not text:
        raise ValueError("Empty dyad_id is not allowed.")

    dyad_match = DYAD_PATTERN.search(text)
    if dyad_match:
        return f"dyad-{int(dyad_match.group(1)):03d}"

    if text.isdigit():
        return f"dyad-{int(text):03d}"

    return text.lower()


def normalize_subject_id(value: Any) -> str:
    """Normalize subject identifier to ``sub-XXX`` format when possible."""
    if value is None:
        raise ValueError("subject_id cannot be null.")
    text = str(value).strip()
    if not text:
        raise ValueError("Empty subject_id is not allowed.")

    subject_match = SUBJECT_PATTERN.search(text)
    if subject_match:
        return f"sub-{int(subject_match.group(1)):03d}"

    if text.isdigit():
        return f"sub-{int(text):03d}"

    return text.lower()


def dyad_from_subject_id(subject_id: Any) -> str:
    """Map subject id to dyad id using odd/even pairing."""
    normalized_subject = normalize_subject_id(subject_id)
    match = SUBJECT_PATTERN.search(normalized_subject)
    if not match:
        raise ValueError(f"Could not derive dyad from subject_id={subject_id!r}")
    subject_num = int(match.group(1))
    dyad_num = (subject_num + 1) // 2
    return f"dyad-{dyad_num:03d}"


def normalize_task(value: Any) -> str:
    """Normalize task values to non-empty strings."""
    text = str(value).strip()
    if not text:
        raise ValueError("dataset.tasks cannot include empty values.")
    return text


def normalize_run(value: Any) -> str:
    """Normalize run values to single-digit strings when numeric."""
    if value is None:
        raise ValueError("run cannot be null.")
    text = str(value).strip()
    if not text:
        raise ValueError("dataset.runs cannot include empty values.")

    run_match = RUN_PATTERN.search(text)
    if run_match:
        return str(int(run_match.group(1)))

    if text.isdigit():
        return str(int(text))

    return text


def parse_dyad_from_path(path: Path) -> str | None:
    """Extract normalized dyad id from a path."""
    dyad_match = DYAD_PATTERN.search(path.as_posix())
    if not dyad_match:
        return None
    return f"dyad-{int(dyad_match.group(1)):03d}"


def discover_subjects_from_annotations(annotations_dir: Path) -> list[str]:
    """Discover subject ids from annotation file/directory names."""
    if not annotations_dir.exists():
        LOGGER.warning("annotations_dir does not exist: %s", annotations_dir)
        return []

    discovered: set[str] = set()
    for path in sorted(annotations_dir.rglob("*")):
        for match in SUBJECT_PATTERN.finditer(path.as_posix()):
            discovered.add(f"sub-{int(match.group(1)):03d}")

    ordered = sorted(discovered)
    LOGGER.info("Discovered %d subjects from annotations.", len(ordered))
    return ordered


def _dyad_from_subject(subject_id: int) -> str:
    """Map BIDS subject id to dyad id using odd/even pairing."""
    dyad_num = (subject_id + 1) // 2
    return f"dyad-{dyad_num:03d}"


def discover_subjects_from_bids(bids_dir: Path) -> list[str]:
    """Discover subject ids from a BIDS tree."""
    if not bids_dir.exists():
        LOGGER.warning("bids_dir does not exist: %s", bids_dir)
        return []

    discovered: set[str] = set()
    for path in sorted(bids_dir.rglob("*")):
        for match in SUBJECT_PATTERN.finditer(path.as_posix()):
            discovered.add(f"sub-{int(match.group(1)):03d}")

    ordered = sorted(discovered)
    LOGGER.info("Discovered %d subjects from BIDS.", len(ordered))
    return ordered


def _infer_dyads(explicit_dyads: set[str], subject_ids: Iterable[str]) -> list[str]:
    """Combine explicit dyads with dyads inferred from subjects."""
    discovered: set[str] = set(explicit_dyads)
    for subject_id in subject_ids:
        discovered.add(dyad_from_subject_id(subject_id))
    ordered = sorted(discovered)
    return ordered


def discover_dyads_from_annotations(annotations_dir: Path) -> list[str]:
    """Discover dyad ids from annotation file/directory names."""
    if not annotations_dir.exists():
        LOGGER.warning("annotations_dir does not exist: %s", annotations_dir)
        return []

    explicit: set[str] = set()
    for path in sorted(annotations_dir.rglob("*")):
        dyad_id = parse_dyad_from_path(path)
        if dyad_id:
            explicit.add(dyad_id)

    subjects = discover_subjects_from_annotations(annotations_dir)
    ordered = _infer_dyads(explicit, subjects)
    LOGGER.info("Discovered %d dyads from annotations.", len(ordered))
    return ordered


def discover_dyads_from_bids(bids_dir: Path) -> list[str]:
    """Discover dyad ids from a BIDS tree."""
    if not bids_dir.exists():
        LOGGER.warning("bids_dir does not exist: %s", bids_dir)
        return []

    explicit: set[str] = set()
    for path in sorted(bids_dir.rglob("*")):
        dyad_id = parse_dyad_from_path(path)
        if dyad_id:
            explicit.add(dyad_id)

    subjects = discover_subjects_from_bids(bids_dir)
    ordered = _infer_dyads(explicit, subjects)
    LOGGER.info("Discovered %d dyads from BIDS.", len(ordered))
    return ordered


def discover_subjects(annotations_dir: Path, bids_dir: Path) -> list[str]:
    """Discover subject ids using configured source precedence."""
    from_annotations = discover_subjects_from_annotations(annotations_dir)
    if from_annotations:
        LOGGER.info("Using annotations-based subject discovery.")
        return from_annotations

    LOGGER.info("No subjects discovered from annotations; falling back to BIDS.")
    return discover_subjects_from_bids(bids_dir)


def discover_dyads(annotations_dir: Path, bids_dir: Path) -> list[str]:
    """Discover dyad ids using configured source precedence."""
    from_annotations = discover_dyads_from_annotations(annotations_dir)
    if from_annotations:
        LOGGER.info("Using annotations-based dyad discovery.")
        return from_annotations

    LOGGER.info("No dyads discovered from annotations; falling back to BIDS.")
    return discover_dyads_from_bids(bids_dir)


def subject_tokens_for_dyad(dyad_id: str) -> list[str]:
    """Build subject token candidates for odd/even paired dyads."""
    match = DYAD_PATTERN.search(dyad_id)
    if not match:
        return []
    dyad_num = int(match.group(1))
    odd_subject = (2 * dyad_num) - 1
    even_subject = 2 * dyad_num

    tokens: list[str] = []
    for width in (3, 2):
        tokens.extend(
            [
                f"sub-{odd_subject:0{width}d}",
                f"sub-{even_subject:0{width}d}",
            ]
        )
    return tokens


def find_unit_path(root_dir: Path, dyad_id: str, task: str, run: str, suffixes: set[str]) -> str | None:
    """Find best-matching file path for one dyad/task/run unit."""
    if not root_dir.exists():
        return None

    dyad_tokens = unique_preserve([dyad_id.lower(), dyad_id.replace("-", "").lower(), *subject_tokens_for_dyad(dyad_id)])
    run_tokens = [f"run-{run}".lower(), f"run_{run}".lower(), f"run{run}".lower()]
    if run.isdigit():
        run_tokens.extend([f"run-{int(run):02d}", f"run_{int(run):02d}", f"run{int(run):02d}"])
    run_tokens = unique_preserve(run_tokens)
    task_tokens = unique_preserve([task.lower(), f"task-{task}".lower(), f"task_{task}".lower()])

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

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


__all__ = [
    "ANNOTATION_SUFFIXES",
    "AUDIO_SUFFIXES",
    "EEG_SUFFIXES",
    "dyad_from_subject_id",
    "discover_dyads",
    "discover_dyads_from_annotations",
    "discover_dyads_from_bids",
    "discover_subjects_from_annotations",
    "discover_subjects_from_bids",
    "discover_subjects",
    "find_unit_path",
    "normalize_dyad_id",
    "normalize_run",
    "normalize_subject_id",
    "normalize_task",
    "unique_preserve",
]
