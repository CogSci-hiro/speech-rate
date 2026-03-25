"""Dataset discovery utilities (dyads/runs/tasks/file location)."""

from dataclasses import dataclass
from functools import lru_cache
import logging
import re
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)

DYAD_PATTERN = re.compile(r"dyad[-_]?(\d+)", re.IGNORECASE)
SUBJECT_PATTERN = re.compile(r"sub[-_]?(\d+)", re.IGNORECASE)
RUN_PATTERN = re.compile(r"run[-_]?(\d+)", re.IGNORECASE)
TASK_PATTERN = re.compile(r"task[-_]?([a-z0-9]+)", re.IGNORECASE)

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
EEG_SUFFIXES = {".fif", ".edf", ".bdf", ".set", ".vhdr"}
ANNOTATION_SUFFIXES = {".csv", ".tsv", ".json", ".txt", ".yaml", ".yml", ".textgrid"}
ANNOTATION_PREFER_TOKENS = ("ipu",)
ANNOTATION_EXCLUDE_TOKENS = ("channels",)


@dataclass(frozen=True)
class IndexedFile:
    """Cached metadata for one filesystem entry."""

    path: Path
    path_text: str
    suffix: str
    subjects: frozenset[str]
    dyads: frozenset[str]
    runs: frozenset[str]
    tasks: frozenset[str]


@dataclass(frozen=True)
class DirectorySnapshot:
    """Cached scan results for one root directory."""

    root_dir: Path
    files: tuple[IndexedFile, ...]
    files_by_suffix: dict[str, tuple[IndexedFile, ...]]
    files_by_run: dict[str, tuple[IndexedFile, ...]]
    discovered_subjects: tuple[str, ...]
    explicit_dyads: tuple[str, ...]


def _is_ignored_artifact(path: Path) -> bool:
    """Return True for filesystem artifacts that should never be treated as data."""
    for part in path.parts:
        if part.startswith("._"):
            return True
        if part in {".DS_Store", "__MACOSX"}:
            return True
    return False


def _contains_token(path_text: str, token: str) -> bool:
    """Match token with non-alphanumeric boundaries to avoid partial-id collisions."""
    escaped = re.escape(token.lower())
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", path_text) is not None


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


def _extract_subject_ids(path_text: str) -> frozenset[str]:
    return frozenset(f"sub-{int(match.group(1)):03d}" for match in SUBJECT_PATTERN.finditer(path_text))


def _extract_dyad_ids(path_text: str) -> frozenset[str]:
    return frozenset(f"dyad-{int(match.group(1)):03d}" for match in DYAD_PATTERN.finditer(path_text))


def _extract_runs(path_text: str) -> frozenset[str]:
    return frozenset(str(int(match.group(1))) for match in RUN_PATTERN.finditer(path_text))


def _extract_tasks(path_text: str) -> frozenset[str]:
    return frozenset(match.group(1).lower() for match in TASK_PATTERN.finditer(path_text))


@lru_cache(maxsize=None)
def scan_directory(root_dir: Path | str) -> DirectorySnapshot:
    """Scan a directory tree once and cache file metadata plus discovered ids."""
    path = Path(root_dir).expanduser()
    if not path.exists():
        return DirectorySnapshot(
            root_dir=path,
            files=(),
            files_by_suffix={},
            files_by_run={},
            discovered_subjects=(),
            explicit_dyads=(),
        )

    files: list[IndexedFile] = []
    files_by_suffix: dict[str, list[IndexedFile]] = {}
    files_by_run: dict[str, list[IndexedFile]] = {}
    discovered_subjects: set[str] = set()
    explicit_dyads: set[str] = set()

    for child in sorted(path.rglob("*")):
        if _is_ignored_artifact(child):
            continue

        child_text = child.as_posix().lower()
        for match in SUBJECT_PATTERN.finditer(child_text):
            discovered_subjects.add(f"sub-{int(match.group(1)):03d}")

        dyad_id = parse_dyad_from_path(child)
        if dyad_id:
            explicit_dyads.add(dyad_id)

        if not child.is_file():
            continue

        subjects = _extract_subject_ids(child_text)
        dyads = _extract_dyad_ids(child_text)
        runs = _extract_runs(child_text)
        tasks = _extract_tasks(child_text)
        entry = IndexedFile(
            path=child,
            path_text=child_text,
            suffix=child.suffix.lower(),
            subjects=subjects,
            dyads=dyads,
            runs=runs,
            tasks=tasks,
        )
        files.append(entry)
        files_by_suffix.setdefault(entry.suffix, []).append(entry)
        for run in entry.runs:
            files_by_run.setdefault(run, []).append(entry)

    return DirectorySnapshot(
        root_dir=path,
        files=tuple(files),
        files_by_suffix={suffix: tuple(entries) for suffix, entries in files_by_suffix.items()},
        files_by_run={run: tuple(entries) for run, entries in files_by_run.items()},
        discovered_subjects=tuple(sorted(discovered_subjects)),
        explicit_dyads=tuple(sorted(explicit_dyads)),
    )


def discover_subjects_from_annotations(annotations_dir: Path) -> list[str]:
    """Discover subject ids from annotation file/directory names."""
    snapshot = scan_directory(annotations_dir)
    if not snapshot.root_dir.exists():
        LOGGER.warning("annotations_dir does not exist: %s", annotations_dir)
        return []

    ordered = list(snapshot.discovered_subjects)
    LOGGER.info("Discovered %d subjects from annotations.", len(ordered))
    return ordered


def _dyad_from_subject(subject_id: int) -> str:
    """Map BIDS subject id to dyad id using odd/even pairing."""
    dyad_num = (subject_id + 1) // 2
    return f"dyad-{dyad_num:03d}"


def discover_subjects_from_bids(bids_dir: Path) -> list[str]:
    """Discover subject ids from a BIDS tree."""
    snapshot = scan_directory(bids_dir)
    if not snapshot.root_dir.exists():
        LOGGER.warning("bids_dir does not exist: %s", bids_dir)
        return []

    ordered = list(snapshot.discovered_subjects)
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
    snapshot = scan_directory(annotations_dir)
    if not snapshot.root_dir.exists():
        LOGGER.warning("annotations_dir does not exist: %s", annotations_dir)
        return []

    ordered = _infer_dyads(set(snapshot.explicit_dyads), snapshot.discovered_subjects)
    LOGGER.info("Discovered %d dyads from annotations.", len(ordered))
    return ordered


def discover_dyads_from_bids(bids_dir: Path) -> list[str]:
    """Discover dyad ids from a BIDS tree."""
    snapshot = scan_directory(bids_dir)
    if not snapshot.root_dir.exists():
        LOGGER.warning("bids_dir does not exist: %s", bids_dir)
        return []

    ordered = _infer_dyads(set(snapshot.explicit_dyads), snapshot.discovered_subjects)
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


def _files_for_query(snapshot: DirectorySnapshot, *, suffixes: set[str], run: str) -> Iterable[IndexedFile]:
    run_key = normalize_run(run)
    if run_key in snapshot.files_by_run:
        base_entries = snapshot.files_by_run[run_key]
    else:
        base_entries = snapshot.files

    if not suffixes:
        return base_entries

    allowed_entries = {
        id(entry)
        for suffix in {suffix.lower() for suffix in suffixes}
        for entry in snapshot.files_by_suffix.get(suffix, ())
    }
    return [entry for entry in base_entries if id(entry) in allowed_entries]


def find_unit_path(
    root_dir: Path | DirectorySnapshot,
    dyad_id: str,
    task: str,
    run: str,
    suffixes: set[str],
    subject_id: str | None = None,
) -> str | None:
    """Find best-matching file path for one dyad/task/run unit."""
    snapshot = root_dir if isinstance(root_dir, DirectorySnapshot) else scan_directory(root_dir)
    if not snapshot.root_dir.exists():
        return None

    dyad_subjects = frozenset(subject_tokens_for_dyad(dyad_id))
    subject_id_norm: str | None = None
    if subject_id:
        subject_id_norm = normalize_subject_id(subject_id)
    run_norm = normalize_run(run)
    task_norm = normalize_task(task).lower()

    candidates: list[tuple[int, int, str]] = []

    for entry in _files_for_query(snapshot, suffixes=suffixes, run=run_norm):
        if run_norm not in entry.runs:
            continue

        task_match = task_norm in entry.tasks or _contains_token(entry.path_text, task_norm)
        dyad_match = dyad_id in entry.dyads or bool(entry.subjects & dyad_subjects)

        if subject_id_norm is not None:
            subject_match = subject_id_norm in entry.subjects
            if subject_match:
                candidates.append((0, 2 if task_match else 1, str(entry.path)))
                continue
            if dyad_match:
                candidates.append((1, 2 if task_match else 1, str(entry.path)))
            continue

        if dyad_match:
            candidates.append((0, 2 if task_match else 1, str(entry.path)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], -item[1], item[2]))
    return candidates[0][2]


def find_annotation_path(
    root_dir: Path | DirectorySnapshot,
    dyad_id: str,
    task: str,
    run: str,
    subject_id: str | None = None,
) -> str | None:
    """Find best-matching annotation path, preferring IPU files over channel metadata."""
    snapshot = root_dir if isinstance(root_dir, DirectorySnapshot) else scan_directory(root_dir)
    if not snapshot.root_dir.exists():
        return None

    subject_id_norm = normalize_subject_id(subject_id) if subject_id else None
    dyad_subjects = frozenset(subject_tokens_for_dyad(dyad_id))
    run_norm = normalize_run(run)
    task_norm = normalize_task(task).lower()

    candidates: list[tuple[int, str]] = []
    for entry in _files_for_query(snapshot, suffixes=ANNOTATION_SUFFIXES, run=run_norm):
        path_text = entry.path_text
        if any(token in path_text for token in ANNOTATION_EXCLUDE_TOKENS):
            continue
        if run_norm not in entry.runs:
            continue
        if subject_id_norm is not None:
            dyad_match = subject_id_norm in entry.subjects
        else:
            dyad_match = dyad_id in entry.dyads or bool(entry.subjects & dyad_subjects)
        if not dyad_match:
            continue

        score = 1
        if "ipu" in path_text or "/ipu_" in path_text or "/ipu/" in path_text:
            score += 2
        if task_norm in entry.tasks or _contains_token(path_text, task_norm):
            score += 1
        if any(token in path_text for token in ANNOTATION_PREFER_TOKENS):
            score += 3
        candidates.append((score, str(entry.path)))

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
    "find_annotation_path",
    "find_unit_path",
    "normalize_dyad_id",
    "normalize_run",
    "normalize_subject_id",
    "normalize_task",
    "unique_preserve",
]
