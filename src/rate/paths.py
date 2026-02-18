"""Output path helpers for the rate package."""

from pathlib import Path
from typing import Any, Mapping


def ensure_parent_dir(file_path: Path) -> None:
    """Ensure a file path's parent directory exists."""
    file_path.parent.mkdir(parents=True, exist_ok=True)


def resolve_out_dir(config: Mapping[str, Any]) -> Path:
    """Resolve ``paths.out_dir`` from a config mapping."""
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("Config is missing required object: paths")

    out_dir = paths.get("out_dir")
    if not out_dir:
        raise ValueError("Config is missing required key: paths.out_dir")

    return Path(str(out_dir))


def resolve_logs_dir(config: Mapping[str, Any]) -> Path:
    """Resolve ``paths.logs_dir`` from a config mapping."""
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("Config is missing required object: paths")

    logs_dir = paths.get("logs_dir")
    if not logs_dir:
        raise ValueError("Config is missing required key: paths.logs_dir")

    return Path(str(logs_dir))
