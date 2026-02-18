"""Utility helpers for SG0 scaffolding."""

from __future__ import annotations

import logging
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

LOGGER = logging.getLogger(__name__)

DEFAULT_RANDOM_SEED = 1729
KEY_PACKAGES = ("numpy", "pandas", "mne", "snakemake")


def ensure_parent_dir(file_path: Path) -> None:
    """Ensure a file path's parent directory exists.

    Parameters
    ----------
    file_path
        Target file path whose parent should exist.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)


def get_git_commit_hash(cwd: Path | None = None) -> str:
    """Return the current git commit hash.

    Parameters
    ----------
    cwd
        Working directory where `git` should run.

    Returns
    -------
    str
        Short git commit hash, or ``"unknown"`` when unavailable.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"

    commit_hash = completed.stdout.strip()
    return commit_hash or "unknown"


def get_python_version() -> str:
    """Return the runtime Python version."""
    return platform.python_version()


def get_package_versions(packages: Iterable[str]) -> dict[str, str]:
    """Get installed versions for requested packages.

    Parameters
    ----------
    packages
        Iterable of package names.

    Returns
    -------
    dict[str, str]
        Mapping from package name to version string. Missing packages are
        reported as ``"not installed"``.
    """
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def resolve_random_seed(config: Mapping[str, Any], default_seed: int = DEFAULT_RANDOM_SEED) -> int:
    """Resolve the random seed from config with deterministic fallback.

    Parameters
    ----------
    config
        Resolved configuration mapping.
    default_seed
        Fallback seed when config does not define one.

    Returns
    -------
    int
        Deterministic random seed.
    """
    seed_value = config.get("random_seed")
    if seed_value is None:
        return default_seed

    try:
        return int(seed_value)
    except (TypeError, ValueError):
        LOGGER.warning("Invalid random_seed=%r; using default=%d.", seed_value, default_seed)
        return default_seed
