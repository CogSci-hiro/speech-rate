"""Utility helpers for SG0 scaffolding."""

import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_RANDOM_SEED = 1729
KEY_PACKAGES = ("numpy", "pandas", "mne", "snakemake")
DARWIN_BROTLI_LIB = Path("/opt/anaconda3/lib/libbrotlicommon.1.1.0.dylib")


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
    """Return the interpreter version string for the current runtime.

    Returns
    -------
    str
        Runtime Python version in dotted-string form.
    """
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


def write_parquet_dataframe(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Write a dataframe to parquet with a Darwin fallback.

    Parameters
    ----------
    dataframe
        DataFrame to serialize.
    output_path
        Destination parquet path.
    """
    try:
        dataframe.to_parquet(output_path, index=False)
        return
    except ImportError as exc:
        LOGGER.warning("Parquet engine import failed in current interpreter: %s", exc)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp_file:
        dataframe.to_json(tmp_file.name, orient="split")
        temp_json_path = Path(tmp_file.name)

    script = (
        "import pandas as pd,sys;"
        "df=pd.read_json(sys.argv[1],orient='split');"
        "df.to_parquet(sys.argv[2],index=False)"
    )
    errors: list[str] = []
    try:
        for python_exe in _candidate_python_interpreters():
            env = _fallback_env()
            try:
                subprocess.run(
                    [python_exe, "-c", script, str(temp_json_path), str(output_path)],
                    check=True,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                return
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                errors.append(f"{python_exe}: {stderr}")
                continue
    finally:
        temp_json_path.unlink(missing_ok=True)
    joined = "\n".join(errors)
    raise RuntimeError(f"Failed to write parquet via fallback interpreters:\n{joined}")


def read_parquet_dataframe(parquet_path: Path) -> pd.DataFrame:
    """Read a parquet file into a dataframe with a Darwin fallback.

    Parameters
    ----------
    parquet_path
        Source parquet file path.

    Returns
    -------
    pandas.DataFrame
        Parsed parquet data.
    """
    try:
        return pd.read_parquet(parquet_path)
    except ImportError as exc:
        LOGGER.warning("Parquet engine import failed in current interpreter: %s", exc)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp_file:
        temp_json_path = Path(tmp_file.name)

    script = (
        "import pandas as pd,sys;"
        "df=pd.read_parquet(sys.argv[1]);"
        "df.to_json(sys.argv[2],orient='split')"
    )
    errors: list[str] = []
    try:
        for python_exe in _candidate_python_interpreters():
            env = _fallback_env()
            try:
                subprocess.run(
                    [python_exe, "-c", script, str(parquet_path), str(temp_json_path)],
                    check=True,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                return pd.read_json(temp_json_path, orient="split")
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                errors.append(f"{python_exe}: {stderr}")
                continue
    finally:
        temp_json_path.unlink(missing_ok=True)
    joined = "\n".join(errors)
    raise RuntimeError(f"Failed to read parquet via fallback interpreters:\n{joined}")


def _candidate_python_interpreters() -> list[str]:
    """Return likely Python interpreters to try for parquet fallback."""
    candidates: list[str] = [sys.executable, "/opt/anaconda3/bin/python", "/usr/local/bin/python3.11"]
    for name in ("python3", "python"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(resolved)

    unique: list[str] = []
    for exe in candidates:
        if not exe:
            continue
        exe_path = str(Path(exe))
        if exe_path not in unique and Path(exe_path).exists():
            unique.append(exe_path)
    return unique


def _fallback_env() -> dict[str, str]:
    """Build environment for parquet fallback subprocesses."""
    env = os.environ.copy()
    if platform.system() == "Darwin" and DARWIN_BROTLI_LIB.exists():
        env["DYLD_INSERT_LIBRARIES"] = str(DARWIN_BROTLI_LIB)
    return env
