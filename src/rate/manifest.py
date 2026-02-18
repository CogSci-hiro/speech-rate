"""Run manifest creation helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from rate.config import ConfigValidationError, apply_config_defaults
from rate.util import (
    DEFAULT_RANDOM_SEED,
    KEY_PACKAGES,
    ensure_parent_dir,
    get_git_commit_hash,
    get_package_versions,
    get_python_version,
    resolve_random_seed,
)

LOGGER = logging.getLogger(__name__)

MANIFEST_RELATIVE_PATH = Path("_meta") / "run_manifest.json"


def build_run_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the manifest payload for one run.

    Parameters
    ----------
    config
        Config mapping. Defaults are applied before manifest creation.

    Returns
    -------
    dict[str, Any]
        Manifest payload.
    """
    resolved_config = apply_config_defaults(config)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": get_git_commit_hash(),
        "resolved_config": resolved_config,
        "python_version": get_python_version(),
        "package_versions": get_package_versions(KEY_PACKAGES),
        "random_seed": resolve_random_seed(resolved_config, default_seed=DEFAULT_RANDOM_SEED),
    }


def _default_manifest_path(config: Mapping[str, Any]) -> Path:
    """Compute default manifest path based on ``paths.out_dir``."""
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ConfigValidationError("Config is missing required object: paths")

    out_dir = paths.get("out_dir")
    if not out_dir:
        raise ConfigValidationError("Config is missing required key: paths.out_dir")

    return Path(out_dir) / MANIFEST_RELATIVE_PATH


def write_run_manifest(config: Mapping[str, Any], manifest_path: Path | None = None) -> Path:
    """Write run manifest JSON to disk.

    Parameters
    ----------
    config
        Config mapping. Defaults are applied before manifest creation.
    manifest_path
        Explicit output path. Uses ``{out_dir}/_meta/run_manifest.json`` when omitted.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    payload = build_run_manifest(config)
    output_path = manifest_path or _default_manifest_path(payload["resolved_config"])
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Wrote run manifest to %s", output_path)
    return output_path
