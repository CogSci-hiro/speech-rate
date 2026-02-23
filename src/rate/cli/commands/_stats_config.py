from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from rate.config import (
    ConfigValidationError,
    apply_config_defaults,
    load_and_validate_config,
    load_yaml_config,
    validate_config,
)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], Mapping)
            and isinstance(value, Mapping)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _looks_like_stats_overlay(raw: Mapping[str, Any]) -> bool:
    keys = set(raw.keys())
    return bool(keys) and keys.issubset({"paths", "reporting", "stats", "debug"})


def load_stats_command_config(
    config_path: Path | str,
    *,
    schema_path: Path | str,
    base_config_path: Path | str = Path("config/compute.yaml"),
) -> dict[str, Any]:
    """
    Load stats command config.

    Supports either:
    - a full project config (validated directly), or
    - a stats-only overlay (e.g., config/stats.yaml), merged onto compute config.
    """
    cfg_path = Path(config_path)
    schema = Path(schema_path)

    try:
        return load_and_validate_config(cfg_path, schema_path=schema)
    except ConfigValidationError:
        raw_overlay = load_yaml_config(cfg_path)
        if not _looks_like_stats_overlay(raw_overlay):
            raise

        raw_base = load_yaml_config(Path(base_config_path))
        merged = _deep_merge(raw_base, raw_overlay)
        resolved = apply_config_defaults(merged)
        validate_config(resolved, schema_path=schema)
        return resolved
