"""Configuration loading, modular merging, defaulting, and schema validation."""

from __future__ import annotations

import copy
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

LOGGER = logging.getLogger(__name__)

DEFAULT_LOGS_SUBDIR = "logs"
DEFAULT_DEBUG_ENABLED = False
DEFAULT_DEBUG_MAX_FILES: int | None = None
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "config" / "schema.json"
MODULAR_COMPUTE_FILES = {
    "dataset": "dataset.yaml",
    "preprocessing": "preprocessing.yaml",
    "epochs": "epochs.yaml",
    "margin_epochs": "margin_epochs.yaml",
    "glm": "glm.yaml",
    "fooof": "fooof.yaml",
    "lmeeeg": "lmeeeg.yaml",
    "lmeeeg_induced": "lmeeeg_induced.yaml",
    "lmeeeg_fooof": "lmeeeg_fooof.yaml",
    "lmeeeg_variance_partition": "lmeeeg.yaml",
}


class ConfigValidationError(RuntimeError):
    """Raised when a config file is invalid."""


def load_yaml_config(config_path: Path | str) -> dict[str, Any]:
    """Load a YAML config mapping from disk."""
    path = Path(config_path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigValidationError(f"Unable to read config file: {path}") from exc

    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"Invalid YAML in config file: {path}") from exc

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ConfigValidationError(f"Config root must be a mapping: {path}")
    return payload


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _maybe_attach_modular_sections(
    raw_config: Mapping[str, Any],
    *,
    config_path: Path | str,
) -> dict[str, Any]:
    """Attach sibling modular sections when loading ``config/config.yaml``."""
    path = Path(config_path)
    resolved = dict(raw_config)
    if path.name != "config.yaml":
        return resolved

    for section_name, filename in MODULAR_COMPUTE_FILES.items():
        modular_path = path.parent / filename
        if not modular_path.exists():
            continue
        modular_payload = load_yaml_config(modular_path)
        shared_defaults = {key: value for key, value in modular_payload.items() if key != section_name}
        if shared_defaults:
            resolved = _deep_merge(shared_defaults, resolved)
        if section_name not in resolved and section_name in modular_payload:
            resolved[section_name] = modular_payload[section_name]
    return resolved


def apply_config_defaults(config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply repository defaults to a config payload."""
    resolved = copy.deepcopy(dict(config))

    paths_value = resolved.setdefault("paths", {})
    if not isinstance(paths_value, dict):
        raise ConfigValidationError("`paths` must be a mapping.")

    out_dir = paths_value.get("out_dir")
    logs_dir = paths_value.get("logs_dir")
    if out_dir and not logs_dir:
        paths_value["logs_dir"] = str(Path(out_dir) / DEFAULT_LOGS_SUBDIR)

    dataset_value = resolved.setdefault("dataset", {})
    if not isinstance(dataset_value, dict):
        raise ConfigValidationError("`dataset` must be a mapping.")

    exclude_value = dataset_value.setdefault("exclude", {})
    if not isinstance(exclude_value, dict):
        raise ConfigValidationError("`dataset.exclude` must be a mapping.")
    exclude_value.setdefault("subjects", [])
    exclude_value.setdefault("subject_runs", [])
    exclude_value.setdefault("dyads", [])
    exclude_value.setdefault("dyad_runs", [])

    for section_name in (
        "preprocessing",
        "epochs",
        "margin_epochs",
        "glm",
        "fooof",
        "lmeeeg",
        "lmeeeg_induced",
        "lmeeeg_fooof",
        "lmeeeg_variance_partition",
    ):
        section_value = resolved.setdefault(section_name, {})
        if not isinstance(section_value, dict):
            raise ConfigValidationError(f"`{section_name}` must be a mapping.")

    debug_value = resolved.setdefault("debug", {})
    if debug_value is None:
        debug_value = {}
        resolved["debug"] = debug_value
    if not isinstance(debug_value, dict):
        raise ConfigValidationError("`debug` must be a mapping.")
    debug_value.setdefault("enabled", DEFAULT_DEBUG_ENABLED)
    debug_value.setdefault("max_files", DEFAULT_DEBUG_MAX_FILES)

    if not isinstance(debug_value.get("enabled"), bool):
        raise ConfigValidationError("`debug.enabled` must be a boolean.")
    max_files = debug_value.get("max_files")
    if max_files is not None:
        try:
            max_files_int = int(max_files)
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError("`debug.max_files` must be a positive integer or null.") from exc
        if max_files_int < 1:
            raise ConfigValidationError("`debug.max_files` must be >= 1 when provided.")
        debug_value["max_files"] = max_files_int

    return resolved


def _load_schema(schema_path: Path | str) -> dict[str, Any]:
    """Load a JSON schema from disk."""
    path = Path(schema_path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigValidationError(f"Unable to read schema file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(f"Invalid JSON in schema file: {path}") from exc


@lru_cache(maxsize=None)
def _load_schema_cached(schema_path: str) -> dict[str, Any]:
    """Cache parsed schemas by resolved absolute path."""
    return _load_schema(Path(schema_path))


def validate_config(config: Mapping[str, Any], schema_path: Path | str = DEFAULT_SCHEMA_PATH) -> None:
    """Validate a resolved config against the repository schema."""
    schema = _load_schema_cached(str(Path(schema_path).expanduser().resolve()))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda err: list(err.path))
    if not errors:
        return

    details: list[str] = []
    for error in errors:
        path = ".".join(str(part) for part in error.path) or "<root>"
        details.append(f"{path}: {error.message}")
    raise ConfigValidationError("Config validation failed:\n" + "\n".join(details))


def load_and_validate_config(
    config_path: Path | str,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
) -> dict[str, Any]:
    """Load YAML config, merge modular sections, apply defaults, and validate."""
    return copy.deepcopy(
        _load_and_validate_config_cached(
            str(Path(config_path).expanduser().resolve()),
            str(Path(schema_path).expanduser().resolve()),
        )
    )


@lru_cache(maxsize=None)
def _load_and_validate_config_cached(
    config_path: str,
    schema_path: str,
) -> dict[str, Any]:
    """Cached implementation backing ``load_and_validate_config``."""
    raw_config = load_yaml_config(config_path)
    merged = _maybe_attach_modular_sections(raw_config, config_path=config_path)
    resolved = apply_config_defaults(merged)
    validate_config(resolved, schema_path=schema_path)
    return resolved
