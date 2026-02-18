"""Configuration loading, defaulting, and schema validation."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

LOGGER = logging.getLogger(__name__)

DEFAULT_LOGS_SUBDIR = "logs"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "config" / "schema.json"


class ConfigValidationError(RuntimeError):
    """Raised when a config file is invalid."""


def load_yaml_config(config_path: Path | str) -> dict[str, Any]:
    """Load YAML config from disk.

    Parameters
    ----------
    config_path
        YAML config file path.

    Returns
    -------
    dict[str, Any]
        Parsed config dictionary.
    """
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


def apply_config_defaults(config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply SG0 defaults to a config payload.

    Parameters
    ----------
    config
        Raw configuration dictionary.

    Returns
    -------
    dict[str, Any]
        Deep-copied resolved config.
    """
    resolved = copy.deepcopy(dict(config))
    paths_value = resolved.setdefault("paths", {})
    if not isinstance(paths_value, dict):
        raise ConfigValidationError("`paths` must be a mapping.")

    out_dir = paths_value.get("out_dir")
    logs_dir = paths_value.get("logs_dir")
    if out_dir and not logs_dir:
        paths_value["logs_dir"] = str(Path(out_dir) / DEFAULT_LOGS_SUBDIR)

    return resolved


def _load_schema(schema_path: Path | str) -> dict[str, Any]:
    """Load JSON schema from disk."""
    path = Path(schema_path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigValidationError(f"Unable to read schema file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(f"Invalid JSON in schema file: {path}") from exc


def validate_config(config: Mapping[str, Any], schema_path: Path | str = DEFAULT_SCHEMA_PATH) -> None:
    """Validate config against a Draft 2020-12 JSON schema.

    Parameters
    ----------
    config
        Resolved configuration mapping.
    schema_path
        Schema file path.
    """
    schema = _load_schema(schema_path)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda err: list(err.path))
    if not errors:
        return

    details: list[str] = []
    for err in errors:
        path = ".".join(str(part) for part in err.path) or "<root>"
        details.append(f"{path}: {err.message}")

    message = "Config validation failed:\n" + "\n".join(details)
    raise ConfigValidationError(message)


def load_and_validate_config(
    config_path: Path | str,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
) -> dict[str, Any]:
    """Load YAML config, apply defaults, and validate it.

    Parameters
    ----------
    config_path
        YAML config path.
    schema_path
        JSON schema path.

    Returns
    -------
    dict[str, Any]
        Resolved and validated config dictionary.
    """
    raw_config = load_yaml_config(config_path)
    resolved_config = apply_config_defaults(raw_config)
    validate_config(resolved_config, schema_path=schema_path)
    LOGGER.info("Validated config: %s", config_path)
    return resolved_config
