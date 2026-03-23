"""Configuration loading, defaulting, and schema validation."""

import copy
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

LOGGER = logging.getLogger(__name__)

DEFAULT_LOGS_SUBDIR = "logs"
DEFAULT_IPU_MIN_DURATION_S = 0.30
DEFAULT_IPU_MIN_SYLLABLES = 3
DEFAULT_TIME_WITHIN_RUN_NORMALIZE = True
DEFAULT_TIME_WITHIN_RUN_METHOD = "run_duration"
DEFAULT_MATCHING_ABSDIFF_WINDOW_S = 1.0
DEFAULT_MATCHING_TIME_LOCK = "onset"
DEFAULT_DEBUG_ENABLED = False
DEFAULT_DEBUG_MAX_FILES: int | None = None
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "config" / "schema.json"
MODULAR_COMPUTE_FILES = {
    "emg": "emg.yaml",
    "evoked_trf": "evoked_trf.yaml",
    "induced_trf": "induced_trf.yaml",
    "fooof": "fooof.yaml",
}


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


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _maybe_attach_modular_compute_sections(
    raw_config: Mapping[str, Any],
    *,
    config_path: Path | str,
) -> dict[str, Any]:
    """Attach modular compute sections when loading the base `config/config.yaml`.

    This keeps CLI commands schema-valid after the monolithic `compute.yaml` is retired.
    Only compute-layer sections are attached here; stats/viz remain separate.
    """
    path = Path(config_path)
    out: dict[str, Any] = dict(raw_config)
    if path.name != "config.yaml":
        return out

    parent = path.parent
    for section_key, filename in MODULAR_COMPUTE_FILES.items():
        modular_path = parent / filename
        if not modular_path.exists():
            continue
        modular_payload = load_yaml_config(modular_path)
        sibling_defaults = {key: value for key, value in modular_payload.items() if key != section_key}
        if sibling_defaults:
            # Let modular files contribute shared top-level defaults, while
            # keeping any values already defined in config.yaml authoritative.
            out = _deep_merge(sibling_defaults, out)
        if section_key in out:
            continue
        if section_key in modular_payload:
            out[section_key] = modular_payload[section_key]
    return out


def _require_section_downsample_hz(cfg: Mapping[str, Any], *, section_name: str) -> float:
    section = cfg.get(str(section_name), {})
    if isinstance(section, Mapping) and "downsample_hz" in section:
        return float(section["downsample_hz"])
    raise ConfigValidationError(f"Missing {section_name}.downsample_hz.")


def get_evoked_downsample_hz(cfg: Mapping[str, Any]) -> float:
    """Return the evoked-TRF sampling/downsample frequency in Hz."""
    return _require_section_downsample_hz(cfg, section_name="evoked_trf")


def get_induced_downsample_hz(cfg: Mapping[str, Any]) -> float:
    """Return the induced-TRF sampling/downsample frequency in Hz."""
    return _require_section_downsample_hz(cfg, section_name="induced_trf")


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

    events_value = resolved.setdefault("events", {})
    if not isinstance(events_value, dict):
        raise ConfigValidationError("`events` must be a mapping.")

    events_value.setdefault("ipu_min_duration_s", DEFAULT_IPU_MIN_DURATION_S)
    events_value.setdefault("ipu_min_syllables", DEFAULT_IPU_MIN_SYLLABLES)

    time_value = events_value.setdefault("time_within_run", {})
    if not isinstance(time_value, dict):
        raise ConfigValidationError("`events.time_within_run` must be a mapping.")
    time_value.setdefault("normalize", DEFAULT_TIME_WITHIN_RUN_NORMALIZE)
    time_value.setdefault("method", DEFAULT_TIME_WITHIN_RUN_METHOD)

    matching_value = events_value.setdefault("matching", {})
    if not isinstance(matching_value, dict):
        raise ConfigValidationError("`events.matching` must be a mapping.")
    matching_value.setdefault("absdiff_window_s", DEFAULT_MATCHING_ABSDIFF_WINDOW_S)
    matching_value.setdefault("time_lock", DEFAULT_MATCHING_TIME_LOCK)

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

    Returns
    -------
    None
        Returns when validation succeeds.
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
    raw_config = _maybe_attach_modular_compute_sections(raw_config, config_path=config_path)
    resolved_config = apply_config_defaults(raw_config)
    validate_config(resolved_config, schema_path=schema_path)
    LOGGER.info("Validated config: %s", config_path)
    return resolved_config
