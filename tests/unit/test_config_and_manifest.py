"""Unit tests for SG0 config loading and run manifest scaffolding."""

import json
from pathlib import Path

import pytest
import yaml

from rate.config import ConfigValidationError, load_and_validate_config
from rate.manifest import write_run_manifest


def _write_yaml(path: Path, payload: dict) -> None:
    """Write a dictionary to YAML on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _minimal_config(out_dir: Path) -> dict:
    """Build a minimal valid config payload."""
    return {
        "paths": {
            "out_dir": str(out_dir),
        },
        "dataset": {},
        "events": {},
        "evoked_trf": {},
        "induced_trf": {},
        "reporting": {},
    }


def test_schema_validates_known_good_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_payload = _minimal_config(out_dir=tmp_path / "outputs")
    _write_yaml(cfg_path, cfg_payload)

    resolved = load_and_validate_config(cfg_path, schema_path=Path("config/schema.json"))

    assert resolved["paths"]["out_dir"] == str(tmp_path / "outputs")


def test_repository_config_validates() -> None:
    resolved = load_and_validate_config(
        Path("config/config.yaml"),
        schema_path=Path("config/schema.json"),
    )

    expected_logs = str(Path(resolved["paths"]["out_dir"]) / "logs")
    assert resolved["paths"]["logs_dir"] == expected_logs


def test_schema_rejects_missing_required_key(tmp_path: Path) -> None:
    cfg_path = tmp_path / "invalid.yaml"
    cfg_payload = {
        "paths": {},
        "dataset": {},
        "events": {},
        "evoked_trf": {},
        "induced_trf": {},
        "reporting": {},
    }
    _write_yaml(cfg_path, cfg_payload)

    with pytest.raises(ConfigValidationError):
        load_and_validate_config(cfg_path, schema_path=Path("config/schema.json"))


def test_default_logs_dir_applied(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "outputs"
    cfg_payload = _minimal_config(out_dir=out_dir)
    _write_yaml(cfg_path, cfg_payload)

    resolved = load_and_validate_config(cfg_path, schema_path=Path("config/schema.json"))

    assert resolved["paths"]["logs_dir"] == str(out_dir / "logs")


def test_manifest_written(tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    cfg = _minimal_config(out_dir=out_dir)

    manifest_path = write_run_manifest(cfg)

    assert manifest_path == out_dir / "_meta" / "run_manifest.json"
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "timestamp" in payload
    assert payload["resolved_config"]["paths"]["out_dir"] == str(out_dir)
    assert payload["random_seed"] is not None
