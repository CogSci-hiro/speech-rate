"""Unit tests for config loading and run manifest scaffolding."""

import json
from pathlib import Path

import pytest
import yaml

from rate.config import ConfigValidationError, load_and_validate_config
from rate.manifest import write_run_manifest


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _minimal_config(out_dir: Path) -> dict:
    return {
        "paths": {
            "out_dir": str(out_dir),
            "ica_dir": str(out_dir / "ica"),
        },
        "dataset": {},
        "preprocessing": {},
        "epochs": {},
        "glm": {},
        "fooof": {},
        "lmeeeg": {},
    }


def test_schema_validates_known_good_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_config(tmp_path / "outputs"))

    resolved = load_and_validate_config(cfg_path, schema_path=Path("config/schema.json"))

    assert resolved["paths"]["out_dir"] == str(tmp_path / "outputs")


def test_repository_config_validates() -> None:
    resolved = load_and_validate_config(Path("config/config.yaml"), schema_path=Path("config/schema.json"))

    expected_logs = str(Path(resolved["paths"]["out_dir"]) / "logs")
    assert resolved["paths"]["logs_dir"] == expected_logs
    assert resolved["preprocessing"]["downsample"]["sfreq_hz"] == pytest.approx(200.0)
    assert resolved["epochs"]["metadata_source"] == "Epochs.metadata"


def test_schema_rejects_missing_required_key(tmp_path: Path) -> None:
    cfg_path = tmp_path / "invalid.yaml"
    payload = {
        "paths": {},
        "dataset": {},
        "preprocessing": {},
        "epochs": {},
        "glm": {},
        "fooof": {},
    }
    _write_yaml(cfg_path, payload)

    with pytest.raises(ConfigValidationError):
        load_and_validate_config(cfg_path, schema_path=Path("config/schema.json"))


def test_default_logs_dir_applied(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "outputs"
    _write_yaml(cfg_path, _minimal_config(out_dir))

    resolved = load_and_validate_config(cfg_path, schema_path=Path("config/schema.json"))

    assert resolved["paths"]["logs_dir"] == str(out_dir / "logs")


def test_modular_compute_files_attach_mass_univariate_sections(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "paths": {
                "out_dir": str(tmp_path / "outputs"),
                "ica_dir": str(tmp_path / "outputs" / "ica"),
            }
        },
    )
    _write_yaml(tmp_path / "dataset.yaml", {"dataset": {"tasks": ["conversation"], "runs": [1]}})
    _write_yaml(tmp_path / "preprocessing.yaml", {"preprocessing": {"downsample_hz": 128.0}})
    _write_yaml(tmp_path / "epochs.yaml", {"epochs": {"metadata_source": "Epochs.metadata"}})
    _write_yaml(tmp_path / "glm.yaml", {"glm": {"regressors": ["speech_rate"]}})
    _write_yaml(tmp_path / "fooof.yaml", {"fooof": {"window_s": 1.0}})
    _write_yaml(tmp_path / "lmeeeg.yaml", {"lmeeeg": {"enabled": False}})

    resolved = load_and_validate_config(cfg_path, schema_path=Path("config/schema.json"))

    assert resolved["preprocessing"]["downsample_hz"] == pytest.approx(128.0)
    assert resolved["glm"]["regressors"] == ["speech_rate"]


def test_manifest_written(tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    cfg = _minimal_config(out_dir)

    manifest_path = write_run_manifest(cfg)

    assert manifest_path == out_dir / "_meta" / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["resolved_config"]["paths"]["out_dir"] == str(out_dir)
