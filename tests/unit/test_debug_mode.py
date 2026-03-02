from pathlib import Path

import pandas as pd
import pytest
import yaml

from rate.config import ConfigValidationError, load_and_validate_config
from rate.debug import get_debug_max_files, limit_dataframe_by_debug_max_files


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _minimal_config(out_dir: Path) -> dict:
    return {
        "paths": {"out_dir": str(out_dir)},
        "dataset": {},
        "events": {},
        "evoked_trf": {},
        "induced_trf": {},
        "reporting": {},
    }


def test_debug_max_files_from_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    payload = _minimal_config(tmp_path / "out")
    payload["debug"] = {"enabled": True, "max_files": 2}
    _write_yaml(cfg_path, payload)

    resolved = load_and_validate_config(cfg_path, schema_path=Path("config/schema.json"))

    assert get_debug_max_files(resolved) == 2


def test_schema_rejects_invalid_debug_max_files(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    payload = _minimal_config(tmp_path / "out")
    payload["debug"] = {"enabled": True, "max_files": 0}
    _write_yaml(cfg_path, payload)

    with pytest.raises(ConfigValidationError):
        load_and_validate_config(cfg_path, schema_path=Path("config/schema.json"))


def test_limit_dataframe_by_debug_max_files_uses_unique_units() -> None:
    df = pd.DataFrame(
        [
            {"dyad_id": "dyad-001", "subject_id": "sub-001", "run": "1", "value": 1},
            {"dyad_id": "dyad-001", "subject_id": "sub-001", "run": "1", "value": 2},
            {"dyad_id": "dyad-001", "subject_id": "sub-002", "run": "1", "value": 3},
            {"dyad_id": "dyad-001", "subject_id": "sub-002", "run": "1", "value": 4},
            {"dyad_id": "dyad-001", "subject_id": "sub-001", "run": "2", "value": 5},
        ]
    )
    cfg = {"debug": {"enabled": True, "max_files": 2}}

    limited = limit_dataframe_by_debug_max_files(df, cfg)

    assert len(limited) == 4
    kept_units = list(limited[["dyad_id", "subject_id", "run"]].drop_duplicates().itertuples(index=False, name=None))
    assert kept_units == [("dyad-001", "sub-001", "1"), ("dyad-001", "sub-002", "1")]


def test_limit_dataframe_by_debug_max_files_prioritizes_ok_status() -> None:
    df = pd.DataFrame(
        [
            {"dyad_id": "dyad-001", "subject_id": "sub-001", "run": "1", "status": "excluded"},
            {"dyad_id": "dyad-001", "subject_id": "sub-002", "run": "1", "status": "missing"},
            {"dyad_id": "dyad-002", "subject_id": "sub-003", "run": "1", "status": "ok"},
            {"dyad_id": "dyad-002", "subject_id": "sub-004", "run": "1", "status": "ok"},
        ]
    )
    cfg = {"debug": {"enabled": True, "max_files": 2}}

    limited = limit_dataframe_by_debug_max_files(df, cfg)

    kept_units = set(limited[["dyad_id", "subject_id", "run"]].itertuples(index=False, name=None))
    assert ("dyad-002", "sub-003", "1") in kept_units
    assert ("dyad-002", "sub-004", "1") in kept_units
