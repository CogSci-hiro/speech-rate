"""Unit tests for SG1 dataset index generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rate.dataset import build_dataset_index

VALID_STATUSES = {"ok", "missing", "excluded"}


def _touch(path: Path) -> None:
    """Create an empty file at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _base_config(tmp_path: Path) -> dict:
    """Build a baseline config dict for dataset index tests."""
    in_root = tmp_path / "inputs"
    return {
        "paths": {
            "out_dir": str(tmp_path / "outputs"),
            "bids_dir": str(in_root / "bids"),
            "eeg_dir": str(in_root / "eeg"),
            "annotations_dir": str(in_root / "annotations"),
        },
        "dataset": {
            "tasks": ["chat"],
            "runs": [1, 2],
            "missing_runs": [],
            "exclude_runs": [],
        },
    }


def _make_present_triplet(cfg: dict, dyad_id: str, task: str, run: int) -> None:
    """Create matching audio/eeg/annotation files for one dataset unit."""
    run_label = f"run-{run:02d}"
    stem = f"{dyad_id}_task-{task}_{run_label}"
    _touch(Path(cfg["paths"]["annotations_dir"]) / f"{stem}.csv")
    _touch(Path(cfg["paths"]["bids_dir"]) / "audio" / f"{stem}.wav")
    _touch(Path(cfg["paths"]["eeg_dir"]) / f"{stem}.fif")


def test_no_duplicate_dyad_run_task_rows(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    _make_present_triplet(cfg, dyad_id="dyad-01", task="chat", run=1)

    index_df = build_dataset_index(cfg)

    duplicate_mask = index_df.duplicated(subset=["dyad_id", "run", "task"], keep=False)
    assert not duplicate_mask.any()


def test_every_row_has_valid_status(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    _make_present_triplet(cfg, dyad_id="dyad-01", task="chat", run=1)

    index_df = build_dataset_index(cfg)

    assert set(index_df["status"]).issubset(VALID_STATUSES)


def test_config_exclusions_are_applied_with_precedence(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["dataset"]["runs"] = [1]
    _make_present_triplet(cfg, dyad_id="dyad-01", task="chat", run=1)
    cfg["dataset"]["missing_runs"] = [{"dyad_id": "dyad-01", "run": 1}]
    cfg["dataset"]["exclude_runs"] = [{"dyad_id": "dyad-01", "run": 1, "reason": "manual"}]

    index_df = build_dataset_index(cfg)
    row = index_df.iloc[0]

    assert row["status"] == "excluded"
    assert row["reason"] == "manual"


def test_missing_runs_are_explicit_rows(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    _make_present_triplet(cfg, dyad_id="dyad-01", task="chat", run=1)
    cfg["dataset"]["missing_runs"] = [{"dyad_id": "dyad-01", "run": 2}]

    index_df = build_dataset_index(cfg)
    run2_rows = index_df[index_df["run"] == "02"]

    assert not run2_rows.empty
    assert set(run2_rows["status"]) == {"missing"}
    assert any("missing_runs" in str(reason) for reason in run2_rows["reason"])


def test_writeable_parquet_index_shape(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    _make_present_triplet(cfg, dyad_id="dyad-01", task="chat", run=1)

    index_df = build_dataset_index(cfg)

    required_columns = [
        "dyad_id",
        "subject_id",
        "role",
        "task",
        "run",
        "eeg_path",
        "audio_path",
        "annotations_path",
        "status",
        "reason",
    ]
    assert isinstance(index_df, pd.DataFrame)
    assert list(index_df.columns) == required_columns
