"""Unit tests for SG1 dataset index generation."""

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
            "exclude": {
                "subjects": [],
                "subject_runs": [],
                "dyads": [],
                "dyad_runs": [],
            },
        },
    }


def _make_present_triplet(cfg: dict, dyad_id: str, task: str, run: int) -> None:
    """Create matching audio/eeg/annotation files for one dataset unit."""
    run_label = f"run-{run}"
    stem = f"{dyad_id}_task-{task}_{run_label}"
    _touch(Path(cfg["paths"]["annotations_dir"]) / f"{stem}.csv")
    _touch(Path(cfg["paths"]["bids_dir"]) / "audio" / f"{stem}.wav")
    _touch(Path(cfg["paths"]["eeg_dir"]) / f"{stem}.fif")


def test_no_duplicate_dyad_run_task_rows(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    _make_present_triplet(cfg, dyad_id="dyad-001", task="chat", run=1)

    index_df = build_dataset_index(cfg)

    duplicate_mask = index_df.duplicated(subset=["dyad_id", "run", "task", "subject_id"], keep=False)
    assert not duplicate_mask.any()


def test_every_row_has_valid_status(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    _make_present_triplet(cfg, dyad_id="dyad-001", task="chat", run=1)

    index_df = build_dataset_index(cfg)

    assert set(index_df["status"]).issubset(VALID_STATUSES)


def test_config_exclusions_are_applied_with_precedence(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["dataset"]["runs"] = [1]
    _make_present_triplet(cfg, dyad_id="dyad-001", task="chat", run=1)
    cfg["dataset"]["missing_runs"] = [{"dyad_id": "dyad-001", "run": 1}]
    cfg["dataset"]["exclude"]["dyad_runs"] = [{"dyad_id": "dyad-001", "run": 1, "reason": "manual"}]

    index_df = build_dataset_index(cfg)
    row = index_df.iloc[0]

    assert row["status"] == "excluded"
    assert "manual" in str(row["reason"])


def test_missing_runs_are_explicit_rows(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    _make_present_triplet(cfg, dyad_id="dyad-001", task="chat", run=1)
    cfg["dataset"]["missing_runs"] = [{"dyad_id": "dyad-001", "run": 2}]

    index_df = build_dataset_index(cfg)
    run2_rows = index_df[index_df["run"] == "2"]

    assert not run2_rows.empty
    assert set(run2_rows["status"]) == {"missing"}
    assert any("missing_runs" in str(reason) for reason in run2_rows["reason"])


def test_writeable_parquet_index_shape(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    _make_present_triplet(cfg, dyad_id="dyad-001", task="chat", run=1)

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


def test_exclude_subject_marks_all_runs_for_dyad(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    _make_present_triplet(cfg, dyad_id="dyad-001", task="chat", run=1)
    _make_present_triplet(cfg, dyad_id="dyad-001", task="chat", run=2)
    cfg["dataset"]["exclude"]["subjects"] = [
        {"subject_id": "sub-001", "reason": "excessive_artifact"},
    ]

    index_df = build_dataset_index(cfg)

    excluded_rows = index_df[index_df["status"] == "excluded"]
    ok_rows = index_df[index_df["status"] == "ok"]

    assert not excluded_rows.empty
    assert all(excluded_rows["subject_id"] == "sub-001")
    assert all("exclude.subjects:sub-001:excessive_artifact" in str(reason) for reason in excluded_rows["reason"])
    assert set(ok_rows["subject_id"]) == {"sub-002"}


def test_exclude_subject_run_marks_only_target_run(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    _make_present_triplet(cfg, dyad_id="dyad-001", task="chat", run=1)
    _make_present_triplet(cfg, dyad_id="dyad-001", task="chat", run=2)
    cfg["dataset"]["exclude"]["subject_runs"] = [
        {"subject_id": "sub-001", "run": 2, "reason": "bad_audio_sync"},
    ]

    index_df = build_dataset_index(cfg)
    run1 = index_df[index_df["run"] == "1"]
    run2 = index_df[index_df["run"] == "2"]

    sub1_run1 = run1[run1["subject_id"] == "sub-001"].iloc[0]
    sub1_run2 = run2[run2["subject_id"] == "sub-001"].iloc[0]
    sub2_run2 = run2[run2["subject_id"] == "sub-002"].iloc[0]

    assert sub1_run1["status"] == "ok"
    assert sub1_run2["status"] == "excluded"
    assert "exclude.subject_runs:sub-001/run-2:bad_audio_sync" in str(sub1_run2["reason"])
    assert sub2_run2["status"] == "ok"


def test_exclude_dyad_marks_all_runs(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    _make_present_triplet(cfg, dyad_id="dyad-001", task="chat", run=1)
    _make_present_triplet(cfg, dyad_id="dyad-001", task="chat", run=2)
    cfg["dataset"]["exclude"]["dyads"] = [
        {"dyad_id": "dyad-001", "reason": "incomplete_recording"},
    ]

    index_df = build_dataset_index(cfg)

    assert set(index_df["status"]) == {"excluded"}
    assert all("exclude.dyads:dyad-001:incomplete_recording" in str(reason) for reason in index_df["reason"])


def test_exclude_dyad_run_marks_target_run(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    _make_present_triplet(cfg, dyad_id="dyad-001", task="chat", run=1)
    _make_present_triplet(cfg, dyad_id="dyad-001", task="chat", run=2)
    cfg["dataset"]["exclude"]["dyad_runs"] = [
        {"dyad_id": "dyad-001", "run": 2, "reason": "corrupted_eeg"},
    ]

    index_df = build_dataset_index(cfg)
    run1 = index_df[index_df["run"] == "1"].iloc[0]
    run2 = index_df[index_df["run"] == "2"].iloc[0]

    assert run1["status"] == "ok"
    assert run2["status"] == "excluded"
    assert "exclude.dyad_runs:dyad-001/run-2:corrupted_eeg" in str(run2["reason"])
