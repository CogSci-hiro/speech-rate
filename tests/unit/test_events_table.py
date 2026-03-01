"""Unit tests for SG2 canonical IPU events table building."""

from pathlib import Path

import pandas as pd

from rate.events import build_events_table

REQUIRED_EVENT_COLUMNS = {
    "dyad_id",
    "subject_id",
    "run",
    "event_id",
    "t_onset",
    "t_offset",
    "speaker",
    "listener",
    "self_rate",
    "other_rate",
    "latency",
    "abs_diff",
    "self_rate_sq",
    "other_rate_sq",
    "abs_diff_sq",
    "time_within_run",
}


def _write_annotation(path: Path) -> None:
    """Write a tiny annotation table with valid and invalid IPUs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {"t_onset": 0.0, "t_offset": 0.7, "syllables": 4, "speaker": "A"},
            {"t_onset": 0.5, "t_offset": 1.5, "syllables": 6, "speaker": "B"},
            {"t_onset": 1.6, "t_offset": 1.8, "syllables": 4, "speaker": "A"},  # too short
            {"t_onset": 1.9, "t_offset": 2.4, "syllables": 1, "speaker": "B"},  # too few syllables
            {"t_onset": 2.6, "t_offset": 2.5, "syllables": 4, "speaker": "A"},  # invalid timing
        ]
    )
    df.to_csv(path, index=False)


def _base_config(tmp_path: Path) -> dict:
    """Return a minimal SG2 config payload."""
    return {
        "paths": {
            "out_dir": str(tmp_path / "outputs"),
            "annotations_dir": str(tmp_path / "annotations"),
        },
        "events": {
            "ipu_min_duration_s": 0.3,
            "ipu_min_syllables": 3,
            "time_within_run": {
                "normalize": True,
                "method": "run_duration",
            },
            "matching": {
                "absdiff_window_s": 1.0,
                "time_lock": "onset",
            },
        },
    }


def _dataset_index_df(annotation_path: Path) -> pd.DataFrame:
    """Build a tiny SG1-like dataset index frame."""
    return pd.DataFrame(
        [
            {
                "dyad_id": "dyad-01",
                "subject_id": None,
                "role": None,
                "task": "dialogue",
                "run": "1",
                "eeg_path": "/tmp/fake.fif",
                "audio_path": "/tmp/fake.wav",
                "annotations_path": str(annotation_path),
                "status": "ok",
                "reason": "",
            }
        ]
    )


def test_required_columns_exist(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    annotation_path = tmp_path / "annotations" / "dyad-01_task-dialogue_run-01.csv"
    _write_annotation(annotation_path)
    dataset_df = _dataset_index_df(annotation_path)

    events_df = build_events_table(config, dataset_df)

    assert REQUIRED_EVENT_COLUMNS.issubset(set(events_df.columns))


def test_timing_is_valid_and_filters_are_applied(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    annotation_path = tmp_path / "annotations" / "dyad-01_task-dialogue_run-01.csv"
    _write_annotation(annotation_path)
    dataset_df = _dataset_index_df(annotation_path)

    events_df = build_events_table(config, dataset_df)

    assert (events_df["t_onset"] < events_df["t_offset"]).all()
    assert (events_df["duration_s"] >= config["events"]["ipu_min_duration_s"]).all()
    assert (events_df["syllables"] >= config["events"]["ipu_min_syllables"]).all()
    assert len(events_df) == 2


def test_time_within_run_is_bounded(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    annotation_path = tmp_path / "annotations" / "dyad-01_task-dialogue_run-01.csv"
    _write_annotation(annotation_path)
    dataset_df = _dataset_index_df(annotation_path)

    events_df = build_events_table(config, dataset_df)

    assert ((events_df["time_within_run"] >= 0.0) & (events_df["time_within_run"] <= 1.0)).all()


def test_event_ids_are_deterministic(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    annotation_path = tmp_path / "annotations" / "dyad-01_task-dialogue_run-01.csv"
    _write_annotation(annotation_path)
    dataset_df = _dataset_index_df(annotation_path)

    first = build_events_table(config, dataset_df)
    second = build_events_table(config, dataset_df)

    assert len(first) == len(second)
    assert first["event_id"].tolist() == second["event_id"].tolist()


def test_other_rate_and_abs_diff_present_for_two_subject_streams(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    ann_a = tmp_path / "annotations" / "sub-001_task-dialogue_run-1_ipu.csv"
    ann_b = tmp_path / "annotations" / "sub-002_task-dialogue_run-1_ipu.csv"
    _write_annotation(ann_a)
    _write_annotation(ann_b)

    dataset_df = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-001",
                "role": "odd",
                "task": "dialogue",
                "run": "1",
                "eeg_path": "/tmp/fake_a.fif",
                "audio_path": "/tmp/fake_a.wav",
                "annotations_path": str(ann_a),
                "status": "ok",
                "reason": "",
            },
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-002",
                "role": "even",
                "task": "dialogue",
                "run": "1",
                "eeg_path": "/tmp/fake_b.fif",
                "audio_path": "/tmp/fake_b.wav",
                "annotations_path": str(ann_b),
                "status": "ok",
                "reason": "",
            },
        ]
    )

    events_df = build_events_table(config, dataset_df)

    assert not events_df.empty
    assert events_df["other_rate"].notna().all()
    assert events_df["latency"].notna().all()
    assert events_df["abs_diff"].notna().all()
    assert events_df["self_rate_sq"].notna().all()
    assert events_df["other_rate_sq"].notna().all()
    assert events_df["abs_diff_sq"].notna().all()


def test_other_rate_is_nan_when_no_partner_target_in_window(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config["events"]["matching"]["absdiff_window_s"] = 0.1

    ann_a = tmp_path / "annotations" / "sub-001_task-dialogue_run-1_ipu.csv"
    ann_b = tmp_path / "annotations" / "sub-002_task-dialogue_run-1_ipu.csv"
    ann_a.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"t_onset": 0.0, "t_offset": 1.0, "syllables": 4, "speaker": "A"},
            {"t_onset": 10.0, "t_offset": 11.0, "syllables": 4, "speaker": "A"},
        ]
    ).to_csv(ann_a, index=False)
    pd.DataFrame(
        [
            {"t_onset": 3.0, "t_offset": 4.0, "syllables": 5, "speaker": "B"},
            {"t_onset": 13.0, "t_offset": 14.0, "syllables": 5, "speaker": "B"},
        ]
    ).to_csv(ann_b, index=False)

    dataset_df = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-001",
                "role": "odd",
                "task": "dialogue",
                "run": "1",
                "eeg_path": "/tmp/fake_a.fif",
                "audio_path": "/tmp/fake_a.wav",
                "annotations_path": str(ann_a),
                "status": "ok",
                "reason": "",
            },
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-002",
                "role": "even",
                "task": "dialogue",
                "run": "1",
                "eeg_path": "/tmp/fake_b.fif",
                "audio_path": "/tmp/fake_b.wav",
                "annotations_path": str(ann_b),
                "status": "ok",
                "reason": "",
            },
        ]
    )

    events_df = build_events_table(config, dataset_df)

    assert not events_df.empty
    assert events_df["other_rate"].isna().all()
    assert events_df["latency"].isna().all()
    assert events_df["listener"].isna().all()
    assert events_df["abs_diff"].isna().all()
    assert events_df["self_rate_sq"].notna().all()
    assert events_df["other_rate_sq"].isna().all()
    assert events_df["abs_diff_sq"].isna().all()


def test_latency_is_self_onset_minus_matched_other_offset(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    ann_a = tmp_path / "annotations" / "sub-001_task-dialogue_run-1_ipu.csv"
    ann_b = tmp_path / "annotations" / "sub-002_task-dialogue_run-1_ipu.csv"
    ann_a.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"t_onset": 1.0, "t_offset": 2.0, "syllables": 4, "speaker": "A"}]).to_csv(ann_a, index=False)
    pd.DataFrame([{"t_onset": 0.8, "t_offset": 1.2, "syllables": 4, "speaker": "B"}]).to_csv(ann_b, index=False)

    dataset_df = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-001",
                "role": "odd",
                "task": "dialogue",
                "run": "1",
                "eeg_path": "/tmp/fake_a.fif",
                "audio_path": "/tmp/fake_a.wav",
                "annotations_path": str(ann_a),
                "status": "ok",
                "reason": "",
            },
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-002",
                "role": "even",
                "task": "dialogue",
                "run": "1",
                "eeg_path": "/tmp/fake_b.fif",
                "audio_path": "/tmp/fake_b.wav",
                "annotations_path": str(ann_b),
                "status": "ok",
                "reason": "",
            },
        ]
    )

    events_df = build_events_table(config, dataset_df)

    row_a = events_df.loc[events_df["speaker"] == "sub-001"].iloc[0]
    row_b = events_df.loc[events_df["speaker"] == "sub-002"].iloc[0]
    assert row_a["latency"] == 1.0 - 1.2
    assert pd.isna(row_b["latency"])


def test_matching_uses_other_offset_not_other_onset_for_onset_timelock(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config["events"]["matching"]["absdiff_window_s"] = 0.2
    ann_a = tmp_path / "annotations" / "sub-001_task-dialogue_run-1_ipu.csv"
    ann_b = tmp_path / "annotations" / "sub-002_task-dialogue_run-1_ipu.csv"
    ann_a.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"t_onset": 1.0, "t_offset": 1.6, "syllables": 4, "speaker": "A"}]).to_csv(ann_a, index=False)
    pd.DataFrame(
        [
            {"t_onset": 0.9, "t_offset": 2.0, "syllables": 6, "speaker": "B"},
            {"t_onset": 0.0, "t_offset": 1.05, "syllables": 4, "speaker": "B"},
        ]
    ).to_csv(ann_b, index=False)

    dataset_df = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-001",
                "role": "odd",
                "task": "dialogue",
                "run": "1",
                "eeg_path": "/tmp/fake_a.fif",
                "audio_path": "/tmp/fake_a.wav",
                "annotations_path": str(ann_a),
                "status": "ok",
                "reason": "",
            },
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-002",
                "role": "even",
                "task": "dialogue",
                "run": "1",
                "eeg_path": "/tmp/fake_b.fif",
                "audio_path": "/tmp/fake_b.wav",
                "annotations_path": str(ann_b),
                "status": "ok",
                "reason": "",
            },
        ]
    )

    events_df = build_events_table(config, dataset_df)
    row_a = events_df.loc[events_df["speaker"] == "sub-001"].iloc[0]

    assert row_a["listener"] == "sub-002"
    assert row_a["latency"] == 1.0 - 1.05
