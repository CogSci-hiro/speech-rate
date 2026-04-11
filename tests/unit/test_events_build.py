"""Unit tests for canonical events building and questionnaire joins."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import pandas as pd
import pytest

from rate.events import EventsTableError, build_events_table
from rate.events.questionnaires import QuestionnaireError, build_subject_questionnaire_table


def _write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _write_annotation(path: Path, text: str) -> None:
    _write_csv(path, text)


def _config(out_dir: Path, questionnaire_dir: Path) -> dict:
    return {
        "paths": {
            "out_dir": str(out_dir),
            "questionnaire_dir": str(questionnaire_dir),
        },
        "dataset": {
            "questionnaires": {
                "likeability_filename": "likeability.csv",
                "personality_filename": "personality.csv",
            }
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


def test_build_subject_questionnaire_table_normalizes_ids(tmp_path: Path) -> None:
    questionnaire_dir = tmp_path / "questionnaires"
    _write_csv(
        questionnaire_dir / "likeability.csv",
        """
        subject_id,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,Score
        001,1,1,1,1,1,1,2,2,4,1,15
        002,1,1,1,1,1,1,2,2,4,1,10
        """,
    )
    _write_csv(
        questionnaire_dir / "personality.csv",
        """
        subject_id,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,extraversion,agreeableness,conscienciousness,neuroticism,openness
        001,3,2,4,2,3,2,3,2,4,1,4,4,7,7,3
        002,3,2,4,2,3,2,3,2,4,1,5,6,7,2,8
        """,
    )

    table = build_subject_questionnaire_table(_config(tmp_path / "out", questionnaire_dir))

    assert list(table["subject_id"]) == ["sub-001", "sub-002"]
    assert list(table.columns) == [
        "subject_id",
        "likeability",
        "extraversion",
        "agreeableness",
        "conscienciousness",
        "neuroticism",
        "openness",
    ]


def test_build_subject_questionnaire_table_rejects_duplicate_subjects(tmp_path: Path) -> None:
    questionnaire_dir = tmp_path / "questionnaires"
    _write_csv(questionnaire_dir / "likeability.csv", "subject_id,Score\n001,15\nsub-001,16")
    _write_csv(
        questionnaire_dir / "personality.csv",
        "subject_id,extraversion,agreeableness,conscienciousness,neuroticism,openness\n001,4,4,7,7,3",
    )

    with pytest.raises(QuestionnaireError, match="Duplicate questionnaire subject_id rows"):
        build_subject_questionnaire_table(_config(tmp_path / "out", questionnaire_dir))


def test_build_events_table_restores_proximity_columns_and_appends_questionnaires(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    out_dir = tmp_path / "out"
    questionnaire_dir = tmp_path / "questionnaires"
    config = _config(out_dir, questionnaire_dir)

    ann_a = tmp_path / "annotations" / "sub-001_task-dialogue_run-1_ipu.csv"
    ann_b = tmp_path / "annotations" / "sub-002_task-dialogue_run-1_ipu.csv"
    _write_annotation(
        ann_a,
        """
        t_onset,t_offset,syllables,speaker
        0.0,0.7,4,A
        1.0,1.6,5,A
        """,
    )
    _write_annotation(
        ann_b,
        """
        t_onset,t_offset,syllables,speaker
        0.3,1.1,6,B
        1.4,2.0,4,B
        """,
    )

    dataset_index_df = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-001",
                "role": "odd",
                "task": "dialogue",
                "run": "1",
                "annotations_path": str(ann_a),
                "status": "ok",
            },
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-002",
                "role": "even",
                "task": "dialogue",
                "run": "1",
                "annotations_path": str(ann_b),
                "status": "ok",
            },
        ]
    )

    _write_csv(questionnaire_dir / "likeability.csv", "subject_id,Score\n001,15\n003,12")
    _write_csv(
        questionnaire_dir / "personality.csv",
        """
        subject_id,extraversion,agreeableness,conscienciousness,neuroticism,openness
        001,4,4,7,7,3
        003,5,6,7,2,8
        """,
    )

    caplog.set_level("WARNING")
    events_df, summary = build_events_table(config, dataset_index_df=dataset_index_df)

    required_columns = {
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
        "proximity",
        "convergence",
        "mean_rate",
        "self_rate_sq",
        "other_rate_sq",
        "proximity_sq",
        "convergence_sq",
        "time_within_run",
        "task",
        "duration_s",
        "syllables",
        "self_n_syllables",
        "other_n_syllables",
        "self_ipu_duration",
        "other_ipu_duration",
        "annotations_path",
    }
    assert required_columns.issubset(set(events_df.columns))
    assert {"likeability", "extraversion", "agreeableness", "conscienciousness", "neuroticism", "openness"}.issubset(
        set(events_df.columns)
    )
    assert len(events_df) == 4
    assert events_df["event_id"].str.startswith("dyad-001_1_dialogue_ev").all()
    assert events_df["self_rate"].notna().all()
    assert events_df["self_n_syllables"].eq(events_df["syllables"]).all()
    assert events_df["self_ipu_duration"].eq(events_df["duration_s"]).all()
    assert events_df["other_n_syllables"].notna().sum() >= 3
    assert events_df["other_ipu_duration"].notna().sum() >= 3
    assert events_df["time_within_run"].between(0.0, 1.0).all()
    assert events_df.loc[events_df["subject_id"] == "sub-001", "likeability"].eq(15).all()
    assert events_df.loc[events_df["subject_id"] == "sub-002", "likeability"].isna().all()
    assert summary["events_without_questionnaires"] == ["sub-002"]
    assert summary["questionnaires_without_events"] == ["sub-003"]
    assert "Questionnaire data missing for 1 event subjects" in caplog.text
    assert "Questionnaire rows found for 1 subjects not present in events" in caplog.text


def test_build_events_table_requires_annotation_columns(tmp_path: Path) -> None:
    config = _config(tmp_path / "out", tmp_path / "questionnaires")
    ann = tmp_path / "annotations" / "sub-001_task-dialogue_run-1_ipu.csv"
    _write_annotation(ann, "t_onset,t_offset\n0.0,0.7")
    _write_csv(tmp_path / "questionnaires" / "likeability.csv", "subject_id,Score\n001,15")
    _write_csv(
        tmp_path / "questionnaires" / "personality.csv",
        "subject_id,extraversion,agreeableness,conscienciousness,neuroticism,openness\n001,4,4,7,7,3",
    )

    dataset_index_df = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-001",
                "role": "odd",
                "task": "dialogue",
                "run": "1",
                "annotations_path": str(ann),
                "status": "ok",
            }
        ]
    )

    with pytest.raises(EventsTableError, match="missing required columns: syllables"):
        build_events_table(config, dataset_index_df=dataset_index_df)


def test_build_events_table_shifts_annotation_times_by_raw_anchor_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "out"
    questionnaire_dir = tmp_path / "questionnaires"
    config = _config(out_dir, questionnaire_dir)

    ann = tmp_path / "annotations" / "sub-001_task-dialogue_run-1_ipu.csv"
    _write_annotation(
        ann,
        """
        t_onset,t_offset,syllables,speaker
        0.0,0.7,4,A
        1.0,1.6,5,A
        """,
    )
    _write_csv(questionnaire_dir / "likeability.csv", "subject_id,Score\n001,15")
    _write_csv(
        questionnaire_dir / "personality.csv",
        "subject_id,extraversion,agreeableness,conscienciousness,neuroticism,openness\n001,4,4,7,7,3",
    )
    raw_path = tmp_path / "sub-001_raw.fif"
    raw_path.write_text("", encoding="utf-8")

    dataset_index_df = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-001",
                "subject_id": "sub-001",
                "role": "odd",
                "task": "dialogue",
                "run": "1",
                "annotations_path": str(ann),
                "raw_eeg_path": str(raw_path),
                "status": "ok",
            }
        ]
    )

    monkeypatch.setattr("rate.events.build.find_run_anchor_event_from_file", lambda raw_path, config: (500, 2.5))

    events_df, _ = build_events_table(config, dataset_index_df=dataset_index_df)

    assert events_df["t_onset"].tolist() == pytest.approx([2.5, 3.5])
    assert events_df["t_offset"].tolist() == pytest.approx([3.2, 4.1])
