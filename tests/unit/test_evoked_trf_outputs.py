import pandas as pd
import pytest

from rate.evoked_trf.score import validate_events_table
from rate.trf import scoring as trf_scoring


def test_scores_schema_baseline_only_contract() -> None:
    # This is a lightweight contract test: no EEG required.
    scores = pd.DataFrame(
        [
            {
                "subject": "sub-001",
                "dyad_id": "dyad-01",
                "run": 1,
                "model": "baseline",
                "fold_id": 0,
                "r": 0.10,
                "r_baseline": 0.10,
                "delta_r": 0.00,
            },
        ]
    )

    required = {"subject", "dyad_id", "run", "model", "fold_id", "r"}
    assert required.issubset(scores.columns)

    baseline = scores["model"] == "baseline"
    assert (scores.loc[baseline, "delta_r"] == 0.0).all()


def test_cv_splits_schema() -> None:
    splits = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-01",
                "run": 1,
                "fold_id": 0,
                "bin_id": 0,
                "bin_t0": 0.0,
                "bin_t1": 10.0,
                "split": "test",
            }
        ]
    )
    required = {"dyad_id", "run", "fold_id", "bin_id", "bin_t0", "bin_t1", "split"}
    assert required.issubset(splits.columns)


def test_target_aware_summary_keeps_baseline_within_target() -> None:
    outer = pd.DataFrame(
        [
            {"dyad_id": "dyad-01", "subject": "sub-01", "target": "evoked", "model": "baseline", "fold_id": 0, "test_run": 1, "alpha": 1.0, "chosen_alpha": 1.0, "r": 0.1},
            {"dyad_id": "dyad-01", "subject": "sub-01", "target": "evoked", "model": "rate_effect", "fold_id": 0, "test_run": 1, "alpha": 1.0, "chosen_alpha": 1.0, "r": 0.3},
            {"dyad_id": "dyad-01", "subject": "sub-01", "target": "theta", "model": "baseline", "fold_id": 0, "test_run": 1, "alpha": 1.0, "chosen_alpha": 1.0, "r": 0.2},
            {"dyad_id": "dyad-01", "subject": "sub-01", "target": "theta", "model": "rate_effect", "fold_id": 0, "test_run": 1, "alpha": 1.0, "chosen_alpha": 1.0, "r": 0.5},
        ]
    )

    summary = trf_scoring.summarize_scores(dyad_id="dyad-01", subject="sub-01", outer_scores=outer)

    evoked = summary.loc[(summary["target"] == "evoked") & (summary["model"] == "rate_effect")].iloc[0]
    theta = summary.loc[(summary["target"] == "theta") & (summary["model"] == "rate_effect")].iloc[0]
    assert evoked["delta_r"] == pytest.approx(0.2)
    assert theta["delta_r"] == pytest.approx(0.3)


def test_validate_events_table_backfills_mean_rate_for_legacy_events() -> None:
    events = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-01",
                "subject_id": "sub-01",
                "run": 1,
                "t_onset": 0.0,
                "t_offset": 1.0,
                "self_rate": 4.0,
                "other_rate": 2.0,
            }
        ]
    )

    out = validate_events_table(events)

    assert "mean_rate" in out.columns
    assert out.loc[0, "mean_rate"] == pytest.approx(3.0)
