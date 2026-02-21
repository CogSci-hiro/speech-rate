import pandas as pd


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
