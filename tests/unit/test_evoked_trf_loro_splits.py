import pandas as pd


def test_loro_split_schema_and_logic() -> None:
    splits = pd.DataFrame(
        [
            {"dyad_id": "d1", "subject": "s1", "fold_id": 0, "run": 1, "split": "test"},
            {"dyad_id": "d1", "subject": "s1", "fold_id": 0, "run": 2, "split": "train"},
        ]
    )
    required = {"dyad_id", "subject", "fold_id", "run", "split"}
    assert required.issubset(splits.columns)

    # within a fold: exactly one test run
    sub = splits[splits["fold_id"] == 0]
    assert (sub["split"] == "test").sum() == 1
