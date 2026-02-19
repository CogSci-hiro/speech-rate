"""Dataset index validation rules."""

import pandas as pd

VALID_STATUSES = {"ok", "missing", "excluded"}

REQUIRED_INDEX_COLUMNS = [
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


def validate_dataset_index(index_df: pd.DataFrame) -> None:
    """Validate canonical dataset index invariants."""
    missing_cols = [col for col in REQUIRED_INDEX_COLUMNS if col not in index_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required dataset index columns: {missing_cols}")

    duplicate_mask = index_df.duplicated(subset=["dyad_id", "run", "task", "subject_id"], keep=False)
    if duplicate_mask.any():
        duplicates = index_df.loc[duplicate_mask, ["dyad_id", "run", "task", "subject_id"]].drop_duplicates()
        duplicate_records = duplicates.to_dict(orient="records")
        raise ValueError(f"Duplicate rows detected for dyad/run/task/subject keys: {duplicate_records}")

    bad_statuses = sorted(set(index_df["status"]) - VALID_STATUSES)
    if bad_statuses:
        raise ValueError(f"Invalid status values generated: {bad_statuses}")
