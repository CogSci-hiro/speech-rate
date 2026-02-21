from pathlib import Path
import pandas as pd
import numpy as np


def write_emg_parquet(
    out_path: Path,
    times: np.ndarray,
    emg_env: np.ndarray,
) -> None:
    """
    Save EMG envelope to parquet file.

    Parameters
    ----------
    out_path : Path
        Output parquet path.
    times : ndarray
        Time vector (seconds).
    emg_env : ndarray
        EMG envelope time series.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "time": times,
            "emg_env": emg_env,
        }
    )

    df.to_parquet(out_path, index=False)


def write_subject_emg_parquet(
    out_path: Path,
    rows: pd.DataFrame,
) -> None:
    """
    Save subject-level EMG envelope table aggregated across runs.

    Required columns
    ----------------
    dyad_id, subject_id, run, time, emg_env
    """
    required = {"dyad_id", "subject_id", "run", "time", "emg_env"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"subject EMG table missing columns: {sorted(missing)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = rows[["dyad_id", "subject_id", "run", "time", "emg_env"]].copy()
    ordered = ordered.sort_values(["run", "time"], kind="mergesort").reset_index(drop=True)
    ordered.to_parquet(out_path, index=False)
