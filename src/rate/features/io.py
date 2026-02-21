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