import numpy as np
import pandas as pd


def align_to_events(
    feature_times: np.ndarray,
    feature_values: np.ndarray,
    events: pd.DataFrame,
    event_time_col: str,
) -> pd.DataFrame:
    """
    Align continuous feature to event times via nearest sample.
    """
    aligned = []

    for _, row in events.iterrows():
        t = row[event_time_col]
        idx = np.argmin(np.abs(feature_times - t))

        aligned.append(
            {
                **row.to_dict(),
                "feature_value": float(feature_values[idx]),
            }
        )

    return pd.DataFrame(aligned)
