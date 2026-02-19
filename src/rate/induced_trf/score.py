import pandas as pd
import numpy as np


def compute_scores(
    model,
    X: np.ndarray,
    y: np.ndarray,
    metadata: dict,
) -> pd.DataFrame:

    r = float(model.score(X, y))
    baseline = 0.0

    return pd.DataFrame(
        [
            {
                **metadata,
                "r": r,
                "r_baseline": baseline,
                "delta_r": r - baseline,
            }
        ]
    )
