import numpy as np
from mne.decoding import ReceptiveField


def fit_trf(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    tmin: float,
    tmax: float,
):
    """
    X shape: (n_times, n_features)
    y shape: (n_times,)
    """
    model = ReceptiveField(
        tmin=tmin,
        tmax=tmax,
        sfreq=sfreq,
        scoring="corrcoef",
    )
    model.fit(X, y)
    return model
