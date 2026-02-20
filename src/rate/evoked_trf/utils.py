import numpy as np


def pearsonr_1d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation for 1D arrays. Returns NaN if variance is zero.

    Usage example
    -------------
        r = pearsonr_1d(y_true, y_pred)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")

    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def score_multichannel_mean_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Pearson r across channels.

    Parameters
    ----------
    y_true, y_pred
        Arrays shaped (n_samples, n_channels).

    Usage example
    -------------
        r = score_multichannel_mean_r(y_test, y_hat)
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    rs: list[float] = []
    for ch in range(y_true.shape[1]):
        rs.append(pearsonr_1d(y_true[:, ch], y_pred[:, ch]))
    return float(np.nanmean(rs))


def standardize_design_matrix_foldwise(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fold-wise z-scoring using TRAIN stats only (no leakage).

    Notes
    -----
    Constant features in TRAIN keep std=1.

    Usage example
    -------------
        Xtr, Xte = standardize_design_matrix_foldwise(X_train, X_test)
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std


def max_lag_samples(*, sfreq_hz: float, tmin_s: float, tmax_s: float) -> int:
    """
    Conservative max lag in samples for gap insertion.

    Usage example
    -------------
        gap = max_lag_samples(sfreq_hz=100.0, tmin_s=-0.2, tmax_s=0.8) + 1
    """
    return int(np.ceil(max(abs(tmin_s), abs(tmax_s)) * sfreq_hz))


def concatenate_with_gap(
    arrays: list[np.ndarray], *, gap_samples: int, fill_value: float = 0.0
) -> np.ndarray:
    """
    Concatenate with a gap segment inserted between arrays to prevent lag bleed.

    Usage example
    -------------
        y = concatenate_with_gap([y1, y2], gap_samples=81)
    """
    if not arrays:
        raise ValueError("arrays must be non-empty")
    if gap_samples < 0:
        raise ValueError("gap_samples must be >= 0")

    parts: list[np.ndarray] = []
    for i, a in enumerate(arrays):
        parts.append(np.asarray(a, dtype=float))
        if i < len(arrays) - 1 and gap_samples > 0:
            gap_shape = (gap_samples,) + parts[-1].shape[1:]
            parts.append(np.full(gap_shape, fill_value, dtype=float))
    return np.concatenate(parts, axis=0)
