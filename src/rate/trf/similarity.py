import numpy as np


def cross_kernel_temporal_similarity(
    kernel_a: np.ndarray,
    kernel_b: np.ndarray,
    t0_index: int,
) -> dict[str, np.ndarray]:
    """
    Compare temporal similarity of two TRF kernels using the lag-t0 spatial pattern of A.

    Parameters
    ----------
    kernel_a, kernel_b
        Arrays of shape (n_channels, n_lags) or (n_folds, n_channels, n_lags).
        If fold axis is present, folds are averaged first.
    t0_index
        Lag index used to define the reference spatial pattern in kernel_a.
    """
    a = np.asarray(kernel_a, dtype=float)
    b = np.asarray(kernel_b, dtype=float)
    if a.ndim == 3:
        a = np.nanmean(a, axis=0)
    if b.ndim == 3:
        b = np.nanmean(b, axis=0)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("kernel_a and kernel_b must be 2D or 3D with shape (channels, lags)")
    if a.shape != b.shape:
        raise ValueError(f"Kernel shape mismatch: {a.shape} vs {b.shape}")
    n_channels, n_lags = a.shape
    if not (0 <= int(t0_index) < n_lags):
        raise IndexError(f"t0_index={t0_index} out of bounds for n_lags={n_lags}")

    ref = a[:, int(t0_index)]

    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        x0 = np.asarray(x, dtype=float) - np.nanmean(x)
        y0 = np.asarray(y, dtype=float) - np.nanmean(y)
        denom = np.sqrt(np.nansum(x0**2) * np.nansum(y0**2))
        if not np.isfinite(denom) or denom <= 0:
            return float("nan")
        return float(np.nansum(x0 * y0) / denom)

    self_self = np.array([_corr(ref, a[:, lag]) for lag in range(n_lags)], dtype=float)
    self_other = np.array([_corr(ref, b[:, lag]) for lag in range(n_lags)], dtype=float)
    return {"self_self_corr": self_self, "self_other_corr": self_other}
