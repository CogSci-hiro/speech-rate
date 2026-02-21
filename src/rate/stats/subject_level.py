import numpy as np
from scipy import stats


def subject_t_to_signed_z(
    t_values: np.ndarray,
    df: int,
) -> np.ndarray:
    """
    Convert signed t-values to signed z-values using the CDF mapping.

    z = Φ^{-1}(F_t(t; df))

    Parameters
    ----------
    t_values
        Array of t-values.
    df
        Degrees of freedom for the t-distribution.

    Returns
    -------
    np.ndarray
        Array of z-values with same shape as t_values.

    Notes
    -----
    - This preserves sign.
    - Values are clipped to avoid inf from numerical issues.

    Usage example
    -------------
        z = subject_t_to_signed_z(t_values=t_map, df=7)
    """
    cdf = stats.t.cdf(t_values, df=df)
    cdf = np.clip(cdf, 1e-12, 1.0 - 1e-12)
    z = stats.norm.ppf(cdf)
    return z


def compute_subject_level_z_maps(
    feature_kernels: np.ndarray | list[np.ndarray],
) -> np.ndarray:
    """
    Compute subject-level signed z-maps from outer-fold kernels (two-level stage 1).

    Parameters
    ----------
    feature_kernels
        Either:
        - Array of shape (n_subjects, n_folds, n_channels, n_lags), or
        - List of per-subject arrays shaped (n_folds, n_channels, n_lags),
          allowing variable n_folds across subjects.

    Returns
    -------
    np.ndarray
        Subject-level z-maps of shape (n_subjects, n_channels, n_lags).

    Rationale
    ---------
    For each subject and each (channel, lag) bin, we test whether the mean
    kernel coefficient across outer folds differs from 0:

        t = mean(beta_fold) / (std(beta_fold) / sqrt(n_folds))

    Then map t -> signed z to standardize scale across subjects.

    Usage example
    -------------
        z_maps = compute_subject_level_z_maps(feature_kernels)
    """
    if isinstance(feature_kernels, np.ndarray):
        if feature_kernels.ndim != 4:
            raise ValueError(f"Expected ndim=4, got shape={feature_kernels.shape}")
        subject_arrays = [feature_kernels[i] for i in range(feature_kernels.shape[0])]
    else:
        subject_arrays = list(feature_kernels)

    if not subject_arrays:
        raise ValueError("No subject kernel arrays provided.")

    expected_ch_lag: tuple[int, int] | None = None
    z_maps: list[np.ndarray] = []
    for i, subj in enumerate(subject_arrays):
        if subj.ndim != 3:
            raise ValueError(f"Subject[{i}] expected ndim=3 (fold, ch, lag), got shape={subj.shape}")
        n_folds, n_channels, n_lags = subj.shape
        if n_folds < 2:
            raise ValueError(
                f"Subject[{i}] has n_folds={n_folds}. Need at least 2 folds for subject-level t-values."
            )
        ch_lag = (n_channels, n_lags)
        if expected_ch_lag is None:
            expected_ch_lag = ch_lag
        elif ch_lag != expected_ch_lag:
            raise ValueError(
                f"Inconsistent subject geometry at Subject[{i}]: (ch, lag)={ch_lag}, "
                f"expected={expected_ch_lag}"
            )

        mean_beta = subj.mean(axis=0)  # (ch, lag)
        std_beta = subj.std(axis=0, ddof=1)  # (ch, lag)
        se_beta = std_beta / np.sqrt(float(n_folds))
        se_beta = np.where(se_beta == 0.0, np.nan, se_beta)
        t_map = mean_beta / se_beta
        z_map = subject_t_to_signed_z(t_values=t_map, df=n_folds - 1)
        z_maps.append(np.nan_to_num(z_map, nan=0.0, posinf=0.0, neginf=0.0))

    return np.stack(z_maps, axis=0)
