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
    feature_kernels: np.ndarray,
) -> np.ndarray:
    """
    Compute subject-level signed z-maps from outer-fold kernels (two-level stage 1).

    Parameters
    ----------
    feature_kernels
        Array of shape (n_subjects, n_folds, n_channels, n_lags)
        corresponding to one feature's kernels across outer folds.

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
    if feature_kernels.ndim != 4:
        raise ValueError(f"Expected ndim=4, got shape={feature_kernels.shape}")

    n_subjects, n_folds, n_channels, n_lags = feature_kernels.shape
    if n_folds < 2:
        raise ValueError("Need at least 2 outer folds to compute subject-level t-values.")

    # t-test against zero across folds for each subject/bin
    mean_beta = feature_kernels.mean(axis=1)  # (subj, ch, lag)
    std_beta = feature_kernels.std(axis=1, ddof=1)  # (subj, ch, lag)
    se_beta = std_beta / np.sqrt(float(n_folds))

    # Avoid division by zero (if std=0 across folds)
    se_beta = np.where(se_beta == 0.0, np.nan, se_beta)

    t_map = mean_beta / se_beta
    df = n_folds - 1
    z_map = subject_t_to_signed_z(t_values=t_map, df=df)

    # Replace NaNs (bins with zero variance across folds) with 0 contribution
    z_map = np.nan_to_num(z_map, nan=0.0, posinf=0.0, neginf=0.0)
    return z_map