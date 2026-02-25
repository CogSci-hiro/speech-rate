from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from mne.stats import permutation_cluster_1samp_test
from scipy import sparse
from scipy import stats as sp_stats

from .config import ClusterMethod, StatsConfig


@dataclass(frozen=True)
class ClusterPermutationResult:
    """
    Output of a cluster permutation test.

    Attributes
    ----------
    t_obs
        Observed test statistic at each bin (flattened).
    clusters
        Cluster definitions (index arrays).
    cluster_p_values
        P-values per cluster.
    h0
        Null distribution of max cluster statistics.
    """
    t_obs: np.ndarray
    clusters: object
    cluster_p_values: np.ndarray
    h0: np.ndarray


def _threshold_from_p(p: float, df: int, tail: int) -> float:
    """
    Convert a p-value to a t-threshold for cluster forming.

    For tail=0 (two-sided), uses p/2 and absolute threshold.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"threshold_p must be in (0,1), got {p}")

    if tail == 0:
        return float(sp_stats.t.ppf(1.0 - p / 2.0, df=df))
    if tail == 1:
        return float(sp_stats.t.ppf(1.0 - p, df=df))
    if tail == -1:
        return float(sp_stats.t.ppf(p, df=df))
    raise ValueError(f"Invalid tail={tail}. Use -1, 0, or 1.")


def run_cluster_1samp(
    data: np.ndarray,
    adjacency: sparse.csr_matrix,
    config: StatsConfig,
) -> ClusterPermutationResult:
    """
    One-sample cluster permutation test across subjects.

    Parameters
    ----------
    data
        Array of shape (n_subjects, n_tests), where n_tests = n_channels * n_lags.
        In our two-level approach this is typically the subject-level z-maps flattened.
    adjacency
        Spatiotemporal adjacency over the n_tests dimension.
    config
        StatsConfig controlling thresholding/TFCE and permutation settings.

    Returns
    -------
    ClusterPermutationResult

    Usage example
    -------------
        result = run_cluster_1samp(data=z_flat, adjacency=adj, config=config)
    """
    if data.ndim != 2:
        raise ValueError(f"Expected data ndim=2, got shape={data.shape}")

    n_subjects = data.shape[0]
    df = n_subjects - 1

    if config.method == ClusterMethod.TFCE:
        threshold = dict(start=config.tfce_start, step=config.tfce_step)
    else:
        threshold = _threshold_from_p(p=config.threshold_p, df=df, tail=config.tail)

    t_obs, clusters, cluster_p_values, h0 = permutation_cluster_1samp_test(
        data,
        threshold=threshold,
        adjacency=adjacency,
        tail=config.tail,
        n_permutations=config.n_permutations,
        seed=config.seed,
        out_type="indices",
        step_down_p=0.0,
        buffer_size=None,
        # MNE uses a tqdm-backed ProgressBar internally for permutations when verbose is enabled.
        verbose=True,
    )

    return ClusterPermutationResult(
        t_obs=t_obs,
        clusters=clusters,
        cluster_p_values=cluster_p_values,
        h0=h0,
    )
