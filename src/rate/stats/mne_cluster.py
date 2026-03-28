"""Wrappers around MNE cluster/permutation routines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ClusterPermutationResult:
    """Group-level one-sample cluster permutation outputs."""

    observed_statistic: np.ndarray
    corrected_p_values: np.ndarray
    null_distribution: np.ndarray
    cluster_p_values: np.ndarray | None
    backend_metadata: dict[str, Any]


def run_cluster_permutation(
    beta_maps: np.ndarray,
    *,
    correction: str,
    n_permutations: int,
    tail: int,
    threshold: float | Mapping[str, float] | None,
    adjacency,
    seed: int,
) -> ClusterPermutationResult:
    """Run a spatiotemporal one-sample cluster test against zero.

    Parameters
    ----------
    beta_maps
        Array with shape ``(n_subjects, n_channels, n_times)``.
    correction
        One of ``"cluster"``, ``"tfce"``, or ``"maxstat"``.
    """
    maps = np.asarray(beta_maps, dtype=float)
    if maps.ndim != 3:
        raise ValueError("beta_maps must have shape (n_subjects, n_channels, n_times).")
    if maps.shape[0] < 2:
        raise ValueError("At least two subject beta maps are required for one-sample group statistics.")
    if np.isnan(maps).any():
        raise ValueError("beta_maps must be finite before running cluster permutation inference.")

    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

    from mne.stats import spatio_temporal_cluster_1samp_test

    # MNE expects observations x time x space for spatiotemporal tests.
    spatiotemporal_maps = np.transpose(maps, (0, 2, 1))
    observed, clusters, cluster_p_values, null_distribution = spatio_temporal_cluster_1samp_test(
        spatiotemporal_maps,
        threshold=None if correction == "maxstat" else threshold,
        n_permutations=int(n_permutations),
        tail=int(tail),
        adjacency=adjacency,
        seed=int(seed),
        out_type="indices",
        verbose=False,
    )
    observed = np.asarray(observed, dtype=float)
    cluster_p_array = np.asarray(cluster_p_values, dtype=float)
    corrected_p_values = _cluster_p_values_to_map(clusters, cluster_p_array, observed.shape)

    return ClusterPermutationResult(
        observed_statistic=observed.T,
        corrected_p_values=corrected_p_values.T,
        null_distribution=np.asarray(null_distribution, dtype=float),
        cluster_p_values=cluster_p_array,
        backend_metadata={
            "engine": "mne.stats.spatio_temporal_cluster_1samp_test",
            "correction": str(correction),
            "n_clusters": int(len(clusters)),
            "threshold": _normalize_threshold(threshold),
            "tail": int(tail),
            "n_permutations": int(n_permutations),
            "adjacency_used": adjacency is not None,
        },
    )


def _cluster_p_values_to_map(
    clusters: list[Any],
    cluster_p_values: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    corrected = np.ones(shape, dtype=float)
    if len(clusters) == 0 or cluster_p_values.size == 0:
        return corrected

    for cluster, cluster_p_value in zip(clusters, cluster_p_values, strict=False):
        cluster_index = _cluster_to_index(cluster)
        corrected[cluster_index] = np.minimum(corrected[cluster_index], float(cluster_p_value))
    return corrected


def _cluster_to_index(cluster: Any):
    if isinstance(cluster, tuple):
        return tuple(np.asarray(index, dtype=int) for index in cluster)
    return np.asarray(cluster, dtype=bool)


def _normalize_threshold(threshold: float | Mapping[str, float] | None) -> float | dict[str, float] | None:
    if threshold is None:
        return None
    if isinstance(threshold, Mapping):
        return {str(key): float(value) for key, value in threshold.items()}
    return float(threshold)
