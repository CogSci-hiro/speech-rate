"""Thin wrappers around group-level statistics utilities."""

from rate.stats.mne_cluster import ClusterPermutationResult, run_cluster_permutation

__all__ = ["ClusterPermutationResult", "run_cluster_permutation"]
