from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .adjacency import make_spatiotemporal_adjacency
from .cluster import ClusterPermutationResult, run_cluster_1samp
from .config import StatsConfig
from .io import stack_subject_feature_kernels
from .subject_level import compute_subject_level_z_maps


@dataclass(frozen=True)
class TwoLevelClusterResult:
    """
    Two-level cluster result for kernel-vs-zero inference.

    Attributes
    ----------
    subject_z_maps
        Subject-level z maps, shape (n_subjects, n_channels, n_lags).
    group_cluster
        Cluster permutation result on flattened z-maps.
    ch_names
        Channel names in the same order as the kernel arrays.
    """
    subject_z_maps: np.ndarray
    group_cluster: ClusterPermutationResult
    ch_names: List[str]


def run_two_level_kernel0(
    out_dir: Path,
    analysis: str,
    subject_keys: Sequence[Tuple[str, str]],
    model: str,
    feature_index: int,
    ch_names: List[str],
    config: StatsConfig,
) -> TwoLevelClusterResult:
    """
    Two-level kernel-vs-zero cluster test (feature kernel != 0).

    Stage 1 (within-subject)
        For each subject and each (channel, lag), compute a t-stat across
        outer folds and map it to a signed z-score.

    Stage 2 (group)
        Perform a one-sample cluster permutation test across subjects on the
        subject-level z-maps.

    Parameters
    ----------
    out_dir
        Pipeline output directory.
    analysis
        "evoked_trf" or "induced_trf".
    subject_keys
        Sequence of (dyad_id, subject_id).
    model
        Model name, e.g. "self_rate".
    feature_index
        Index along the (n_features) axis of kernels_outer.npy.
    ch_names
        Channel names matching the kernel channel axis.
    config
        Statistical configuration (thresholded or TFCE).

    Returns
    -------
    TwoLevelClusterResult

    Usage example
    -------------
        result = run_two_level_kernel0(
            out_dir=Path("/Volumes/work-4T/speech-rate-testing"),
            analysis="evoked_trf",
            subject_keys=[("dyad-012", "sub-023"), ("dyad-012", "sub-024")],
            model="self_rate",
            feature_index=0,
            ch_names=biosemi64_names,
            config=StatsConfig(),
        )
    """
    # Load per-subject outer fold kernels for the feature
    feature_kernels = stack_subject_feature_kernels(
        out_dir=out_dir,
        analysis=analysis,
        subject_keys=subject_keys,
        model=model,
        feature_index=feature_index,
    )  # (subj, fold, ch, lag)

    # Stage 1: subject-level t across folds -> signed z
    subject_z_maps = compute_subject_level_z_maps(feature_kernels)  # (subj, ch, lag)

    n_subjects, n_channels, n_lags = subject_z_maps.shape
    if n_channels != len(ch_names):
        raise ValueError(
            f"Channel name mismatch: z_maps has n_channels={n_channels} but "
            f"len(ch_names)={len(ch_names)}"
        )

    # Stage 2: group-level cluster test on flattened maps
    adjacency = make_spatiotemporal_adjacency(
        ch_names=ch_names,
        n_lags=n_lags,
        montage_name=config.montage_name,
    )
    # `combine_adjacency(n_lags, ch_adj)` expects feature indexing with lag/time as
    # the first dimension and channel as the second.
    z_flat = np.transpose(subject_z_maps, (0, 2, 1)).reshape(n_subjects, n_lags * n_channels)

    group_cluster = run_cluster_1samp(data=z_flat, adjacency=adjacency, config=config)

    return TwoLevelClusterResult(
        subject_z_maps=subject_z_maps,
        group_cluster=group_cluster,
        ch_names=ch_names,
    )
