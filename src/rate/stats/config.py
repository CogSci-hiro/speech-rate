from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ClusterMethod(str, Enum):
    """Cluster inference method."""
    THRESHOLD = "threshold"
    TFCE = "tfce"


@dataclass(frozen=True)
class StatsConfig:
    """
    Configuration for group-level cluster permutation tests.

    Parameters
    ----------
    n_permutations
        Number of Monte Carlo permutations.
    seed
        Random seed for determinism.
    method
        Cluster method: "threshold" or "tfce".
    threshold_p
        Cluster-forming p-value for the thresholded method (converted to a
        two-sided t-threshold at runtime). Ignored for TFCE.
    tfce_start
        TFCE start parameter (MNE "threshold" dict). Typical: 0.0.
    tfce_step
        TFCE step parameter (MNE "threshold" dict). Typical: 0.2 (but scale
        depends on your statistic magnitude).
    tail
        Tail for the group-level one-sample test: 0=two-sided, 1=positive, -1=negative.
    montage_name
        MNE montage name used to derive EEG sensor adjacency, e.g. "biosemi64".

    Usage example
    -------------
        config = StatsConfig(
            n_permutations=4000,
            seed=42,
            method=ClusterMethod.TFCE,
            threshold_p=0.001,
            tfce_start=0.0,
            tfce_step=0.2,
            tail=0,
            montage_name="biosemi64",
        )
    """
    n_permutations: int = 4000
    seed: int = 42
    method: ClusterMethod = ClusterMethod.TFCE

    threshold_p: float = 0.001

    tfce_start: float = 0.0
    tfce_step: float = 0.2

    tail: int = 0
    montage_name: str = "biosemi64"

    # Optional: allow bypassing montage adjacency if you want to supply your own later.
    min_cluster_size: Optional[int] = None