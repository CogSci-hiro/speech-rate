from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .two_level import TwoLevelClusterResult


def write_two_level_cluster_result(
    result: TwoLevelClusterResult,
    out_dir: Path,
    analysis: str,
    model: str,
    feature_name: str,
    feature_index: int,
) -> Path:
    """
    Write cluster results to disk.

    Outputs
    -------
    - clusters.parquet
    - t_obs.npy
    - h0.npy
    - subject_z_maps.npy

    Returns
    -------
    Path
        Directory written to.

    Usage example
    -------------
        out_path = write_two_level_cluster_result(
            result,
            out_dir,
            analysis="evoked_trf",
            model="self_rate",
            feature_name="self_rate",
            feature_index=0,
        )
    """
    out_path = (
        out_dir
        / "final"
        / "kernel_stats"
        / analysis
        / model
        / feature_name
        / f"feature-{feature_index}"
    )
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / "t_obs.npy", result.group_cluster.t_obs)
    np.save(out_path / "h0.npy", result.group_cluster.h0)
    np.save(out_path / "subject_z_maps.npy", result.subject_z_maps)

    # Flatten cluster index sets into rows
    rows = []
    clusters = result.group_cluster.clusters
    pvals = result.group_cluster.cluster_p_values

    for i, (clu, p) in enumerate(zip(clusters, pvals)):
        # clu is a tuple of index arrays (because out_type="indices")
        # For flattened tests, it's typically (indices,)
        idx = clu[0]
        rows.append(
            dict(
                cluster_id=i,
                p_value=float(p),
                n_bins=int(len(idx)),
            )
        )

    df = pd.DataFrame(rows).sort_values("p_value", ascending=True)
    df.to_parquet(out_path / "clusters.parquet", index=False)

    return out_path
