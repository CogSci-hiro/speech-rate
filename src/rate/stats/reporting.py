from pathlib import Path
from typing import List, Optional

import mne
import numpy as np
import pandas as pd

from .two_level import TwoLevelClusterResult


def _resolve_feature_dir(feature_name: str, feature_index: int) -> str:
    """Resolve output directory name for a model feature."""
    if str(feature_name) == "base":
        return "base"
    return f"feature-{feature_index}"


def write_two_level_cluster_result(
    result: TwoLevelClusterResult,
    out_dir: Path,
    analysis: str,
    model: str,
    feature_name: str,
    feature_index: int,
    lags: np.ndarray,
    tables_format: str = "parquet",
) -> Path:
    """
    Write cluster results to disk.

    Outputs
    -------
    - clusters.{csv|parquet}
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
    feature_dir = _resolve_feature_dir(feature_name=feature_name, feature_index=feature_index)
    out_path = out_dir / "stats" / analysis / model / feature_dir
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / "t_obs.npy", result.group_cluster.t_obs)
    np.save(out_path / "h0.npy", result.group_cluster.h0)
    np.save(out_path / "subject_z_maps.npy", result.subject_z_maps)

    # Save grand-average z-scored TRF as an Evoked list (single evoked entry).
    if lags.ndim != 1 or len(lags) < 1:
        raise ValueError(f"lags must be a 1D non-empty array, got shape={lags.shape}")
    if result.subject_z_maps.shape[2] != len(lags):
        raise ValueError(
            f"Lag mismatch: subject_z_maps n_lags={result.subject_z_maps.shape[2]} != len(lags)={len(lags)}"
        )
    if len(result.ch_names) != result.subject_z_maps.shape[1]:
        raise ValueError(
            f"Channel mismatch: len(ch_names)={len(result.ch_names)} != n_channels={result.subject_z_maps.shape[1]}"
        )
    if len(lags) > 1:
        dt = float(np.median(np.diff(lags)))
        if dt <= 0:
            raise ValueError(f"Expected increasing lag grid, got median dt={dt}")
        sfreq = 1.0 / dt
    else:
        sfreq = 1.0
    info = mne.create_info(ch_names=result.ch_names, sfreq=sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("biosemi64")
    info.set_montage(montage, on_missing="ignore")
    grand_avg = result.subject_z_maps.mean(axis=0)
    grand_evoked = mne.EvokedArray(
        grand_avg,
        info=info,
        tmin=float(lags[0]),
        comment=f"{analysis}:{model}:{feature_name}:feature-{feature_index}",
        nave=int(result.subject_z_maps.shape[0]),
        verbose=False,
    )
    mne.write_evokeds(out_path / "grand_average_trfs-ave.fif", [grand_evoked], overwrite=True)

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
    if tables_format == "csv":
        df.to_csv(out_path / "clusters.csv", index=False)
    else:
        df.to_parquet(out_path / "clusters.parquet", index=False)

    return out_path
