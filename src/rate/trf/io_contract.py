from pathlib import Path

import numpy as np

from rate.trf.engine import TRFResult


def write_trf_artifacts(result: TRFResult, output_dir: Path) -> None:
    """
    Write the stable per-model TRF artifact contract.
    The caller controls directory layout and passes the desired output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result.outer_scores.to_parquet(output_dir / "outer_scores.parquet", index=False)
    np.save(output_dir / "kernels_outer.npy", np.asarray(result.kernels_outer))
    np.save(output_dir / "lags.npy", np.asarray(result.lags, dtype=float))
    if result.reconstruction_r2 is not None:
        result.reconstruction_r2.to_parquet(output_dir / "reconstruction_r2.parquet", index=False)
