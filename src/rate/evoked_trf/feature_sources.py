from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def subject_emg_path(*, out_dir: Path, dyad_id: str, subject: str) -> Path:
    return out_dir / "features" / "emg" / str(dyad_id) / str(subject) / "emg_env.parquet"


def load_subject_emg(*, out_dir: Path, dyad_id: str, subject: str) -> pd.DataFrame:
    path = subject_emg_path(out_dir=out_dir, dyad_id=dyad_id, subject=subject)
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_parquet(path)
    required = {"run", "time", "emg_env"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"subject EMG table missing required columns: {sorted(missing)}")
    out = df.copy()
    out["run"] = pd.to_numeric(out["run"], errors="coerce")
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["emg_env"] = pd.to_numeric(out["emg_env"], errors="coerce")
    out = out.dropna(subset=["run", "time", "emg_env"]).copy()
    out["run"] = out["run"].astype(int)
    return out


def emg_for_run_aligned(
    *,
    emg_subject: pd.DataFrame,
    run: int,
    n_samples: int,
    sfreq_hz: float,
    sample_t0_s: float,
    zscore_within_run: bool,
) -> np.ndarray:
    run_df = emg_subject.loc[emg_subject["run"].astype(int) == int(run), ["time", "emg_env"]].copy()
    if run_df.empty:
        raise ValueError(f"No EMG rows for run={run}")
    run_df = run_df.sort_values("time", kind="mergesort")

    src_t = run_df["time"].to_numpy(dtype=float)
    src_v = run_df["emg_env"].to_numpy(dtype=float)
    if src_t.size < 2:
        raise ValueError(f"Not enough EMG samples for run={run}")

    target_t = float(sample_t0_s) + (np.arange(int(n_samples), dtype=float) / float(sfreq_hz))
    aligned = np.interp(target_t, src_t, src_v, left=0.0, right=0.0).astype(float)

    if zscore_within_run:
        mu = float(np.nanmean(aligned))
        sd = float(np.nanstd(aligned))
        if np.isfinite(sd) and sd > 0:
            aligned = (aligned - mu) / sd
        else:
            aligned = aligned - mu
    return aligned
