from dataclasses import dataclass
from pathlib import Path
import uuid

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetailedPaths:
    """
    Standard output locations for detailed evoked-TRF results.

    Usage example
    -------------
        p = DetailedPaths.from_out_dir(Path("..."))
    """

    root: Path
    inner_scores: Path
    outer_scores: Path
    best_alpha: Path
    best_model: Path
    models_dir: Path

    @staticmethod
    def from_out_dir(out_dir: Path) -> "DetailedPaths":
        root = out_dir / "evoked_trf" / "detailed"
        return DetailedPaths(
            root=root,
            inner_scores=root / "inner_cv_scores.parquet",
            outer_scores=root / "outer_scores_full.parquet",
            best_alpha=root / "best_alpha.parquet",
            best_model=root / "best_model_per_subject.parquet",
            models_dir=root / "models",
        )


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Write parquet with parent mkdir.

    Usage example
    -------------
        write_parquet(df, Path("out.parquet"))
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        df.to_parquet(tmp_path, index=False)
        # Read-back verification avoids propagating silently corrupted outputs.
        pd.read_parquet(tmp_path)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def save_receptive_field_npz(
    *,
    path: Path,
    dyad_id: str,
    subject: str,
    model: str,
    sfreq_hz: float,
    tmin_s: float,
    tmax_s: float,
    alpha: float,
    feature_names: list[str],
    coef: np.ndarray,
    intercept: np.ndarray | None,
) -> None:
    """
    Save the minimal parameters required to reproduce predictions.

    Notes
    -----
    We store coefficients and metadata, not the Python object.

    Usage example
    -------------
        save_receptive_field_npz(path=..., dyad_id=..., ...)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dyad_id=np.array(dyad_id),
        subject=np.array(subject),
        model=np.array(model),
        sfreq_hz=np.array(float(sfreq_hz)),
        tmin_s=np.array(float(tmin_s)),
        tmax_s=np.array(float(tmax_s)),
        alpha=np.array(float(alpha)),
        feature_names=np.array(feature_names, dtype=object),
        coef=np.asarray(coef),
        intercept=np.asarray(intercept) if intercept is not None else np.array([]),
    )
