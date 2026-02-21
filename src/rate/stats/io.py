from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class KernelOuterLoad:
    """
    Loaded outer-fold kernels for one subject.

    Attributes
    ----------
    kernels_outer
        Array of shape (n_folds, n_channels, n_features, n_lags).
    """
    kernels_outer: np.ndarray


def _subject_model_dir(
    out_dir: Path,
    analysis: str,
    dyad_id: str,
    subject_id: str,
    model: str,
) -> Path:
    return out_dir / analysis / "subjects" / dyad_id / subject_id / "models" / model


def load_subject_kernels_outer(
    out_dir: Path,
    analysis: str,
    dyad_id: str,
    subject_id: str,
    model: str,
) -> KernelOuterLoad:
    """
    Load per-subject outer-fold kernels.

    Parameters
    ----------
    out_dir
        Pipeline output directory.
    analysis
        "evoked_trf" or "induced_trf".
    dyad_id
        Dyad identifier, e.g. "dyad-012".
    subject_id
        Subject identifier, e.g. "sub-023".
    model
        Model name, e.g. "self_rate".

    Returns
    -------
    KernelOuterLoad
        kernels_outer with shape (n_folds, n_channels, n_features, n_lags).

    Usage example
    -------------
        load = load_subject_kernels_outer(
            out_dir=Path("/Volumes/work-4T/speech-rate-testing"),
            analysis="evoked_trf",
            dyad_id="dyad-012",
            subject_id="sub-023",
            model="self_rate",
        )
        kernels_outer = load.kernels_outer
    """
    model_dir = _subject_model_dir(out_dir, analysis, dyad_id, subject_id, model)
    kernels_path = model_dir / "kernels_outer.npy"
    kernels_outer = np.load(kernels_path)
    if kernels_outer.ndim != 4:
        raise ValueError(
            f"Expected kernels_outer.npy to have ndim=4, got shape={kernels_outer.shape} "
            f"at {kernels_path}"
        )
    return KernelOuterLoad(kernels_outer=kernels_outer)


def stack_subject_feature_kernels(
    out_dir: Path,
    analysis: str,
    subject_keys: Sequence[Tuple[str, str]],
    model: str,
    feature_index: int,
) -> List[np.ndarray]:
    """
    Load and stack kernels for a single feature across subjects.

    Parameters
    ----------
    out_dir
        Pipeline output directory.
    analysis
        "evoked_trf" or "induced_trf".
    subject_keys
        Sequence of (dyad_id, subject_id) pairs.
    model
        Model name.
    feature_index
        Feature index along the n_features axis in kernels_outer.npy.

    Returns
    -------
    list[np.ndarray]
        Per-subject arrays of shape (n_folds, n_channels, n_lags).
        The number of folds may vary by subject.

    Usage example
    -------------
        keys = [("dyad-012", "sub-023"), ("dyad-012", "sub-024")]
        data = stack_subject_feature_kernels(
            out_dir=Path("/Volumes/work-4T/speech-rate-testing"),
            analysis="evoked_trf",
            subject_keys=keys,
            model="self_rate",
            feature_index=0,
        )
    """
    per_subject: List[np.ndarray] = []
    expected_ch_lag: tuple[int, int] | None = None
    mismatched_shapes: List[str] = []
    for dyad_id, subject_id in subject_keys:
        load = load_subject_kernels_outer(out_dir, analysis, dyad_id, subject_id, model)
        kernels_outer = load.kernels_outer
        n_folds, n_channels, n_features, n_lags = kernels_outer.shape
        if not (0 <= feature_index < n_features):
            raise IndexError(
                f"feature_index={feature_index} out of bounds for n_features={n_features} "
                f"(subject {dyad_id}/{subject_id}, model={model})"
            )
        subj_arr = kernels_outer[:, :, feature_index, :]  # (fold, ch, lag)
        subj_ch_lag = (subj_arr.shape[1], subj_arr.shape[2])
        if expected_ch_lag is None:
            expected_ch_lag = subj_ch_lag
        elif subj_ch_lag != expected_ch_lag:
            mismatched_shapes.append(
                f"{dyad_id}/{subject_id}: (ch, lag)={subj_ch_lag}, expected={expected_ch_lag}"
            )
            continue
        per_subject.append(subj_arr)

    if mismatched_shapes:
        raise ValueError(
            "Inconsistent (n_channels, n_lags) across subjects for group stats. "
            "Fix upstream preprocessing so all subjects share channel/lag geometry. "
            "Examples: " + "; ".join(mismatched_shapes[:5])
        )
    if not per_subject:
        raise ValueError("No valid subject kernels found for group stats.")
    return per_subject
