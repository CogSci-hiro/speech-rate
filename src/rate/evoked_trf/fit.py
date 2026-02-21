from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rate.evoked_trf.spec import ModelSpec, RegressorSpec, get_event_times_and_amps

_MNE_LOGGING_CONFIGURED = False


def _configure_mne_logging() -> None:
    """Silence verbose MNE info output during repeated CV fits."""
    global _MNE_LOGGING_CONFIGURED
    if _MNE_LOGGING_CONFIGURED:
        return
    import mne

    mne.set_log_level("WARNING")
    _MNE_LOGGING_CONFIGURED = True


@dataclass
class FoldData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


@dataclass(frozen=True)
class EegRunData:
    """
    Container for run-level EEG arrays used for TRF fitting.

    Attributes
    ----------
    y
        EEG data of shape (n_samples, n_channels).
    sfreq_hz
        Sampling frequency in Hz.
    times_s
        Time axis in seconds of shape (n_samples,).

    Usage example
    -------------
        run = load_eeg_run(eeg_path, downsample_hz=100.0)
        assert run.y.shape[0] == run.times_s.size
    """

    y: np.ndarray
    sfreq_hz: float
    times_s: np.ndarray


def prepare_fold_data(
    *,
    events_df: pd.DataFrame,
    model_spec,
    train_runs: list[int],
    test_run: int,
):
    """
    Build and standardize train/test arrays for one fold.

    No alpha here.
    No RF fitting here.
    """

    # Build full design matrix once
    X, y = build_design_matrix(events_df, model_spec)

    train_mask = events_df["run"].isin(train_runs)
    test_mask = events_df["run"] == test_run

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    return X_train_std, y_train, X_test_std, y_test


def load_eeg_run(
    eeg_path: Path,
    *,
    downsample_hz: float,
    picks: str | list[str] = "eeg",
) -> EegRunData:
    """
    Load a preprocessed EEG run with MNE and resample to `downsample_hz`.

    Parameters
    ----------
    eeg_path
        Path to a MNE Raw file (e.g., .fif / .fif.gz).
    downsample_hz
        Target sampling rate in Hz.
    picks
        MNE picks passed to Raw.get_data(). Default: "eeg".

    Returns
    -------
    EegRunData
        y is (n_samples, n_channels), times_s is (n_samples,).

    Notes
    -----
    - This function performs *only* resampling. All other preprocessing is assumed
      to have happened upstream, per your repo constraints.
    - If you want to exclude bad channels, do it upstream or pass a pick list.

    Usage example
    -------------
        run = load_eeg_run(Path("sub-001_run-1_raw.fif"), downsample_hz=100.0)
        y = run.y
    """
    import mne  # local import keeps module import light
    _configure_mne_logging()

    eeg_path = Path(eeg_path)
    if not eeg_path.exists():
        raise FileNotFoundError(str(eeg_path))

    raw = mne.io.read_raw(eeg_path, preload=True, verbose="ERROR")

    if downsample_hz <= 0:
        raise ValueError("downsample_hz must be > 0")

    if not np.isclose(raw.info["sfreq"], downsample_hz, atol=1e-9):
        # MNE uses anti-alias filtering internally for resample
        raw.resample(downsample_hz, npad="auto", verbose="ERROR")

    data = raw.get_data(picks=picks)  # (n_channels, n_samples)
    y = np.asarray(data, dtype=float).T  # (n_samples, n_channels)
    sfreq_hz = float(raw.info["sfreq"])
    times_s = np.asarray(raw.times, dtype=float)  # (n_samples,)

    if y.ndim != 2:
        raise ValueError(f"Expected 2D EEG array after picking; got shape={y.shape}")

    return EegRunData(y=y, sfreq_hz=sfreq_hz, times_s=times_s)


def _events_to_continuous_regressor(
    *,
    events: pd.DataFrame,
    reg: RegressorSpec,
    sfreq_hz: float,
    n_samples: int,
    sample_t0_s: float = 0.0,
) -> np.ndarray:
    """
    Convert event times + amplitudes into a continuous regressor time series.

    The regressor is an impulse train with amplitudes at nearest sample.

    Parameters
    ----------
    events
        Events table for a single (dyad_id, subject_id, run).
        Must already contain the time-lock columns required by `reg`.
    reg
        Regressor specification.
    sfreq_hz
        Sampling frequency in Hz.
    n_samples
        Number of samples in the EEG run.
    sample_t0_s
        Time (seconds) represented by sample index 0 in the loaded run.
        Use ``raw.times[0]`` to account for non-zero ``raw.first_samp``.

    Returns
    -------
    x
        Array of shape (n_samples,).

    Usage example
    -------------
        x = _events_to_continuous_regressor(events=ev, reg=some_reg, sfreq_hz=100.0, n_samples=12000)
    """
    times_s, amps = get_event_times_and_amps(events, reg)

    if np.any(~np.isfinite(times_s)):
        raise ValueError(f"Non-finite event times for regressor {reg.name}")

    finite_amp = np.isfinite(amps)
    if not np.all(finite_amp):
        # Drop events with missing/inf amplitudes for this regressor.
        times_s = times_s[finite_amp]
        amps = amps[finite_amp]

    idx = np.round((times_s - float(sample_t0_s)) * sfreq_hz).astype(int)
    idx = np.clip(idx, 0, n_samples - 1)

    x = np.zeros(n_samples, dtype=float)
    # Sum amplitudes if multiple events land on the same sample
    np.add.at(x, idx, amps)
    return x


def build_design_matrix(
    *,
    events: pd.DataFrame,
    model: ModelSpec,
    sfreq_hz: float,
    n_samples: int,
    sample_t0_s: float = 0.0,
    continuous_regressors: dict[str, np.ndarray] | None = None,
    return_feature_names: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[str]]:
    """
    Build the (non-lagged) design matrix X for a model.

    MNE's ReceptiveField handles lag expansion internally, so X here is simply
    the continuous regressor time series (one column per regressor).

    Parameters
    ----------
    events
        Events table for a single (dyad_id, subject_id, run), with required columns.
    model
        Model specification (ordered regressors).
    sfreq_hz
        Sampling frequency in Hz.
    n_samples
        Number of EEG samples.
    sample_t0_s
        Time (seconds) represented by sample index 0 in the loaded run.
        Set this to ``raw.times[0]`` for alignment when ``raw.first_samp`` is non-zero.
    return_feature_names
        If True, return (X, feature_names).

    Returns
    -------
    X
        Array of shape (n_samples, n_regressors), dtype float.
    feature_names
        Names aligned to columns of X, returned only when return_feature_names=True.

    Usage example
    -------------
        X = build_design_matrix(events=ev, model=model, sfreq_hz=run.sfreq_hz, n_samples=run.y.shape[0])
    """
    cols: list[np.ndarray] = []
    for reg in model.regressors:
        if reg.kind == "emg":
            if continuous_regressors is None or reg.name not in continuous_regressors:
                raise ValueError(f"Missing continuous regressor values for '{reg.name}'")
            x = np.asarray(continuous_regressors[reg.name], dtype=float)
        else:
            x = _events_to_continuous_regressor(
                events=events,
                reg=reg,
                sfreq_hz=sfreq_hz,
                n_samples=n_samples,
                sample_t0_s=sample_t0_s,
            )
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.shape[0] != n_samples:
            raise ValueError(
                f"Regressor {reg.name} produced shape {x.shape}; expected ({n_samples},)"
            )
        cols.append(x)

    if not cols:
        raise ValueError("Model has no regressors; cannot build design matrix")

    X = np.column_stack(cols).astype(float)

    if return_feature_names:
        feature_names = [reg.name for reg in model.regressors]
        return X, feature_names

    return X


def fit_receptive_field(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sfreq_hz: float,
    tmin_s: float,
    tmax_s: float,
    ridge_alpha: float,
    regularization_type: str = "ridge",
) -> Any:
    """
    Fit an MNE TRF model with configurable regularization.

    Parameters
    ----------
    X_train
        Shape (n_samples, n_features).
    y_train
        Shape (n_samples, n_channels).
    sfreq_hz
        Sampling rate in Hz.
    tmin_s, tmax_s
        Lag window in seconds.
    ridge_alpha
        Regularization strength (alpha).
    regularization_type
        One of {"ridge", "laplacian"}.

    Returns
    -------
    rf
        Fitted mne.decoding.ReceptiveField instance.

    Notes
    -----
    We explicitly provide an sklearn Ridge estimator to make regularization
    explicit and reproducible.

    Usage example
    -------------
        rf = fit_receptive_field(
            X_train=X_train, y_train=y_train, sfreq_hz=100.0,
            tmin_s=-0.2, tmax_s=0.8, ridge_alpha=1.0
        )
    """
    if X_train.ndim != 2:
        raise ValueError("X_train must be 2D (samples x features)")
    if y_train.ndim != 2:
        raise ValueError("y_train must be 2D (samples x channels)")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same n_samples")
    if ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be > 0")

    reg = str(regularization_type).strip().lower()
    if reg not in {"ridge", "laplacian"}:
        raise ValueError("regularization_type must be 'ridge' or 'laplacian'")
    _configure_mne_logging()

    if reg == "ridge":
        from mne.decoding import ReceptiveField
        from sklearn.linear_model import Ridge

        estimator = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
        rf = ReceptiveField(
            tmin=float(tmin_s),
            tmax=float(tmax_s),
            sfreq=float(sfreq_hz),
            estimator=estimator,
            scoring="corrcoef",
        )
    else:
        from mne.decoding import TimeDelayingRidge

        rf = TimeDelayingRidge(
            tmin=float(tmin_s),
            tmax=float(tmax_s),
            sfreq=float(sfreq_hz),
            alpha=float(ridge_alpha),
            reg_type="laplacian",
            fit_intercept=True,
        )
    rf.fit(X_train, y_train)
    return rf


def predict_receptive_field(rf: Any, X_test: np.ndarray) -> np.ndarray:
    """
    Predict EEG from a fitted ReceptiveField.

    Parameters
    ----------
    rf
        Fitted ReceptiveField.
    X_test
        Shape (n_samples, n_features).

    Returns
    -------
    y_pred
        Shape (n_samples, n_channels).

    Usage example
    -------------
        y_hat = predict_receptive_field(rf, X_test)
    """
    y_pred = rf.predict(X_test)
    return np.asarray(y_pred, dtype=float)
