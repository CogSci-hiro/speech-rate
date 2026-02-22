import numpy as np
import pandas as pd


def _event_times_from_table(events_test: pd.DataFrame) -> np.ndarray:
    candidates: list[np.ndarray] = []
    for col in ("self_IPU_onset", "other_IPU_offset", "t_onset", "t_offset"):
        if col in events_test.columns:
            vals = pd.to_numeric(events_test[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                candidates.append(vals)
    if not candidates:
        return np.array([], dtype=float)
    times = np.concatenate(candidates, axis=0)
    return np.asarray(np.sort(times), dtype=float)


def _r2_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch for R^2: {yt.shape} vs {yp.shape}")
    ss_res = float(np.sum((yt - yp) ** 2))
    yt_mean = float(np.mean(yt))
    ss_tot = float(np.sum((yt - yt_mean) ** 2))
    if ss_tot <= 0.0 or not np.isfinite(ss_tot):
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def _pearsonr_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch for correlation: {x.shape} vs {y.shape}")
    if x.ndim != 1:
        raise ValueError("Inputs must be 1D")
    x0 = x - np.nanmean(x)
    y0 = y - np.nanmean(y)
    denom = np.sqrt(np.nansum(x0**2) * np.nansum(y0**2))
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(np.nansum(x0 * y0) / denom)


def reconstruct_raw_from_trf(X_test: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Reconstruct continuous EEG from a TRF kernel.

    Parameters
    ----------
    X_test
        Array of shape (n_times, n_features).
    kernel
        Array of shape (n_channels, n_features, n_lags). We assume the zero-lag
        coefficient is at the center index ``n_lags // 2``.

    Returns
    -------
    reconstructed_raw : ndarray, shape (n_channels, n_times)

    Notes
    -----
    This helper is intentionally estimator-agnostic and uses direct convolution.
    The shared engine may still use the estimator's native ``predict`` for exact
    reconstruction parity in production paths.
    """
    X = np.asarray(X_test, dtype=float)
    K = np.asarray(kernel, dtype=float)
    if X.ndim != 2:
        raise ValueError("X_test must be 2D (n_times, n_features)")
    if K.ndim != 3:
        raise ValueError("kernel must be 3D (n_channels, n_features, n_lags)")
    n_times, n_features = X.shape
    n_channels, k_features, n_lags = K.shape
    if n_features != k_features:
        raise ValueError(f"Feature mismatch: X has {n_features}, kernel has {k_features}")
    center = int(n_lags // 2)
    y = np.zeros((n_channels, n_times), dtype=float)
    for lag_idx in range(n_lags):
        shift = int(lag_idx - center)
        if shift == 0:
            x_shift = X
            dst = slice(None)
            src = slice(None)
        elif shift > 0:
            dst = slice(shift, None)
            src = slice(None, n_times - shift)
            x_shift = X[src]
        else:
            dst = slice(None, n_times + shift)
            src = slice(-shift, None)
            x_shift = X[src]
        y[:, dst] += np.einsum("tf,cf->ct", x_shift, K[:, :, lag_idx], optimize=True)
    return np.asarray(y, dtype=float)


def _epoch_average(
    *,
    raw_ch_by_time: np.ndarray,
    event_times_s: np.ndarray,
    sfreq: float,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray | None, np.ndarray, int, int]:
    raw = np.asarray(raw_ch_by_time, dtype=float)
    if raw.ndim != 2:
        raise ValueError("raw_ch_by_time must be 2D (n_channels, n_times)")
    n_channels, n_times = raw.shape
    lag_start = int(np.floor(float(tmin) * float(sfreq)))
    lag_stop = int(np.ceil(float(tmax) * float(sfreq)))
    rel_idx = np.arange(lag_start, lag_stop + 1, dtype=int)
    times_rel = rel_idx / float(sfreq)
    centers = np.round(np.asarray(event_times_s, dtype=float) * float(sfreq)).astype(int)
    epochs: list[np.ndarray] = []
    for c in centers:
        idx = c + rel_idx
        if idx.size == 0 or idx[0] < 0 or idx[-1] >= n_times:
            continue
        epochs.append(raw[:, idx])  # (ch, time)
    if not epochs:
        return None, times_rel, int(centers.size), 0
    return np.mean(np.stack(epochs, axis=0), axis=0), times_rel, int(centers.size), int(len(epochs))


def compute_reconstruction_erp(
    reconstructed_raw: np.ndarray,
    original_raw: np.ndarray,
    events: pd.DataFrame,
    tmin: float,
    tmax: float,
    sfreq: float,
) -> dict[str, np.ndarray]:
    """
    Oganian & Chang-style reconstruction ERP similarity.

    Event lock policy is inferred from the available event columns:
    - prefer ``self_IPU_onset``
    - otherwise use ``other_IPU_offset``

    Returns
    -------
    dict with:
      - ``r_channel`` : (n_channels,)
      - ``r_time``    : (n_channels, n_times_epoch)  [time-resolved pseudo-R^2]
      - ``times``     : (n_times_epoch,)
    """
    rec = np.asarray(reconstructed_raw, dtype=float)
    orig = np.asarray(original_raw, dtype=float)
    if rec.ndim != 2 or orig.ndim != 2:
        raise ValueError("reconstructed_raw and original_raw must be 2D (n_channels, n_times)")
    if rec.shape != orig.shape:
        raise ValueError(f"Shape mismatch: {rec.shape} vs {orig.shape}")

    if "self_IPU_onset" in events.columns:
        event_times = pd.to_numeric(events["self_IPU_onset"], errors="coerce").to_numpy(dtype=float)
    elif "other_IPU_offset" in events.columns:
        event_times = pd.to_numeric(events["other_IPU_offset"], errors="coerce").to_numpy(dtype=float)
    elif "t_onset" in events.columns:
        event_times = pd.to_numeric(events["t_onset"], errors="coerce").to_numpy(dtype=float)
    else:
        raise ValueError("events must contain self_IPU_onset or other_IPU_offset for ERP reconstruction")
    event_times = event_times[np.isfinite(event_times)]

    rec_evoked, times_rel, n_events_total, n_events_used = _epoch_average(
        raw_ch_by_time=rec,
        event_times_s=event_times,
        sfreq=float(sfreq),
        tmin=float(tmin),
        tmax=float(tmax),
    )
    orig_evoked, _, _, _ = _epoch_average(
        raw_ch_by_time=orig,
        event_times_s=event_times,
        sfreq=float(sfreq),
        tmin=float(tmin),
        tmax=float(tmax),
    )
    n_channels = rec.shape[0]
    n_times_epoch = int(times_rel.size)
    if rec_evoked is None or orig_evoked is None:
        return {
            "r_channel": np.full(n_channels, np.nan, dtype=float),
            "r_time": np.full((n_channels, n_times_epoch), np.nan, dtype=float),
            "times": np.asarray(times_rel, dtype=float),
            "n_events_total": np.full(n_channels, n_events_total, dtype=int),
            "n_events_used": np.full(n_channels, n_events_used, dtype=int),
        }

    r_channel = np.array([_pearsonr_1d(orig_evoked[ch, :], rec_evoked[ch, :]) for ch in range(n_channels)], dtype=float)
    # "Optional time-resolved R^2": per-timepoint squared error relative to channel ERP variance.
    r_time = np.full((n_channels, n_times_epoch), np.nan, dtype=float)
    for ch in range(n_channels):
        var = float(np.nanvar(orig_evoked[ch, :]))
        if not np.isfinite(var) or var <= 0:
            continue
        r_time[ch, :] = 1.0 - ((orig_evoked[ch, :] - rec_evoked[ch, :]) ** 2) / var

    return {
        "r_channel": r_channel,
        "r_time": r_time,
        "times": np.asarray(times_rel, dtype=float),
        "n_events_total": np.full(n_channels, n_events_total, dtype=int),
        "n_events_used": np.full(n_channels, n_events_used, dtype=int),
    }


def compute_reconstruction_metrics_df(
    *,
    y_test_raw: np.ndarray,
    y_test_pred: np.ndarray,
    events_test: pd.DataFrame,
    sfreq: float,
    tmin: float,
    tmax: float,
    event_lock: str | None = None,
) -> pd.DataFrame:
    """
    Long-form DataFrame wrapper for ``compute_reconstruction_erp`` suitable for engine callbacks.

    ``y_*`` are expected as (n_samples, n_channels).
    """
    raw = np.asarray(y_test_raw, dtype=float)
    pred = np.asarray(y_test_pred, dtype=float)
    if raw.ndim != 2 or pred.ndim != 2:
        raise ValueError("y_test_raw and y_test_pred must be 2D (samples x channels)")
    if raw.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {raw.shape} vs {pred.shape}")
    events_use = events_test.copy()
    if event_lock is not None:
        if str(event_lock) not in events_use.columns:
            raise ValueError(f"event_lock column not found: {event_lock}")
        # Create a preferred column so compute_reconstruction_erp uses the requested lock.
        events_use["self_IPU_onset"] = pd.to_numeric(events_use[str(event_lock)], errors="coerce")

    metrics = compute_reconstruction_erp(
        reconstructed_raw=np.asarray(pred.T, dtype=float),
        original_raw=np.asarray(raw.T, dtype=float),
        events=events_use,
        tmin=float(tmin),
        tmax=float(tmax),
        sfreq=float(sfreq),
    )
    r_channel = np.asarray(metrics["r_channel"], dtype=float)
    r_time = np.asarray(metrics["r_time"], dtype=float)
    times = np.asarray(metrics["times"], dtype=float)
    n_events_total = np.asarray(metrics["n_events_total"])
    n_events_used = np.asarray(metrics["n_events_used"])
    rows: list[dict[str, object]] = []
    for ch in range(r_time.shape[0]):
        for ti in range(r_time.shape[1]):
            rows.append(
                {
                    "channel": int(ch),
                    "time_s": float(times[ti]),
                    "r_time": float(r_time[ch, ti]),
                    "r_channel": float(r_channel[ch]),
                    "n_events_total": int(n_events_total[ch]),
                    "n_events_used": int(n_events_used[ch]),
                }
            )
    return pd.DataFrame(rows)


def compute_reconstruction_r2(
    y_test_raw: np.ndarray,
    y_test_pred: np.ndarray,
    events_test: pd.DataFrame,
    sfreq: float,
    tmin: float,
    tmax: float,
) -> pd.DataFrame:
    """
    Compute per-channel event-locked reconstruction R^2 on the OUTER test data only.

    We epoch raw/predicted test signals around test-run events, average across events,
    and compare predicted-vs-raw evoked waveforms per channel.
    """
    y_raw = np.asarray(y_test_raw, dtype=float)
    y_pred = np.asarray(y_test_pred, dtype=float)
    if y_raw.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_test_raw and y_test_pred must be 2D (samples x channels)")
    if y_raw.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_raw.shape} vs {y_pred.shape}")
    if float(sfreq) <= 0:
        raise ValueError("sfreq must be > 0")

    n_samples, n_channels = y_raw.shape
    lag_start = int(np.floor(float(tmin) * float(sfreq)))
    lag_stop = int(np.ceil(float(tmax) * float(sfreq)))
    rel_idx = np.arange(lag_start, lag_stop + 1, dtype=int)
    if rel_idx.size < 1:
        raise ValueError("Lag window produced zero samples")

    event_times = _event_times_from_table(events_test)
    if event_times.size == 0:
        return pd.DataFrame(
            {
                "channel": np.arange(n_channels, dtype=int),
                "r2": np.full(n_channels, np.nan, dtype=float),
                "n_events": np.zeros(n_channels, dtype=int),
                "n_events_total": np.zeros(n_channels, dtype=int),
                "n_events_used": np.zeros(n_channels, dtype=int),
            }
        )

    centers = np.round(event_times * float(sfreq)).astype(int)
    epochs_raw: list[np.ndarray] = []
    epochs_pred: list[np.ndarray] = []
    for c in centers:
        idx = c + rel_idx
        if idx[0] < 0 or idx[-1] >= n_samples:
            continue
        epochs_raw.append(y_raw[idx, :])
        epochs_pred.append(y_pred[idx, :])

    n_events_total = int(event_times.size)
    n_events_used = int(len(epochs_raw))
    if n_events_used == 0:
        return pd.DataFrame(
            {
                "channel": np.arange(n_channels, dtype=int),
                "r2": np.full(n_channels, np.nan, dtype=float),
                "n_events": np.zeros(n_channels, dtype=int),
                "n_events_total": np.full(n_channels, n_events_total, dtype=int),
                "n_events_used": np.zeros(n_channels, dtype=int),
            }
        )

    raw_evoked = np.mean(np.stack(epochs_raw, axis=0), axis=0)   # (time, channel)
    pred_evoked = np.mean(np.stack(epochs_pred, axis=0), axis=0)  # (time, channel)

    rows: list[dict[str, object]] = []
    for ch in range(n_channels):
        rows.append(
            {
                "channel": int(ch),
                "r2": _r2_1d(raw_evoked[:, ch], pred_evoked[:, ch]),
                "n_events": int(n_events_used),
                "n_events_total": int(n_events_total),
                "n_events_used": int(n_events_used),
            }
        )
    return pd.DataFrame(rows)
