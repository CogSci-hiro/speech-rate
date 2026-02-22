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
