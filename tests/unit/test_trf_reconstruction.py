from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rate.trf.engine import TRFResult, run_trf
from rate.trf.io_contract import write_trf_artifacts
from rate.trf.reconstruction import compute_reconstruction_r2


def test_run_trf_reconstruction_uses_outer_test_data_only() -> None:
    X_by_run = {
        1: np.array([[1.0], [2.0], [3.0]], dtype=float),
        2: np.array([[10.0], [20.0], [30.0], [40.0]], dtype=float),
    }
    y_by_run = {
        1: np.array([[1.0], [2.0], [3.0]], dtype=float),
        2: np.array([[10.0], [20.0], [30.0], [40.0]], dtype=float),
    }
    events_by_run = {
        1: pd.DataFrame({"self_IPU_onset": [1.0], "run_marker": [1]}),
        2: pd.DataFrame({"self_IPU_onset": [1.0], "run_marker": [2]}),
    }

    calls: list[tuple[int, int]] = []

    class IdentityModel:
        pass

    def fit_fn(*, X_train, y_train, alpha):
        _ = (X_train, y_train, alpha)
        return IdentityModel()

    def predict_fn(model, X_test):
        _ = model
        return np.asarray(X_test, dtype=float)

    def score_fn(y_true, y_pred):
        _ = (y_true, y_pred)
        return 0.0

    def extract_kernel_fn(model):
        _ = model
        return np.zeros((1, 1, 1), dtype=float), np.array([0.0], dtype=float)

    def reconstruction_fn(*, y_test_raw, y_test_pred, events_test, sfreq, tmin, tmax):
        _ = (sfreq, tmin, tmax)
        marker = int(events_test["run_marker"].iloc[0])
        calls.append((marker, int(y_test_raw.shape[0])))
        assert y_test_raw.shape == y_test_pred.shape
        assert int(y_test_raw.shape[0]) == X_by_run[marker].shape[0]
        assert np.array_equal(y_test_raw, y_by_run[marker])
        assert np.array_equal(y_test_pred, X_by_run[marker])
        return pd.DataFrame({"channel": [0], "r2": [0.5]})

    result = run_trf(
        X_by_run=X_by_run,
        y_by_run=y_by_run,
        lags=np.array([0.0], dtype=float),
        alpha_grid=np.array([1.0], dtype=float),
        cv_config={
            "gap_samples": 0,
            "alpha_default": 1.0,
            "concat_fn": lambda arrays, gap_samples=0: np.concatenate(arrays, axis=0),
            "standardize_fn": lambda X_train, X_test: (X_train, X_test),
            "fit_fn": fit_fn,
            "predict_fn": predict_fn,
            "score_fn": score_fn,
            "extract_kernel_fn": extract_kernel_fn,
            "enable_reconstruction": True,
            "reconstruction_fn": reconstruction_fn,
            "events_by_run": events_by_run,
            "reconstruction_kwargs": {"sfreq": 1.0, "tmin": 0.0, "tmax": 0.0},
        },
    )

    assert sorted(calls) == [(1, 3), (2, 4)]
    assert result.reconstruction_r2 is not None
    assert set(["fold_id", "test_run", "alpha"]).issubset(result.reconstruction_r2.columns)


def test_write_trf_artifacts_preserves_legacy_files_when_no_reconstruction(tmp_path: Path) -> None:
    try:
        pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "_probe.parquet", index=False)
    except Exception:
        pytest.skip("Parquet engine not available in test environment")
    (tmp_path / "_probe.parquet").unlink()

    result = TRFResult(
        kernels_outer=np.zeros((1, 1, 1, 1), dtype=float),
        chosen_alpha=np.array([1.0], dtype=float),
        outer_scores=pd.DataFrame({"alpha": [1.0], "r": [0.0]}),
        lags=np.array([0.0], dtype=float),
    )
    write_trf_artifacts(result, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["kernels_outer.npy", "lags.npy", "outer_scores.parquet"]

    result_with_recon = TRFResult(
        kernels_outer=np.zeros((1, 1, 1, 1), dtype=float),
        chosen_alpha=np.array([1.0], dtype=float),
        outer_scores=pd.DataFrame({"alpha": [1.0], "r": [0.0]}),
        lags=np.array([0.0], dtype=float),
        channel_names=("Fz",),
        reconstruction_r2=pd.DataFrame({"channel": [0], "r2": [0.1]}),
    )
    tmp_path2 = tmp_path / "with_recon"
    write_trf_artifacts(result_with_recon, tmp_path2)
    assert sorted(p.name for p in tmp_path2.iterdir()) == [
        "channel_names.npy",
        "kernels_outer.npy",
        "lags.npy",
        "outer_scores.parquet",
        "reconstruction_metrics.parquet",
        "reconstruction_r2.parquet",
    ]
    np.testing.assert_array_equal(np.load(tmp_path2 / "channel_names.npy"), np.array(["Fz"], dtype=str))


def test_compute_reconstruction_r2_returns_per_channel_rows() -> None:
    y_raw = np.arange(20.0, dtype=float).reshape(10, 2)
    y_pred = y_raw.copy()
    events = pd.DataFrame({"self_IPU_onset": [2.0, 5.0]})
    out = compute_reconstruction_r2(
        y_test_raw=y_raw,
        y_test_pred=y_pred,
        events_test=events,
        sfreq=1.0,
        tmin=0.0,
        tmax=1.0,
    )
    assert list(out["channel"]) == [0, 1]
    assert np.allclose(out["r2"].to_numpy(dtype=float), [1.0, 1.0], equal_nan=True)
