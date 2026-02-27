import numpy as np
import pandas as pd
import pytest
from rate.induced_trf.aperiodic import compute_sliding_aperiodic, SlidingAperiodicConfig
from rate.induced_trf.score import aggregate_subject_scores


def test_half_window_edge_drop():
    sfreq = 100
    times = np.linspace(0, 10, 1000)
    data = np.random.randn(1, 1000)

    cfg = SlidingAperiodicConfig(
        window_s=2.0,
        step_s=1.0,
        fit_range_hz=(2, 30),
        edge_drop="half_window",
    )

    df = compute_sliding_aperiodic(data, sfreq, times, cfg, band_hz=(4.0, 8.0), channel_names=["Fz"])

    assert df["time"].min() >= 1.0
    assert df["time"].max() <= 9.0
    assert {"channel", "offset", "exponent", "band_residual_power", "band_peak_power"}.issubset(set(df.columns))


def test_sliding_aperiodic_emits_per_channel_rows():
    rng = np.random.default_rng(0)
    sfreq = 128.0
    times = np.arange(0.0, 6.0, 1.0 / sfreq)
    # Channel 0 includes stronger theta, channel 1 mostly noise.
    x0 = np.sin(2.0 * np.pi * 6.0 * times) + 0.1 * rng.standard_normal(times.shape[0])
    x1 = 0.3 * rng.standard_normal(times.shape[0])
    data = np.vstack([x0, x1])

    cfg = SlidingAperiodicConfig(
        window_s=1.0,
        step_s=0.2,
        fit_range_hz=(2.0, 40.0),
        edge_drop="half_window",
        aperiodic_mode="fixed",
    )
    df = compute_sliding_aperiodic(data, sfreq, times, cfg, band_hz=(4.0, 8.0), channel_names=["Fz", "Pz"])

    assert not df.empty
    assert set(df["channel"].unique().tolist()) == {"Fz", "Pz"}
    assert np.isfinite(df["offset"]).any()
    assert np.isfinite(df["exponent"]).any()
    assert np.isfinite(df["band_total_power"]).any()


def test_induced_aggregate_extracts_target_kind(tmp_path):
    try:
        pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "_probe.parquet", index=False)
    except Exception:
        pytest.skip("Parquet engine not available in test environment")

    base_row = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-01",
                "subject": "sub-01",
                "run": 1,
                "model": "baseline",
                "fold_id": 0,
                "r": 0.1,
            }
        ]
    )
    raw_path = tmp_path / "induced_trf_theta" / "subjects" / "dyad-01" / "sub-01" / "scores.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    base_row.to_parquet(raw_path, index=False)
    res_path = tmp_path / "induced_trf_theta_residual" / "subjects" / "dyad-01" / "sub-01" / "scores.parquet"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    base_row.to_parquet(res_path, index=False)

    out = aggregate_subject_scores(out_dir=tmp_path)
    assert set(out["target_kind"].unique().tolist()) == {"raw", "residual"}
    assert set(out["band"].unique().tolist()) == {"theta"}
