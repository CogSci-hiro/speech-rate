import numpy as np
from rate.induced_trf.aperiodic import compute_sliding_aperiodic, SlidingAperiodicConfig


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

    df = compute_sliding_aperiodic(data, sfreq, times, cfg)

    assert df["time"].min() >= 1.0
    assert df["time"].max() <= 9.0
