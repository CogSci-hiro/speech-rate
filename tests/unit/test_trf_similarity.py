import numpy as np

from rate.trf.reconstruction import compute_reconstruction_erp
from rate.trf.similarity import cross_kernel_temporal_similarity


def test_cross_kernel_temporal_similarity_shapes_and_t0_peak() -> None:
    # channels x lags
    a = np.array(
        [
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
        ]
    )
    b = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )
    out = cross_kernel_temporal_similarity(a, b, t0_index=0)
    assert out["self_self_corr"].shape == (3,)
    assert out["self_other_corr"].shape == (3,)
    assert np.isclose(out["self_self_corr"][0], 1.0, equal_nan=False)


def test_compute_reconstruction_erp_returns_channel_and_time_metrics() -> None:
    # ch x time
    orig = np.vstack([np.linspace(0, 9, 10), np.linspace(10, 19, 10)])
    rec = orig.copy()
    import pandas as pd

    events = pd.DataFrame({"self_IPU_onset": [2.0, 5.0]})
    out = compute_reconstruction_erp(
        reconstructed_raw=rec,
        original_raw=orig,
        events=events,
        tmin=0.0,
        tmax=1.0,
        sfreq=1.0,
    )
    assert set(out.keys()) >= {"r_channel", "r_time", "times"}
    assert out["r_channel"].shape == (2,)
    assert out["r_time"].shape[0] == 2
    assert np.allclose(out["r_channel"], [1.0, 1.0], equal_nan=True)
