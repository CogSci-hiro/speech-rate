import numpy as np
import pandas as pd
import pytest
from rate.config import ConfigValidationError
from rate.induced_trf.aperiodic import compute_sliding_aperiodic, SlidingAperiodicConfig
from rate.induced_trf.score import aggregate_subject_scores, parse_induced_trf_config
from rate.induced_trf.sliding_fooof import parse_sliding_fooof_config
from rate.evoked_trf.score import select_eeg_path


def _base_cfg() -> dict:
    return {
        "evoked_trf": {
            "downsample_hz": 32.0,
            "lags_s": {"tmin": -0.5, "tmax": 0.5},
            "regularization": {"type": "ridge"},
        },
        "induced_trf": {
            "downsample_hz": 32.0,
            "bands_hz": {
                "theta": [4.0, 7.0],
                "alpha": [8.0, 12.0],
            },
            "bands": ["theta", "alpha"],
            "target_kinds": ["raw", "residual"],
            "residual_bands": ["theta", "alpha"],
        },
        "fooof_sliding": {
            "window_s": 1.0,
            "fit_range_hz": [2.0, 40.0],
        },
    }


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


def test_parse_induced_trf_confirmatory_defaults():
    parsed = parse_induced_trf_config(_base_cfg())
    assert parsed.bands == ("theta", "alpha")
    assert parsed.target_kinds == ("raw", "residual")
    assert parsed.residual_bands == ("theta", "alpha")


def test_parse_induced_trf_allows_residual_only_target_kind():
    cfg = _base_cfg()
    cfg["induced_trf"]["target_kinds"] = ["residual"]

    parsed = parse_induced_trf_config(cfg)

    assert parsed.target_kinds == ("residual",)
    assert parsed.residual_bands == ("theta", "alpha")


def test_parse_sliding_fooof_derives_step_from_induced_downsample():
    cfg = _base_cfg()
    cfg["induced_trf"]["downsample_hz"] = 16.0

    parsed = parse_sliding_fooof_config(cfg)

    assert parsed.step_s == pytest.approx(1.0 / 16.0)


def test_parse_induced_trf_requires_branch_local_downsample():
    cfg = _base_cfg()
    del cfg["induced_trf"]["downsample_hz"]

    with pytest.raises(ConfigValidationError, match="induced_trf.downsample_hz"):
        parse_induced_trf_config(cfg)


def test_parse_sliding_fooof_exposes_multitaper_parameters():
    cfg = _base_cfg()
    cfg["fooof_sliding"]["spectrum_method"] = "multitaper"
    cfg["fooof_sliding"]["multitaper_bandwidth_hz"] = 3.0
    cfg["fooof_sliding"]["multitaper_adaptive"] = True
    cfg["fooof_sliding"]["multitaper_low_bias"] = False

    parsed = parse_sliding_fooof_config(cfg)

    assert parsed.spectrum_method == "multitaper"
    assert parsed.multitaper_bandwidth_hz == pytest.approx(3.0)
    assert parsed.multitaper_adaptive is True
    assert parsed.multitaper_low_bias is False


def test_parse_induced_trf_prefers_induced_ridge_settings():
    cfg = _base_cfg()
    cfg["evoked_trf"]["regularization"] = {"type": "laplacian"}
    cfg["evoked_trf"]["ridge_alpha_grid"] = [1.0, 2.0]
    cfg["evoked_trf"]["ridge_alpha"] = 2.0
    cfg["induced_trf"]["regularization"] = {"type": "ridge"}
    cfg["induced_trf"]["ridge_alpha_grid"] = [10.0, 100.0]
    cfg["induced_trf"]["ridge_alpha"] = 100.0

    parsed = parse_induced_trf_config(cfg)

    assert parsed.regularization_type == "ridge"
    assert parsed.ridge_alpha_grid == [10.0, 100.0]
    assert parsed.ridge_alpha_default == 100.0


def test_parse_induced_trf_rejects_empty_residual_bands():
    cfg = _base_cfg()
    cfg["induced_trf"]["residual_bands"] = []
    with pytest.raises(ValueError, match="residual_bands"):
        parse_induced_trf_config(cfg)


def test_parse_induced_trf_rejects_residual_bands_outside_enabled_bands():
    cfg = _base_cfg()
    cfg["induced_trf"]["residual_bands"] = ["theta", "beta"]
    with pytest.raises(ValueError, match="subset of induced_trf.bands"):
        parse_induced_trf_config(cfg)


def test_parse_induced_trf_rejects_missing_band_definition():
    cfg = _base_cfg()
    del cfg["induced_trf"]["bands_hz"]["alpha"]
    with pytest.raises(ValueError, match="Missing induced_trf.bands_hz entries"):
        parse_induced_trf_config(cfg)


def test_select_eeg_path_rejects_missing_value():
    ds = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-002",
                "subject_id": "sub-004",
                "run": 1,
                "eeg_path": np.nan,
            }
        ]
    )

    with pytest.raises(ValueError, match="missing eeg_path"):
        select_eeg_path(ds, dyad_id="dyad-002", subject="sub-004", run=1)
