import numpy as np
import pandas as pd
import pytest

from rate.evoked_trf.fit import _events_to_continuous_regressor, build_design_matrix
from rate.evoked_trf.spec import (
    RegressorSpec,
    add_timelock_columns,
    get_event_times_and_amps,
    get_models,
    self_rate_regressor,
)
from rate.evoked_trf.utils import (
    concatenate_with_gap,
    max_lag_samples,
    pearsonr_1d,
    score_multichannel_mean_r,
    standardize_design_matrix_foldwise,
)


def test_get_models_order_and_regressor_membership_contract() -> None:
    models = get_models()
    assert [m.model_name for m in models] == [
        "baseline",
        "self_rate",
        "other_rate",
        "abs_diff",
        "self_rate_sq",
        "other_rate_sq",
        "abs_diff_sq",
    ]

    by_name = {m.model_name: [r.name for r in m.regressors] for m in models}
    assert by_name["baseline"] == ["self_onset_impulse", "other_offset_impulse", "emg"]
    assert by_name["self_rate"] == by_name["baseline"] + ["self_rate"]
    assert by_name["other_rate"] == by_name["baseline"] + ["other_rate"]
    assert by_name["abs_diff"] == by_name["baseline"] + ["abs_diff"]
    assert by_name["self_rate_sq"] == by_name["baseline"] + ["self_rate", "self_rate_sq"]
    assert by_name["other_rate_sq"] == by_name["baseline"] + ["other_rate", "other_rate_sq"]
    assert by_name["abs_diff_sq"] == by_name["baseline"] + ["abs_diff", "abs_diff_sq"]


def test_add_timelock_columns_copies_and_casts_from_onset_offset() -> None:
    events = pd.DataFrame({"t_onset": [1, 2], "t_offset": [3, 4], "x": [10, 20]})

    out = add_timelock_columns(events)

    assert "self_IPU_onset" not in events.columns
    assert "other_IPU_offset" not in events.columns
    assert np.allclose(out["self_IPU_onset"].to_numpy(), [1.0, 2.0])
    assert np.allclose(out["other_IPU_offset"].to_numpy(), [3.0, 4.0])
    assert out["self_IPU_onset"].dtype.kind == "f"
    assert out["other_IPU_offset"].dtype.kind == "f"


def test_add_timelock_columns_requires_onset_and_offset() -> None:
    with pytest.raises(ValueError, match="t_onset"):
        add_timelock_columns(pd.DataFrame({"t_onset": [0.0]}))


def test_get_event_times_and_amps_impulse_uses_unit_amplitudes() -> None:
    reg = get_models()[0].regressors[0]  # self_onset_impulse
    events = pd.DataFrame({"self_IPU_onset": [0.1, 0.3], "unused": [5.0, 6.0]})

    times, amps = get_event_times_and_amps(events, reg)

    assert np.allclose(times, [0.1, 0.3])
    assert np.allclose(amps, [1.0, 1.0])


def test_get_event_times_and_amps_non_impulse_reads_amplitude_column() -> None:
    reg = self_rate_regressor()
    events = pd.DataFrame({"self_IPU_onset": [0.0, 0.2], "self_rate": [1.5, 2.5]})

    times, amps = get_event_times_and_amps(events, reg)

    assert np.allclose(times, [0.0, 0.2])
    assert np.allclose(amps, [1.5, 2.5])


def test_events_to_continuous_regressor_sums_collisions_and_clips_bounds() -> None:
    reg = RegressorSpec(
        name="x",
        column="amp",
        time_lock="self_IPU_onset",
        kind="self_rate",
    )
    events = pd.DataFrame(
        {
            "self_IPU_onset": [-5.0, 0.11, 0.12, 99.0],
            "amp": [1.0, 2.0, 3.0, 4.0],
        }
    )

    x = _events_to_continuous_regressor(
        events=events,
        reg=reg,
        sfreq_hz=10.0,
        n_samples=5,
        sample_t0_s=0.0,
    )

    assert x.shape == (5,)
    assert np.allclose(x, [1.0, 5.0, 0.0, 0.0, 4.0])


def test_events_to_continuous_regressor_drops_nonfinite_amplitudes_only() -> None:
    reg = RegressorSpec(
        name="x",
        column="amp",
        time_lock="self_IPU_onset",
        kind="self_rate",
    )
    events = pd.DataFrame({"self_IPU_onset": [0.0, 1.0, 2.0], "amp": [1.0, np.nan, np.inf]})

    x = _events_to_continuous_regressor(events=events, reg=reg, sfreq_hz=1.0, n_samples=4)

    assert np.allclose(x, [1.0, 0.0, 0.0, 0.0])


def test_events_to_continuous_regressor_rejects_nonfinite_times() -> None:
    reg = RegressorSpec(
        name="x",
        column="amp",
        time_lock="self_IPU_onset",
        kind="self_rate",
    )
    events = pd.DataFrame({"self_IPU_onset": [0.0, np.nan], "amp": [1.0, 2.0]})

    with pytest.raises(ValueError, match="Non-finite event times"):
        _events_to_continuous_regressor(events=events, reg=reg, sfreq_hz=1.0, n_samples=4)


def test_build_design_matrix_preserves_model_order_and_sample_t0_alignment() -> None:
    baseline = get_models()[0]
    events = pd.DataFrame(
        {
            "self_IPU_onset": [10.0],
            "other_IPU_offset": [11.0],
            "t_onset": [10.0],
            "t_offset": [11.0],
        }
    )
    emg = np.array([0.2, 0.4, 0.6, 0.8], dtype=float)

    X, names = build_design_matrix(
        events=events,
        model=baseline,
        sfreq_hz=1.0,
        n_samples=4,
        sample_t0_s=9.0,
        continuous_regressors={"emg": emg},
        return_feature_names=True,
    )

    assert names == ["self_onset_impulse", "other_offset_impulse", "emg"]
    assert np.allclose(X[:, 0], [0.0, 1.0, 0.0, 0.0])
    assert np.allclose(X[:, 1], [0.0, 0.0, 1.0, 0.0])
    assert np.allclose(X[:, 2], emg)


def test_build_design_matrix_requires_emg_values_when_model_contains_emg() -> None:
    with pytest.raises(ValueError, match="Missing continuous regressor values"):
        build_design_matrix(
            events=pd.DataFrame({"self_IPU_onset": [0.0], "other_IPU_offset": [0.0]}),
            model=get_models()[0],
            sfreq_hz=1.0,
            n_samples=3,
        )


def test_standardize_design_matrix_foldwise_uses_train_stats_and_handles_constant_cols() -> None:
    X_train = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]])
    X_test = np.array([[7.0, 10.0]])

    Xtr, Xte = standardize_design_matrix_foldwise(X_train, X_test)

    assert np.allclose(Xtr[:, 0], [-1.22474487, 0.0, 1.22474487])
    assert np.allclose(Xtr[:, 1], [0.0, 0.0, 0.0])
    assert np.allclose(Xte[0], [2.44948974, 0.0])


def test_utils_scoring_and_concatenation_contracts() -> None:
    assert pearsonr_1d(np.array([1, 2, 3]), np.array([2, 4, 6])) == pytest.approx(1.0)
    assert np.isnan(pearsonr_1d(np.array([1, 1, 1]), np.array([1, 2, 3])))

    y_true = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    y_pred = np.array([[2.0, 2.0], [4.0, 2.0], [6.0, 2.0]])
    # ch0 has r=1, ch1 is constant => NaN, nanmean should keep 1.0
    assert score_multichannel_mean_r(y_true, y_pred) == pytest.approx(1.0)

    out = concatenate_with_gap(
        [np.array([[1.0], [2.0]]), np.array([[3.0]])],
        gap_samples=2,
        fill_value=-1.0,
    )
    assert np.allclose(out[:, 0], [1.0, 2.0, -1.0, -1.0, 3.0])
    assert max_lag_samples(sfreq_hz=100.0, tmin_s=-0.2, tmax_s=0.81) == 81

