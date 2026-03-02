from argparse import Namespace
from pathlib import Path
import importlib
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
from rate.trf.engine import TRFResult
from rate.trf.io_contract import write_trf_artifacts


def _require_parquet_or_skip(tmp_path: Path) -> None:
    try:
        pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "_probe.parquet", index=False)
    except Exception:
        pytest.skip("Parquet engine not available in test environment")
    (tmp_path / "_probe.parquet").unlink()


def _model_dir(root: Path, analysis: str, dyad: str = "dyad-01", subject: str = "sub-01", model: str = "baseline") -> Path:
    return root / analysis / "subjects" / dyad / subject / "models" / model


def _discover_subject_keys_with_model(root: Path, analysis: str, model: str) -> list[tuple[str, str]]:
    base = root / analysis / "subjects"
    out: set[tuple[str, str]] = set()
    for path in base.glob(f"*/*/models/{model}/kernels_outer.npy"):
        out.add((str(path.parents[3].name), str(path.parents[2].name)))
    return sorted(out)


def test_evoked_and_induced_analysis_layouts_are_independent(tmp_path: Path) -> None:
    evoked_dir = _model_dir(tmp_path, "evoked_trf")
    induced_dir = _model_dir(tmp_path, "induced_trf_theta")
    evoked_dir.mkdir(parents=True, exist_ok=True)
    induced_dir.mkdir(parents=True, exist_ok=True)
    np.save(evoked_dir / "kernels_outer.npy", np.zeros((1, 1, 1, 1), dtype=float))
    np.save(evoked_dir / "lags.npy", np.array([0.0], dtype=float))
    np.save(induced_dir / "kernels_outer.npy", np.zeros((1, 1, 1, 1), dtype=float))
    np.save(induced_dir / "lags.npy", np.array([0.0], dtype=float))

    assert _discover_subject_keys_with_model(tmp_path, "evoked_trf", "baseline") == [("dyad-01", "sub-01")]
    assert _discover_subject_keys_with_model(tmp_path, "induced_trf_theta", "baseline") == [("dyad-01", "sub-01")]


def test_induced_artifact_contract_includes_required_files(tmp_path: Path) -> None:
    _require_parquet_or_skip(tmp_path)
    model_dir = _model_dir(tmp_path, "induced_trf_theta")
    result = TRFResult(
        kernels_outer=np.zeros((2, 3, 4, 5), dtype=float),
        chosen_alpha=np.array([1.0, 10.0], dtype=float),
        outer_scores=pd.DataFrame({"alpha": [1.0], "r": [0.0]}),
        lags=np.linspace(-0.1, 0.2, 5),
        reconstruction_r2=pd.DataFrame({"channel": [0, 1], "r2": [0.1, 0.2]}),
    )
    write_trf_artifacts(result, model_dir)
    names = sorted(p.name for p in model_dir.iterdir())
    assert names == [
        "kernels_outer.npy",
        "lags.npy",
        "outer_scores.parquet",
        "reconstruction_metrics.parquet",
        "reconstruction_r2.parquet",
    ]


def test_stats_kernel0_runs_on_induced_output_without_modification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNE_HOME", str(tmp_path / ".mne"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    cmd_stats_kernel0 = importlib.import_module("rate.cli.commands.stats_kernel0")

    model_dir = _model_dir(tmp_path, "induced_trf_theta")
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(model_dir / "kernels_outer.npy", np.zeros((2, 4, 1, 3), dtype=float))
    np.save(model_dir / "lags.npy", np.array([-0.1, 0.0, 0.1], dtype=float))

    calls: dict[str, object] = {}

    class _Montage:
        ch_names = ["Fz", "Cz", "Pz", "Oz"]

    def fake_montage(_name: str):
        return _Montage()

    def fake_run_two_level_kernel0(**kwargs):
        calls["run_two_level_kernel0"] = kwargs
        return object()

    def fake_write_two_level_cluster_result(**kwargs):
        calls["write_two_level_cluster_result"] = kwargs
        return None

    monkeypatch.setattr(cmd_stats_kernel0, "make_standard_montage", fake_montage)
    monkeypatch.setattr(cmd_stats_kernel0, "run_two_level_kernel0", fake_run_two_level_kernel0)
    monkeypatch.setattr(cmd_stats_kernel0, "write_two_level_cluster_result", fake_write_two_level_cluster_result)

    cfg = {
        "paths": {"out_dir": str(tmp_path)},
        "project": {"random_seed": 0},
        "reporting": {"tables_format": "parquet"},
        "stats": {
                "kernel_cluster": {
                    "adjacency": {"montage_name": "standard_1020"},
                    "method": "threshold",
                "tail": 0,
                "seed": 0,
                "cluster": {"n_permutations": 8, "threshold_p": 0.05},
                "tfce": {"start": 0.0, "step": 0.2},
            }
        },
    }
    args = Namespace(
        analysis="induced_trf_theta",
        model="baseline",
        feature_index=0,
        feature_name="self_onset_impulse",
        config="config/config.yaml",
        schema="config/schema.json",
    )
    rc = cmd_stats_kernel0.run(args, config=cfg)
    assert rc == 0
    assert calls["run_two_level_kernel0"]["analysis"] == "induced_trf_theta"
    assert calls["run_two_level_kernel0"]["subject_keys"] == [("dyad-01", "sub-01")]
    assert "write_two_level_cluster_result" in calls


def test_stats_kernel0_prefers_subject_channel_names_for_induced_adjacency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MNE_HOME", str(tmp_path / ".mne"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    cmd_stats_kernel0 = importlib.import_module("rate.cli.commands.stats_kernel0")

    model_dir = _model_dir(tmp_path, "induced_trf_theta")
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(model_dir / "kernels_outer.npy", np.zeros((2, 4, 1, 3), dtype=float))
    np.save(model_dir / "lags.npy", np.array([-0.1, 0.0, 0.1], dtype=float))
    np.save(model_dir / "channel_names.npy", np.array(["Oz", "Pz", "Cz", "Fz"], dtype=str))

    calls: dict[str, object] = {}

    class _Montage:
        ch_names = ["Fz", "Cz", "Pz", "Oz"]

    monkeypatch.setattr(cmd_stats_kernel0, "make_standard_montage", lambda _name: _Montage())
    monkeypatch.setattr(
        cmd_stats_kernel0,
        "run_two_level_kernel0",
        lambda **kwargs: calls.setdefault("run_two_level_kernel0", kwargs) or object(),
    )
    monkeypatch.setattr(
        cmd_stats_kernel0,
        "write_two_level_cluster_result",
        lambda **kwargs: calls.setdefault("write_two_level_cluster_result", kwargs) or None,
    )

    cfg = {
        "paths": {"out_dir": str(tmp_path)},
        "project": {"random_seed": 0},
        "reporting": {"tables_format": "parquet"},
        "stats": {
            "kernel_cluster": {
                "adjacency": {"montage_name": "standard_1020"},
                "method": "threshold",
                "tail": 0,
                "seed": 0,
                "cluster": {"n_permutations": 8, "threshold_p": 0.05},
                "tfce": {"start": 0.0, "step": 0.2},
            }
        },
    }
    args = Namespace(
        analysis="induced_trf_theta",
        model="baseline",
        feature_index=0,
        feature_name="self_onset_impulse",
        config="config/config.yaml",
        schema="config/schema.json",
    )

    rc = cmd_stats_kernel0.run(args, config=cfg)
    assert rc == 0
    assert calls["run_two_level_kernel0"]["ch_names"] == ["Oz", "Pz", "Cz", "Fz"]


def _seed_kernel_subjects(
    root: Path,
    *,
    analysis: str,
    model: str,
    n_subjects: int = 4,
    ch_names: tuple[str, ...] = ("Fz", "Cz", "Pz", "Oz"),
) -> None:
    rng = np.random.default_rng(1234)
    lags = np.array([-0.1, 0.0, 0.1], dtype=float)

    for sidx in range(n_subjects):
        dyad_id = f"dyad-{sidx + 1:02d}"
        subject_id = f"sub-{sidx + 1:02d}"
        model_dir = _model_dir(root, analysis, dyad=dyad_id, subject=subject_id, model=model)
        model_dir.mkdir(parents=True, exist_ok=True)

        kernels = rng.normal(loc=0.0, scale=0.01, size=(4, len(ch_names), 1, len(lags)))
        # Inject the same clear positive spatiotemporal effect across analyses.
        kernels[:, 1:3, 0, 1:] += 0.5
        np.save(model_dir / "kernels_outer.npy", kernels.astype(float))
        np.save(model_dir / "lags.npy", lags)


def _run_real_kernel0_cli(
    *,
    tmp_path: Path,
    analysis: str,
    model: str,
    feature_name: str,
    feature_index: int,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    monkeypatch.setenv("MNE_HOME", str(tmp_path / ".mne"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    cmd_stats_kernel0 = importlib.import_module("rate.cli.commands.stats_kernel0")

    class _Montage:
        ch_names = ["Fz", "Cz", "Pz", "Oz"]

    monkeypatch.setattr(cmd_stats_kernel0, "make_standard_montage", lambda _name: _Montage())

    cfg = {
        "paths": {"out_dir": str(tmp_path)},
        "project": {"random_seed": 0},
        "reporting": {"tables_format": "csv"},
        "stats": {
            "kernel_cluster": {
                "adjacency": {"montage_name": "standard_1020"},
                "method": "threshold",
                "tail": 0,
                "seed": 0,
                "cluster": {"n_permutations": 8, "threshold_p": 0.05},
                "tfce": {"start": 0.0, "step": 0.2},
            }
        },
    }
    args = Namespace(
        analysis=analysis,
        model=model,
        feature_index=feature_index,
        feature_name=feature_name,
        config="config/config.yaml",
        schema="config/schema.json",
    )
    rc = cmd_stats_kernel0.run(args, config=cfg)
    assert rc == 0
    out_dir = tmp_path / "stats" / analysis / model / feature_name
    assert out_dir.exists()
    return out_dir


def test_stats_kernel0_evoked_and_induced_outputs_are_identical_given_same_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for analysis in ("evoked_trf", "induced_trf_theta"):
        _seed_kernel_subjects(tmp_path, analysis=analysis, model="self_rate")

    evoked_out = _run_real_kernel0_cli(
        tmp_path=tmp_path,
        analysis="evoked_trf",
        model="self_rate",
        feature_name="self_rate",
        feature_index=0,
        monkeypatch=monkeypatch,
    )
    induced_out = _run_real_kernel0_cli(
        tmp_path=tmp_path,
        analysis="induced_trf_theta",
        model="self_rate",
        feature_name="self_rate",
        feature_index=0,
        monkeypatch=monkeypatch,
    )

    expected = {
        "clusters.csv",
        "cluster_labels.npy",
        "t_obs.npy",
        "t_obs_map.npy",
        "h0.npy",
        "subject_z_maps.npy",
        "grand_average_trfs-ave.fif",
    }
    assert expected.issubset({p.name for p in evoked_out.iterdir()})
    assert expected.issubset({p.name for p in induced_out.iterdir()})

    for name in ("cluster_labels.npy", "t_obs.npy", "t_obs_map.npy", "h0.npy", "subject_z_maps.npy"):
        np.testing.assert_allclose(np.load(evoked_out / name), np.load(induced_out / name))

    pd.testing.assert_frame_equal(
        pd.read_csv(evoked_out / "clusters.csv"),
        pd.read_csv(induced_out / "clusters.csv"),
        check_dtype=False,
    )

    import mne

    evoked_ga = mne.read_evokeds(evoked_out / "grand_average_trfs-ave.fif")[0]
    induced_ga = mne.read_evokeds(induced_out / "grand_average_trfs-ave.fif")[0]
    np.testing.assert_allclose(evoked_ga.data, induced_ga.data)
    np.testing.assert_allclose(evoked_ga.times, induced_ga.times)
    assert evoked_ga.nave == induced_ga.nave


class _FakeEvoked:
    def __init__(self, data: np.ndarray, times: np.ndarray, ch_names: list[str]):
        self.data = np.asarray(data, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.ch_names = list(ch_names)
        self.info = {"ch_names": list(ch_names)}

    def copy(self):
        return _FakeEvoked(self.data.copy(), self.times.copy(), list(self.ch_names))

    def pick(self, picks: list[str]):
        idx = [self.ch_names.index(p) for p in picks]
        return _FakeEvoked(self.data[idx, :], self.times, [self.ch_names[i] for i in idx])


def _normalize_for_equality(value):
    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, value.shape, value.tolist())
    if isinstance(value, dict):
        return {k: _normalize_for_equality(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_normalize_for_equality(v) for v in value)
    return value


@pytest.mark.parametrize("figure", ["joint", "topo", "timecourse"])
def test_trf_figures_shared_modes_are_family_parity_for_evoked_and_induced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    figure: str,
) -> None:
    figures = importlib.import_module("rate.cli.commands.figures")
    viz_trf = importlib.import_module("rate.viz.trf")

    times = np.array([-0.1, 0.0, 0.1], dtype=float)
    fake = _FakeEvoked(
        data=np.array(
            [
                [0.1, 0.2, 0.3],
                [0.0, 0.1, 0.2],
                [-0.1, 0.0, 0.1],
                [0.2, 0.1, 0.0],
            ],
            dtype=float,
        ),
        times=times,
        ch_names=["Fz", "Cz", "Pz", "Oz"],
    )
    config = {
        "reporting": {
            "alpha": 0.05,
            "trf": {
                "regressors": ["self_rate", "other_rate"],
                "electrodes": ["Fz", "Pz"],
                "joint_times_s": {"default": [-0.1, 0.0, 0.1]},
            },
        }
    }

    monkeypatch.setattr(figures, "_load_group_evoked", lambda **kwargs: fake)
    monkeypatch.setattr(figures, "_load_cluster_intervals", lambda **kwargs: [(-0.1, 0.0)])
    monkeypatch.setattr(figures, "_load_topomap_stat_map", lambda **kwargs: fake.data.T.copy())
    monkeypatch.setattr(
        figures,
        "_load_topomap_significance_mask",
        lambda **kwargs: np.array(
            [
                [False, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
            ],
            dtype=bool,
        ),
    )
    monkeypatch.setattr(figures, "_load_condition_standard_error", lambda **kwargs: np.zeros_like(fake.data))

    calls: list[tuple[str, dict]] = []

    def _capture(name: str):
        def _inner(**kwargs):
            calls.append((name, kwargs))
            return None
        return _inner

    monkeypatch.setattr(viz_trf, "plot_joint_kernel", _capture("joint"))
    monkeypatch.setattr(viz_trf, "plot_stat_topomap_sequence", _capture("topo"))
    monkeypatch.setattr(viz_trf, "plot_electrode_timecourse_panel", _capture("timecourse"))

    normalized_calls: dict[str, dict] = {}
    for family in ("evoked_trf", "induced_trf_theta"):
        calls.clear()
        figures.run_trf_figures(
            out_dir=tmp_path,
            family=family,
            figure=figure,
            regressor="self_rate" if figure in ("joint", "topo") else None,
            config=config,
        )
        assert len(calls) == 1
        name, kwargs = calls[0]
        assert name == figure
        normalized = dict(kwargs)
        normalized["family"] = "<family>"
        if "title" in normalized and isinstance(normalized["title"], str):
            normalized["title"] = normalized["title"].replace(family, "<family>")
        normalized_calls[family] = _normalize_for_equality(normalized)

    assert normalized_calls["evoked_trf"] == normalized_calls["induced_trf_theta"]
