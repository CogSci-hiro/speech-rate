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
