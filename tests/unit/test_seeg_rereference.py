"""Unit tests for sEEG rereferencing helpers and CLI."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np

from rate.cli.commands import seeg_rereference as cmd
from rate.seeg_preproc.rereference import RereferenceSpec, apply_rereference


def test_apply_rereference_monopolar_returns_copy() -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0]])

    rereferenced = apply_rereference(data, spec=RereferenceSpec(mode="monopolar"))

    assert np.array_equal(rereferenced, data)
    assert rereferenced is not data


def test_seeg_rereference_cli_monopolar_preserves_channels(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input_raw.fif"
    output_path = tmp_path / "output_raw.fif"
    input_path.write_text("", encoding="utf-8")

    original = np.array([[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]], dtype=float)
    saved: dict[str, object] = {}

    class FakeRaw:
        def __init__(self, data: np.ndarray, ch_names: list[str]) -> None:
            self._data = np.asarray(data, dtype=float)
            self.ch_names = list(ch_names)
            self.info = {"sfreq": 512.0}
            self.annotations = {"tag": "kept"}

        def get_data(self) -> np.ndarray:
            return self._data

        def copy(self) -> "FakeRaw":
            copied = FakeRaw(self._data.copy(), self.ch_names.copy())
            copied.info = dict(self.info)
            copied.annotations = self.annotations
            return copied

        def set_annotations(self, annotations) -> None:
            self.annotations = annotations

        def save(self, path, overwrite=False) -> None:
            saved["path"] = Path(path)
            saved["overwrite"] = overwrite
            saved["data"] = self._data.copy()
            saved["annotations"] = self.annotations
            saved["ch_names"] = self.ch_names.copy()

        def get_channel_types(self, picks=None):
            return ["seeg" for _ in (picks or [])]

    fake_raw = FakeRaw(original, ["A1", "A2"])

    class FakeMNE:
        class io:
            @staticmethod
            def read_raw_fif(path, preload=True, verbose=False):
                assert Path(path) == input_path
                assert preload is True
                return fake_raw

    monkeypatch.setattr(cmd, "mne", FakeMNE())

    args = Namespace(
        input=str(input_path),
        output=str(output_path),
        config="config/config.yaml",
        schema="config/schema.json",
        mode="monopolar",
        laplacian_neighbors=None,
        overwrite=True,
    )

    result = cmd.run(args, config={})

    assert result == 0
    assert saved["path"] == output_path
    assert saved["overwrite"] is True
    assert np.array_equal(saved["data"], original)
    assert saved["annotations"] == {"tag": "kept"}
    assert saved["ch_names"] == ["A1", "A2"]
