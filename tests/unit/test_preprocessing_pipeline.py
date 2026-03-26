import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import mne
import numpy as np
import pandas as pd
import pytest

from rate.preprocessing.bad_channels import load_bad_channels_from_bids_tsv
from rate.preprocessing.channels import apply_bids_channel_types
from rate.preprocessing.pipeline import preprocess_run


def _make_raw() -> mne.io.RawArray:
    sfreq = 200.0
    channel_names = ["Fz", "Cz", "Pz", "Oz", "EMG1", "MISC1"]
    channel_types = ["eeg", "eeg", "eeg", "eeg", "emg", "misc"]
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)
    rng = np.random.default_rng(7)
    data = rng.normal(scale=1e-6, size=(len(channel_names), int(sfreq * 2.0)))
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(mne.Annotations(onset=[0.5], duration=[0.0], description=["stim/a"]))
    return raw


def _write_channels_tsv(path: Path, *, bad_names: list[str]) -> None:
    table = pd.DataFrame(
        {
            "name": ["Fz", "Cz", "Pz", "Oz", "EMG1"],
            "type": ["EEG", "EEG", "EEG", "EEG", "EMG"],
            "status": ["bad" if name in bad_names else "good" for name in ["Fz", "Cz", "Pz", "Oz", "EMG1"]],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def _write_channels_tsv_with_types(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _config(tmp_path: Path) -> dict:
    return {
        "paths": {
            "out_dir": str(tmp_path / "out"),
            "bids_dir": str(tmp_path / "bids"),
            "ica_dir": str(tmp_path / "ica"),
            "annotations_dir": str(tmp_path / "annotations"),
        },
        "preprocessing": {
            "channels": {
                "emg_names": ["EMG1"],
                "emg_patterns": [],
                "drop_names": ["MISC1"],
            },
            "montage": {"name": "standard_1020", "on_missing": "ignore"},
            "events": {},
            "downsample": {"sfreq_hz": 100.0},
            "filter": {"l_freq_hz": 1.0, "h_freq_hz": 40.0},
            "bad_channels": {},
            "ica": {},
            "output": {
                "eeg_filename": "eeg_preprocessed_raw.fif",
                "emg_filename": "emg.npz",
                "events_filename": "events.tsv",
                "summary_filename": "preprocessing_summary.json",
            },
        },
    }


def test_preprocess_run_writes_eeg_emg_events_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _make_raw()
    raw_path = tmp_path / "input_raw.fif"
    raw.save(raw_path, overwrite=True, verbose="ERROR")

    channels_tsv_path = tmp_path / "sub-001_task-conversation_run-01_channels.tsv"
    _write_channels_tsv(channels_tsv_path, bad_names=["Fz"])

    eeg_output_path = tmp_path / "out" / "eeg_preprocessed_raw.fif"
    emg_output_path = tmp_path / "out" / "emg.npz"
    events_output_path = tmp_path / "out" / "events.tsv"
    summary_output_path = tmp_path / "out" / "preprocessing_summary.json"

    row = {
        "dyad_id": "dyad-001",
        "subject_id": "sub-001",
        "task": "conversation",
        "run": "1",
        "raw_eeg_path": str(raw_path),
        "eeg_path": str(tmp_path / "out" / "preprocessing" / "sub-001" / "task-conversation" / "run-1" / "eeg_preprocessed_raw.fif"),
    }

    monkeypatch.setattr("rate.preprocessing.pipeline.apply_precomputed_ica", lambda raw, path: None)
    monkeypatch.setattr("rate.preprocessing.pipeline.find_run_anchor_event", lambda raw, config: (100, 0.5))

    preprocess_run(
        _config(tmp_path),
        row,
        raw_eeg_path=raw_path,
        channels_tsv_path=channels_tsv_path,
        ica_path=tmp_path / "precomputed-ica.fif",
        eeg_output_path=eeg_output_path,
        emg_output_path=emg_output_path,
        events_output_path=events_output_path,
        summary_output_path=summary_output_path,
    )

    eeg_raw = mne.io.read_raw_fif(eeg_output_path, preload=False, verbose="ERROR")
    assert eeg_raw.ch_names == ["Fz", "Cz", "Pz", "Oz"]
    assert eeg_raw.get_channel_types() == ["eeg", "eeg", "eeg", "eeg"]
    assert eeg_raw.info["sfreq"] == pytest.approx(100.0)

    emg_payload = np.load(emg_output_path, allow_pickle=False)
    assert list(emg_payload["channel_names"]) == ["EMG1"]

    events_df = pd.read_csv(events_output_path, sep="\t")
    assert len(events_df) == 1
    assert {"sample", "time_s", "previous", "event_id"}.issubset(events_df.columns)

    summary = json.loads(summary_output_path.read_text(encoding="utf-8"))
    assert summary["subject_id"] == "sub-001"
    assert "MISC1" in summary["dropped_channels"]
    assert summary["emg_channels"] == ["EMG1"]
    assert summary["anchor_event_sample"] == 100
    assert summary["anchor_event_time_s"] == pytest.approx(0.5)


def test_load_bad_channels_from_bids_tsv_raises_on_partial_name_mismatch(tmp_path: Path) -> None:
    raw = _make_raw().pick(["Fz", "Cz", "Pz", "Oz"])
    channels_tsv_path = tmp_path / "channels.tsv"
    table = pd.DataFrame(
        {
            "name": ["Fz", "NotAChannel"],
            "status": ["bad", "bad"],
        }
    )
    table.to_csv(channels_tsv_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="were not found in raw data"):
        load_bad_channels_from_bids_tsv(channels_tsv_path, raw=raw)


def test_apply_bids_channel_types_overrides_raw_types(tmp_path: Path) -> None:
    raw = mne.io.RawArray(
        np.random.default_rng(5).normal(scale=1e-6, size=(4, 400)),
        mne.create_info(
            ch_names=["Fz", "Cz", "rEAR", "lEAR"],
            sfreq=200.0,
            ch_types=["eeg", "eeg", "eeg", "eeg"],
        ),
        verbose="ERROR",
    )
    channels_tsv_path = tmp_path / "channels.tsv"
    _write_channels_tsv_with_types(
        channels_tsv_path,
        rows=[
            {"name": "Fz", "type": "EEG", "status": "good"},
            {"name": "Cz", "type": "EEG", "status": "good"},
            {"name": "rEAR", "type": "REF", "status": "good"},
            {"name": "lEAR", "type": "REF", "status": "good"},
        ],
    )

    apply_bids_channel_types(raw, channels_tsv_path)

    assert raw.get_channel_types() == ["eeg", "eeg", "misc", "misc"]


def test_apply_bids_channel_types_demotes_ear_references_even_if_labeled_eeg(tmp_path: Path) -> None:
    raw = mne.io.RawArray(
        np.random.default_rng(6).normal(scale=1e-6, size=(4, 400)),
        mne.create_info(
            ch_names=["Fz", "Cz", "rEAR", "lEAR"],
            sfreq=200.0,
            ch_types=["eeg", "eeg", "eeg", "eeg"],
        ),
        verbose="ERROR",
    )
    channels_tsv_path = tmp_path / "channels.tsv"
    _write_channels_tsv_with_types(
        channels_tsv_path,
        rows=[
            {"name": "Fz", "type": "EEG", "status": "good"},
            {"name": "Cz", "type": "EEG", "status": "good"},
            {"name": "rEAR", "type": "EEG", "status": "good"},
            {"name": "lEAR", "type": "EEG", "status": "good"},
        ],
    )

    apply_bids_channel_types(raw, channels_tsv_path)

    assert raw.get_channel_types() == ["eeg", "eeg", "misc", "misc"]


def test_preprocess_run_drops_non_eeg_channels_from_bids_types(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = mne.io.RawArray(
        np.random.default_rng(9).normal(scale=1e-6, size=(6, 400)),
        mne.create_info(
            ch_names=["Fz", "Cz", "Pz", "Oz", "rEAR", "lEAR"],
            sfreq=200.0,
            ch_types=["eeg", "eeg", "eeg", "eeg", "eeg", "eeg"],
        ),
        verbose="ERROR",
    )
    raw.set_annotations(mne.Annotations(onset=[0.5], duration=[0.0], description=["stim/a"]))
    raw_path = tmp_path / "input_raw.fif"
    raw.save(raw_path, overwrite=True, verbose="ERROR")

    channels_tsv_path = tmp_path / "sub-001_task-conversation_run-01_channels.tsv"
    _write_channels_tsv_with_types(
        channels_tsv_path,
        rows=[
            {"name": "Fz", "type": "EEG", "status": "bad"},
            {"name": "Cz", "type": "EEG", "status": "good"},
            {"name": "Pz", "type": "EEG", "status": "good"},
            {"name": "Oz", "type": "EEG", "status": "good"},
            {"name": "rEAR", "type": "REF", "status": "good"},
            {"name": "lEAR", "type": "REF", "status": "good"},
        ],
    )

    row = {
        "dyad_id": "dyad-001",
        "subject_id": "sub-001",
        "task": "conversation",
        "run": "1",
        "raw_eeg_path": str(raw_path),
        "eeg_path": str(tmp_path / "out" / "preprocessing" / "sub-001" / "task-conversation" / "run-1" / "eeg_preprocessed_raw.fif"),
    }

    monkeypatch.setattr("rate.preprocessing.pipeline.apply_precomputed_ica", lambda raw, path: None)
    monkeypatch.setattr("rate.preprocessing.pipeline.find_run_anchor_event", lambda raw, config: (100, 0.5))

    preprocess_run(
        _config(tmp_path),
        row,
        raw_eeg_path=raw_path,
        channels_tsv_path=channels_tsv_path,
        ica_path=tmp_path / "precomputed-ica.fif",
        eeg_output_path=tmp_path / "out" / "eeg_preprocessed_raw.fif",
        emg_output_path=tmp_path / "out" / "emg.npz",
        events_output_path=tmp_path / "out" / "events.tsv",
        summary_output_path=tmp_path / "out" / "preprocessing_summary.json",
    )

    eeg_raw = mne.io.read_raw_fif(tmp_path / "out" / "eeg_preprocessed_raw.fif", preload=False, verbose="ERROR")
    assert eeg_raw.ch_names == ["Fz", "Cz", "Pz", "Oz"]


def test_interpolate_bad_channels_raises_for_eeg_channels_without_positions() -> None:
    raw = mne.io.RawArray(
        np.random.default_rng(11).normal(scale=1e-6, size=(5, 400)),
        mne.create_info(
            ch_names=["Fz", "Cz", "Pz", "Oz", "rEAR"],
            sfreq=200.0,
            ch_types=["eeg", "eeg", "eeg", "eeg", "eeg"],
        ),
        verbose="ERROR",
    )
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    raw.info["bads"] = ["rEAR"]

    with pytest.raises(ValueError, match="without finite sensor positions"):
        from rate.preprocessing.bad_channels import interpolate_bad_channels

        interpolate_bad_channels(raw)

    assert raw.info["bads"] == ["rEAR"]


def test_interpolate_bad_channels_ignores_non_bad_channels_without_positions() -> None:
    from rate.preprocessing.bad_channels import interpolate_bad_channels

    raw = mne.io.RawArray(
        np.random.default_rng(12).normal(scale=1e-6, size=(5, 400)),
        mne.create_info(
            ch_names=["Fz", "Cz", "Pz", "Oz", "rEAR"],
            sfreq=200.0,
            ch_types=["eeg", "eeg", "eeg", "eeg", "eeg"],
        ),
        verbose="ERROR",
    )
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    raw.info["bads"] = ["Fz"]

    interpolate_bad_channels(raw)

    assert raw.info["bads"] == []


def test_load_bad_channels_from_bids_tsv_allows_no_bad_rows(tmp_path: Path) -> None:
    raw = _make_raw().pick(["Fz", "Cz", "Pz", "Oz"])
    channels_tsv_path = tmp_path / "channels.tsv"
    pd.DataFrame(
        {
            "name": ["Fz", "Cz"],
            "status": ["good", "good"],
        }
    ).to_csv(channels_tsv_path, sep="\t", index=False)

    bads = load_bad_channels_from_bids_tsv(channels_tsv_path, raw=raw)

    assert bads == []
    assert raw.info["bads"] == []


def test_interpolate_bad_channels_allows_no_bads_marked() -> None:
    from rate.preprocessing.bad_channels import interpolate_bad_channels

    raw = _make_raw().pick(["Fz", "Cz", "Pz", "Oz"])
    raw.info["bads"] = []

    interpolate_bad_channels(raw)

    assert raw.info["bads"] == []


def test_preprocess_run_raises_on_missing_ica_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _make_raw()
    raw_path = tmp_path / "input_raw.fif"
    raw.save(raw_path, overwrite=True, verbose="ERROR")

    channels_tsv_path = tmp_path / "sub-001_task-conversation_run-01_channels.tsv"
    _write_channels_tsv(channels_tsv_path, bad_names=["Fz"])

    row = {
        "dyad_id": "dyad-001",
        "subject_id": "sub-001",
        "task": "conversation",
        "run": "1",
        "raw_eeg_path": str(raw_path),
        "eeg_path": str(tmp_path / "out" / "preprocessing" / "sub-001" / "task-conversation" / "run-1" / "eeg_preprocessed_raw.fif"),
    }

    monkeypatch.setattr("rate.preprocessing.pipeline.find_run_anchor_event", lambda raw, config: (100, 0.5))

    with pytest.raises(FileNotFoundError, match="Precomputed ICA file does not exist"):
        preprocess_run(
            _config(tmp_path),
            row,
            raw_eeg_path=raw_path,
            channels_tsv_path=channels_tsv_path,
            ica_path=tmp_path / "missing-ica.fif",
            eeg_output_path=tmp_path / "out" / "eeg_preprocessed_raw.fif",
            emg_output_path=tmp_path / "out" / "emg.npz",
            events_output_path=tmp_path / "out" / "events.tsv",
            summary_output_path=tmp_path / "out" / "preprocessing_summary.json",
        )
