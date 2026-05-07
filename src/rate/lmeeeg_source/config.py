"""Configuration helpers for the experimental source-space lmeEEG branch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_LMEEEG_SOURCE_SECTION = "lmeeeg_source"


@dataclass(frozen=True)
class SourceSpaceSettings:
    subjects_dir: Path
    fsaverage_subject: str
    spacing: str
    surface: str
    add_dist: bool
    subject_template: str
    subject_map: dict[str, str]


@dataclass(frozen=True)
class ForwardSettings:
    trans_pattern: str
    bem_solution_pattern: str
    mindist: float
    eeg: bool
    meg: bool


@dataclass(frozen=True)
class CovarianceSettings:
    method: str
    tmin: float | None
    tmax: float | None


@dataclass(frozen=True)
class InverseSettings:
    method: str
    loose: float
    depth: float
    snr: float
    pick_ori: str | None
    lambda2: float


@dataclass(frozen=True)
class FittingSettings:
    n_jobs: int
    batch_size_vertices: int
    verbose: bool
    progress: bool


@dataclass(frozen=True)
class OutputSettings:
    namespace: str
    prepared_dirname: str
    group_dirname: str
    qc_subdir: str


@dataclass(frozen=True)
class PlotSettings:
    predictors: tuple[str, ...]
    view: str
    hemi: str
    measure: str
    time_summary: str
    time_window: tuple[float, float] | None
    time_step_ms: float
    mask_by_significance: bool
    significance_alpha: float
    formats: tuple[str, ...]
    colormap: str


@dataclass(frozen=True)
class EpochSettings:
    section: str
    use_existing_margin_epochs: bool


@dataclass(frozen=True)
class LMEEEGSourceSettings:
    enabled: bool
    source_space: SourceSpaceSettings
    forward: ForwardSettings
    covariance: CovarianceSettings
    inverse: InverseSettings
    fitting: FittingSettings
    outputs: OutputSettings
    plotting: PlotSettings
    epochs: EpochSettings


def load_lmeeeg_source_settings(
    config: Mapping[str, Any],
    *,
    config_section: str = DEFAULT_LMEEEG_SOURCE_SECTION,
) -> LMEEEGSourceSettings:
    """Parse one source-space lmeEEG config section into typed settings."""
    section = dict(config.get(config_section) or {})
    source_space = dict(section.get("source_space") or {})
    forward = dict(section.get("forward") or {})
    covariance = dict(section.get("covariance") or {})
    inverse = dict(section.get("inverse") or {})
    fitting = dict(section.get("fitting") or {})
    outputs = dict(section.get("outputs") or {})
    plotting = dict(section.get("plotting") or {})
    epochs = dict(section.get("epochs") or {})

    subjects_dir_value = source_space.get("subjects_dir")
    if not subjects_dir_value:
        raise ValueError(
            f"`{config_section}.source_space.subjects_dir` must point to a FreeSurfer SUBJECTS_DIR for source reconstruction."
        )
    trans_pattern = forward.get("trans_pattern")
    if not trans_pattern:
        raise ValueError(f"`{config_section}.forward.trans_pattern` must be configured.")
    bem_pattern = forward.get("bem_solution_pattern")
    if not bem_pattern:
        raise ValueError(f"`{config_section}.forward.bem_solution_pattern` must be configured.")

    snr = float(inverse.get("snr", 3.0))
    lambda2 = inverse.get("lambda2")
    resolved_lambda2 = float(lambda2) if lambda2 is not None else 1.0 / (snr * snr)

    time_window = plotting.get("time_window")
    resolved_time_window: tuple[float, float] | None
    if time_window is None:
        resolved_time_window = None
    else:
        if not isinstance(time_window, (list, tuple)) or len(time_window) != 2:
            raise ValueError(f"`{config_section}.plotting.time_window` must be null or a [start, stop] pair.")
        resolved_time_window = (float(time_window[0]), float(time_window[1]))

    return LMEEEGSourceSettings(
        enabled=bool(section.get("enabled", False)),
        source_space=SourceSpaceSettings(
            subjects_dir=Path(str(subjects_dir_value)).expanduser(),
            fsaverage_subject=str(source_space.get("fsaverage", "fsaverage")),
            spacing=str(source_space.get("spacing", "ico2")),
            surface=str(source_space.get("surface", "white")),
            add_dist=bool(source_space.get("add_dist", False)),
            subject_template=str(source_space.get("subject_template", "{subject_id}")),
            subject_map={str(key): str(value) for key, value in dict(source_space.get("subject_map") or {}).items()},
        ),
        forward=ForwardSettings(
            trans_pattern=str(trans_pattern),
            bem_solution_pattern=str(bem_pattern),
            mindist=float(forward.get("mindist", 5.0)),
            eeg=bool(forward.get("eeg", True)),
            meg=bool(forward.get("meg", False)),
        ),
        covariance=CovarianceSettings(
            method=str(covariance.get("method", "empirical")),
            tmin=None if covariance.get("tmin") is None else float(covariance.get("tmin")),
            tmax=None if covariance.get("tmax") is None else float(covariance.get("tmax")),
        ),
        inverse=InverseSettings(
            method=str(inverse.get("method", "MNE")),
            loose=float(inverse.get("loose", 0.2)),
            depth=float(inverse.get("depth", 0.8)),
            snr=snr,
            pick_ori=None if inverse.get("pick_ori") is None else str(inverse.get("pick_ori")),
            lambda2=resolved_lambda2,
        ),
        fitting=FittingSettings(
            n_jobs=max(1, int(fitting.get("n_jobs", 1))),
            batch_size_vertices=max(1, int(fitting.get("batch_size_vertices", 25))),
            verbose=bool(fitting.get("verbose", True)),
            progress=bool(fitting.get("progress", True)),
        ),
        outputs=OutputSettings(
            namespace=str(outputs.get("namespace", config_section)),
            prepared_dirname=str(outputs.get("prepared_dirname", "prepared")),
            group_dirname=str(outputs.get("group_dirname", "group")),
            qc_subdir=str(outputs.get("qc_subdir", "qc/lmeeeg_source")),
        ),
        plotting=PlotSettings(
            predictors=tuple(str(value) for value in plotting.get("predictors", ["self_rate", "other_rate"])),
            view=str(plotting.get("view", "lateral")),
            hemi=str(plotting.get("hemi", "split")),
            measure=str(plotting.get("measure", "beta")),
            time_summary=str(plotting.get("time_summary", "mean")),
            time_window=resolved_time_window,
            time_step_ms=float(plotting.get("time_step_ms", 100.0)),
            mask_by_significance=bool(plotting.get("mask_by_significance", True)),
            significance_alpha=float(plotting.get("significance_alpha", 0.05)),
            formats=tuple(str(value) for value in plotting.get("formats", ["png", "pdf"])),
            colormap=str(plotting.get("colormap", "coolwarm")),
        ),
        epochs=EpochSettings(
            section=str(epochs.get("section", "margin_epochs" if epochs.get("use_existing_margin_epochs", True) else "epochs")),
            use_existing_margin_epochs=bool(epochs.get("use_existing_margin_epochs", True)),
        ),
    )


def resolve_freesurfer_subject(
    subject_id: str,
    settings: LMEEEGSourceSettings,
) -> str:
    """Resolve the FreeSurfer subject name for one project subject."""
    mapped = settings.source_space.subject_map.get(str(subject_id))
    if mapped:
        return mapped
    return settings.source_space.subject_template.format(subject_id=str(subject_id))
