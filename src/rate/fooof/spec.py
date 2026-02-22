from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class FooofSpec:
    window_length_s: float
    window_alignment: str  # e.g. "pre_self_onset"
    freq_range: tuple[float, float]
    peak_threshold: float
    max_n_peaks: int
    bands: dict[str, tuple[float, float]]  # theta, alpha, beta
    rois: dict[str, list[str]]
    window_tmin_s: float
    window_tmax_s: float
    spectrum_method: str
    welch_nperseg_s: float
    welch_overlap: float
    aperiodic_mode: str
    peak_width_limits_hz: tuple[float, float]
    min_peak_height: float
    n_jobs: int


def _as_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _read_rois_from_file(roi_file: str | Path) -> dict[str, list[str]]:
    path = Path(roi_file).expanduser()
    if not path.exists():
        raise ValueError(f"ROI file does not exist: {path}")

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read ROI file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in ROI file: {path}") from exc

    if payload is None:
        raise ValueError(f"ROI file is empty: {path}")

    if isinstance(payload, Mapping) and "rois" in payload:
        payload = payload["rois"]

    if not isinstance(payload, Mapping):
        raise ValueError(f"ROI file must contain a mapping of roi_name -> channel list: {path}")

    rois: dict[str, list[str]] = {}
    for roi_name, channels in payload.items():
        name = str(roi_name).strip()
        if not name:
            raise ValueError("ROI names must be non-empty strings.")
        if not isinstance(channels, list) or not channels:
            raise ValueError(f"ROI '{name}' must map to a non-empty list of channel names.")
        channel_list = [str(ch).strip() for ch in channels if str(ch).strip()]
        if not channel_list:
            raise ValueError(f"ROI '{name}' must contain at least one non-empty channel name.")
        rois[name] = channel_list
    return rois


def _build_bands(features_cfg: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    bands_raw = features_cfg.get("periodic_bands_hz")
    bands_cfg = _as_mapping(bands_raw, "fooof.features.periodic_bands_hz")

    required = ("theta", "alpha", "beta")
    out: dict[str, tuple[float, float]] = {}
    for band_name in required:
        if band_name not in bands_cfg:
            raise ValueError(f"Missing band definition: fooof.features.periodic_bands_hz.{band_name}")
        band_value = bands_cfg[band_name]
        if not isinstance(band_value, (list, tuple)) or len(band_value) != 2:
            raise ValueError(f"Band '{band_name}' must be a length-2 list/tuple [fmin, fmax].")
        fmin = float(band_value[0])
        fmax = float(band_value[1])
        if not (fmin < fmax):
            raise ValueError(f"Band '{band_name}' must satisfy fmin < fmax.")
        out[band_name] = (fmin, fmax)
    return out


def _build_rois(fooof_cfg: Mapping[str, Any]) -> dict[str, list[str]]:
    rois_section = _as_mapping(fooof_cfg.get("rois"), "fooof.rois")
    definition = str(rois_section.get("definition", "inline")).strip().lower()

    if definition == "file":
        roi_file = rois_section.get("roi_file")
        if not roi_file:
            raise ValueError("fooof.rois.roi_file is required when fooof.rois.definition='file'.")
        return _read_rois_from_file(str(roi_file))

    inline_rois = rois_section.get("values", rois_section.get("mapping"))
    if inline_rois is None:
        raise ValueError(
            "Inline ROI configuration requires fooof.rois.values (mapping of roi_name -> channel list)."
        )
    if not isinstance(inline_rois, Mapping):
        raise ValueError("fooof.rois.values must be a mapping of roi_name -> channel list.")

    rois: dict[str, list[str]] = {}
    for roi_name, channels in inline_rois.items():
        name = str(roi_name).strip()
        if not name:
            raise ValueError("ROI names must be non-empty strings.")
        if not isinstance(channels, list) or not channels:
            raise ValueError(f"ROI '{name}' must map to a non-empty list of channel names.")
        rois[name] = [str(ch).strip() for ch in channels if str(ch).strip()]
        if not rois[name]:
            raise ValueError(f"ROI '{name}' must contain at least one non-empty channel name.")
    return rois


def build_fooof_spec(config: dict) -> FooofSpec:
    """Extract and validate FOOOF feature-extraction settings from the main config."""
    root = _as_mapping(config, "config")
    fooof_cfg = _as_mapping(root.get("fooof"), "fooof")

    # Compatibility path for a flattened FOOOF config schema.
    flat_required = {
        "window_length_s",
        "window_alignment",
        "freq_range",
        "peak_threshold",
        "max_n_peaks",
        "bands",
        "rois",
    }
    if flat_required.issubset(set(fooof_cfg.keys())):
        window_length_s = float(fooof_cfg["window_length_s"])
        if window_length_s <= 0:
            raise ValueError("fooof.window_length_s must be > 0.")
        window_alignment = str(fooof_cfg["window_alignment"]).strip()
        if window_alignment != "pre_self_onset":
            raise ValueError(
                f"Unsupported fooof.window_alignment={window_alignment!r}; expected 'pre_self_onset'."
            )

        freq_raw = fooof_cfg["freq_range"]
        if not isinstance(freq_raw, (list, tuple)) or len(freq_raw) != 2:
            raise ValueError("fooof.freq_range must be [fmin, fmax].")
        freq_range = (float(freq_raw[0]), float(freq_raw[1]))
        if not (freq_range[0] < freq_range[1]):
            raise ValueError("fooof.freq_range must satisfy fmin < fmax.")

        bands_cfg = _as_mapping(fooof_cfg["bands"], "fooof.bands")
        bands: dict[str, tuple[float, float]] = {}
        for band_name in ("theta", "alpha", "beta"):
            band_val = bands_cfg.get(band_name)
            if not isinstance(band_val, (list, tuple)) or len(band_val) != 2:
                raise ValueError(f"fooof.bands.{band_name} must be [fmin, fmax].")
            bands[band_name] = (float(band_val[0]), float(band_val[1]))

        rois_cfg = _as_mapping(fooof_cfg["rois"], "fooof.rois")
        rois: dict[str, list[str]] = {}
        for roi_name, channels in rois_cfg.items():
            if not isinstance(channels, list) or not channels:
                raise ValueError(f"fooof.rois.{roi_name} must be a non-empty list of channels.")
            rois[str(roi_name)] = [str(ch).strip() for ch in channels if str(ch).strip()]
            if not rois[str(roi_name)]:
                raise ValueError(f"fooof.rois.{roi_name} contains no usable channel names.")

        n_jobs = int(fooof_cfg.get("n_jobs", 1))
        if n_jobs < 1:
            raise ValueError("fooof.n_jobs must be >= 1.")

        return FooofSpec(
            window_length_s=window_length_s,
            window_alignment=window_alignment,
            freq_range=freq_range,
            peak_threshold=float(fooof_cfg["peak_threshold"]),
            max_n_peaks=int(fooof_cfg["max_n_peaks"]),
            bands=bands,
            rois=rois,
            window_tmin_s=-window_length_s,
            window_tmax_s=0.0,
            spectrum_method="welch",
            welch_nperseg_s=window_length_s,
            welch_overlap=0.5,
            aperiodic_mode="fixed",
            peak_width_limits_hz=(1.0, 12.0),
            min_peak_height=0.0,
            n_jobs=n_jobs,
        )

    prespeech_cfg = _as_mapping(fooof_cfg.get("prespeech_window_s"), "fooof.prespeech_window_s")
    tmin = float(prespeech_cfg.get("tmin"))
    tmax = float(prespeech_cfg.get("tmax"))
    if not (tmin < tmax):
        raise ValueError("fooof.prespeech_window_s must satisfy tmin < tmax.")
    if tmax > 0:
        raise ValueError("fooof.prespeech_window_s.tmax must be <= 0 for a pre-speech window.")
    window_length_s = float(tmax - tmin)
    if window_length_s <= 0:
        raise ValueError("FOOOF window length must be > 0.")

    spectrum_cfg = _as_mapping(fooof_cfg.get("spectrum"), "fooof.spectrum")
    freq_range = (float(spectrum_cfg.get("fmin_hz")), float(spectrum_cfg.get("fmax_hz")))
    if not (freq_range[0] < freq_range[1]):
        raise ValueError("fooof.spectrum must satisfy fmin_hz < fmax_hz.")
    spectrum_method = str(spectrum_cfg.get("method", "welch")).strip().lower()
    if spectrum_method != "welch":
        raise ValueError(f"Unsupported fooof.spectrum.method={spectrum_method!r}; expected 'welch'.")
    welch_nperseg_s = float(spectrum_cfg.get("welch_nperseg_s", window_length_s))
    welch_overlap = float(spectrum_cfg.get("welch_overlap", 0.5))
    if welch_nperseg_s <= 0:
        raise ValueError("fooof.spectrum.welch_nperseg_s must be > 0.")
    if not (0 <= welch_overlap < 1):
        raise ValueError("fooof.spectrum.welch_overlap must be in [0, 1).")

    model_cfg = _as_mapping(fooof_cfg.get("model"), "fooof.model")
    max_n_peaks = int(model_cfg.get("max_n_peaks"))
    if max_n_peaks < 0:
        raise ValueError("fooof.model.max_n_peaks must be >= 0.")

    peak_width_limits_raw = model_cfg.get("peak_width_limits_hz", [1.0, 12.0])
    if not isinstance(peak_width_limits_raw, (list, tuple)) or len(peak_width_limits_raw) != 2:
        raise ValueError("fooof.model.peak_width_limits_hz must be [min_hz, max_hz].")
    peak_width_limits_hz = (float(peak_width_limits_raw[0]), float(peak_width_limits_raw[1]))
    if not (peak_width_limits_hz[0] < peak_width_limits_hz[1]):
        raise ValueError("fooof.model.peak_width_limits_hz must satisfy min_hz < max_hz.")

    min_peak_height = float(model_cfg.get("min_peak_height", 0.0))
    if min_peak_height < 0:
        raise ValueError("fooof.model.min_peak_height must be >= 0.")

    # Allow explicit peak_threshold; fall back to FOOOF default when absent.
    peak_threshold = float(model_cfg.get("peak_threshold", 2.0))
    if peak_threshold < 0:
        raise ValueError("fooof.model.peak_threshold must be >= 0.")

    use_knee = bool(model_cfg.get("knee", False))
    aperiodic_mode = "knee" if use_knee else "fixed"

    n_jobs = int(fooof_cfg.get("n_jobs", 1))
    if n_jobs < 1:
        raise ValueError("fooof.n_jobs must be >= 1.")

    features_cfg = _as_mapping(fooof_cfg.get("features"), "fooof.features")
    bands = _build_bands(features_cfg)
    rois = _build_rois(fooof_cfg)

    for band_name, (bmin, bmax) in bands.items():
        if bmin < freq_range[0] or bmax > freq_range[1]:
            raise ValueError(
                f"Band '{band_name}'={bands[band_name]} lies outside fooof.spectrum freq_range={freq_range}."
            )

    return FooofSpec(
        window_length_s=window_length_s,
        window_alignment="pre_self_onset",
        freq_range=freq_range,
        peak_threshold=peak_threshold,
        max_n_peaks=max_n_peaks,
        bands=bands,
        rois=rois,
        window_tmin_s=tmin,
        window_tmax_s=tmax,
        spectrum_method=spectrum_method,
        welch_nperseg_s=welch_nperseg_s,
        welch_overlap=welch_overlap,
        aperiodic_mode=aperiodic_mode,
        peak_width_limits_hz=peak_width_limits_hz,
        min_peak_height=min_peak_height,
        n_jobs=n_jobs,
    )
