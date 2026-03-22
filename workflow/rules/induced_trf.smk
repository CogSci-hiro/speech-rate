import pandas as pd

OUT_DIR = COMPUTE_OUT_DIR


def _subject_pairs():
    ckpt = checkpoints.subject_targets.get()
    df = pd.read_csv(ckpt.output[0], sep="\t")
    return [(str(r.dyad_id), str(r.subject_id)) for r in df.itertuples(index=False)]


def _induced_bands():
    ind_cfg = compute_induced_trf_config
    bands = ind_cfg.get("bands")
    if bands:
        return [str(b) for b in bands]
    bands_hz = ind_cfg.get("bands_hz", {}) or {}
    return sorted(str(k) for k in bands_hz.keys())


def _induced_target_kinds():
    ind_cfg = compute_induced_trf_config
    kinds = ind_cfg.get("target_kinds")
    if kinds:
        return [str(k).strip().lower() for k in kinds]
    return ["raw"]


def _induced_bands_for_target_kind(target_kind: str):
    base = _induced_bands()
    mode = str(target_kind).strip().lower()
    if mode != "residual":
        return base
    ind_cfg = compute_induced_trf_config
    residual = ind_cfg.get("residual_bands")
    if residual is None:
        return base
    return [str(b).strip() for b in residual if str(b).strip() and str(b).strip() in set(base)]


def _analysis_name(band: str, target_kind: str) -> str:
    band_name = str(band)
    mode = str(target_kind).strip().lower()
    if mode == "raw":
        return f"induced_trf_{band_name}"
    return f"induced_trf_{band_name}_{mode}"


def _analysis_parts(analysis: str):
    text = str(analysis)
    if not text.startswith("induced_trf_"):
        raise ValueError(f"Unsupported induced analysis name: {analysis}")
    suffix = text.removeprefix("induced_trf_")
    if suffix.endswith("_residual"):
        return suffix[: -len("_residual")], "residual"
    return suffix, "raw"


def _induced_analyses():
    return [
        _analysis_name(band, target_kind)
        for target_kind in _induced_target_kinds()
        for band in _induced_bands_for_target_kind(target_kind)
    ]


def _induced_subject_outputs(_wildcards):
    return [
        f"{OUT_DIR}/{analysis}/subjects/{dyad_id}/{subject}/scores.parquet"
        for analysis in _induced_analyses()
        for dyad_id, subject in _subject_pairs()
    ]


def _band_hz(band: str):
    ind_cfg = compute_induced_trf_config
    bands_hz = ind_cfg.get("bands_hz", {}) or {}
    rng = bands_hz.get(str(band))
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        raise ValueError(f"Missing induced_trf.bands_hz entry for band={band!r}")
    return float(rng[0]), float(rng[1])


rule induced_trf_sliding_fooof_subject:
    input:
        config="config/fooof.yaml",
        dataset_index=OUT_DIR + "/dataset/index.parquet",
        events=OUT_DIR + "/events/events.parquet",
    output:
        done=OUT_DIR + "/induced_trf_{band}_residual/subjects/{dyad_id}/{subject}/targets/aperiodic/_done.txt"
    params:
        band_fmin=lambda wc: _band_hz(wc.band)[0],
        band_fmax=lambda wc: _band_hz(wc.band)[1],
    log:
        OUT_DIR + "/logs/induced_trf/sliding_fooof_{band}_{dyad_id}_{subject}.log"
    shell:
        r"""
        set -euo pipefail
        MNE_TQDM=off MNE_LOGGING_LEVEL=WARNING PYTHONPATH=src python -u -m rate.cli.main induced-trf-sliding-fooof-subject \
            --dyad-id {wildcards.dyad_id} \
            --subject {wildcards.subject} \
            --band {wildcards.band} \
            --band-fmin {params.band_fmin} \
            --band-fmax {params.band_fmax} \
            --config config/fooof.yaml \
            --out-dir {OUT_DIR} \
            --done-file {output.done} \
            2>&1 | tee {log}
        """


rule induced_trf_subject_raw:
    input:
        events=OUT_DIR + "/events/events.parquet",
        dataset_index=OUT_DIR + "/dataset/index.parquet",
        emg=OUT_DIR + "/features/emg/{dyad_id}/{subject}/emg_env.parquet",
        validate_config=OUT_DIR + "/_meta/config_validated.json",
        write_manifest=OUT_DIR + "/_meta/run_manifest.json",
    output:
        scores=OUT_DIR + "/induced_trf_{band}/subjects/{dyad_id}/{subject}/scores.parquet"
    threads: 1
    log:
        OUT_DIR + "/logs/induced_trf/induced_trf_{band}_{dyad_id}_{subject}.log"
    shell:
        r"""
        set -euo pipefail
        MNE_TQDM=off MNE_LOGGING_LEVEL=WARNING PYTHONPATH=src python -u -m rate.cli.main induced-trf-subject \
            --dyad-id {wildcards.dyad_id} \
            --subject {wildcards.subject} \
            --band {wildcards.band} \
            --target-kind raw \
            --config config/config.yaml \
            --schema config/schema.json \
            --out-dir {OUT_DIR} \
            2>&1 | tee {log}
        """


rule induced_trf_subject_residual:
    input:
        events=OUT_DIR + "/events/events.parquet",
        dataset_index=OUT_DIR + "/dataset/index.parquet",
        emg=OUT_DIR + "/features/emg/{dyad_id}/{subject}/emg_env.parquet",
        validate_config=OUT_DIR + "/_meta/config_validated.json",
        write_manifest=OUT_DIR + "/_meta/run_manifest.json",
        sliding_done=OUT_DIR + "/induced_trf_{band}_residual/subjects/{dyad_id}/{subject}/targets/aperiodic/_done.txt",
    output:
        scores=OUT_DIR + "/induced_trf_{band}_residual/subjects/{dyad_id}/{subject}/scores.parquet"
    threads: 1
    log:
        OUT_DIR + "/logs/induced_trf/induced_trf_{band}_residual_{dyad_id}_{subject}.log"
    shell:
        r"""
        set -euo pipefail
        MNE_TQDM=off MNE_LOGGING_LEVEL=WARNING PYTHONPATH=src python -u -m rate.cli.main induced-trf-subject \
            --dyad-id {wildcards.dyad_id} \
            --subject {wildcards.subject} \
            --band {wildcards.band} \
            --target-kind residual \
            --config config/config.yaml \
            --schema config/schema.json \
            --out-dir {OUT_DIR} \
            2>&1 | tee {log}
        """


rule induced_trf_aggregate:
    input:
        OUT_DIR + "/events/events.parquet",
        OUT_DIR + "/dataset/index.parquet",
        OUT_DIR + "/_meta/features_emg_done.txt",
        OUT_DIR + "/_meta/config_validated.json",
        OUT_DIR + "/_meta/run_manifest.json",
        _induced_subject_outputs
    output:
        OUT_DIR + "/induced_trf/scores/scores.parquet"
    log:
        OUT_DIR + "/logs/induced_trf/aggregate.log"
    shell:
        r"""
        set -euo pipefail
        MNE_TQDM=off MNE_LOGGING_LEVEL=WARNING PYTHONPATH=src python -u -m rate.cli.main induced-trf-aggregate \
            --config config/config.yaml \
            --schema config/schema.json \
            --out-dir {OUT_DIR} \
            2>&1 | tee {log}
        """
