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
    return [_analysis_name(band, target_kind) for target_kind in _induced_target_kinds() for band in _induced_bands()]


def _induced_subject_outputs(_wildcards):
    return [
        f"{OUT_DIR}/{analysis}/subjects/{dyad_id}/{subject}/scores.parquet"
        for analysis in _induced_analyses()
        for dyad_id, subject in _subject_pairs()
    ]


rule induced_trf_subject:
    wildcard_constraints:
        analysis="induced_trf_.+"
    input:
        events=OUT_DIR + "/events/events.parquet",
        dataset_index=OUT_DIR + "/dataset/index_resampled.parquet",
        emg=OUT_DIR + "/features/emg/{dyad_id}/{subject}/emg_env.parquet",
        validate_config=OUT_DIR + "/_meta/config_validated.json",
        write_manifest=OUT_DIR + "/_meta/run_manifest.json",
    output:
        scores=OUT_DIR + "/{analysis}/subjects/{dyad_id}/{subject}/scores.parquet"
    threads: 1
    params:
        band=lambda wc: _analysis_parts(wc.analysis)[0],
        target_kind=lambda wc: _analysis_parts(wc.analysis)[1],
    log:
        OUT_DIR + "/logs/induced_trf/{analysis}_{dyad_id}_{subject}.log"
    shell:
        r"""
        set -euo pipefail
        MNE_TQDM=off MNE_LOGGING_LEVEL=WARNING PYTHONPATH=src python -u -m rate.cli.main induced-trf-subject \
            --dyad-id {wildcards.dyad_id} \
            --subject {wildcards.subject} \
            --band {params.band} \
            --target-kind {params.target_kind} \
            --config config/config.yaml \
            --schema config/schema.json \
            --out-dir {OUT_DIR} \
            2>&1 | tee {log}
        """


rule induced_trf_aggregate:
    input:
        OUT_DIR + "/events/events.parquet",
        OUT_DIR + "/dataset/index_resampled.parquet",
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
