import pandas as pd

OUT_DIR = COMPUTE_OUT_DIR


def _compute_subject_pairs():
    ckpt = checkpoints.subject_targets.get()
    df = pd.read_csv(ckpt.output[0], sep="\t")
    return [(str(r.dyad_id), str(r.subject_id)) for r in df.itertuples(index=False)]


def _compute_induced_requested_targets():
    ind_cfg = compute_induced_trf_config
    models = ind_cfg.get("models", []) or []
    out = []
    seen = set()
    for spec in models:
        if not isinstance(spec, dict):
            continue
        for target in spec.get("targets", []) or []:
            name = str(target).strip().lower()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
    return out


def _compute_induced_aperiodic_targets():
    return [target for target in _compute_induced_requested_targets() if target in {"offset", "exponent"}]


def _compute_induced_target_outputs(_wildcards):
    return [
        f"{OUT_DIR}/induced_trf_{target}/subjects/{dyad_id}/{subject}/scores.parquet"
        for target in _compute_induced_requested_targets()
        for dyad_id, subject in _compute_subject_pairs()
    ]


def _compute_aperiodic_source_band():
    ind_cfg = compute_induced_trf_config
    residual_bands = [str(b).strip() for b in (ind_cfg.get("residual_bands") or []) if str(b).strip()]
    if residual_bands:
        return residual_bands[0]
    bands = [str(b).strip() for b in (ind_cfg.get("bands") or []) if str(b).strip()]
    if bands:
        return bands[0]
    bands_hz = ind_cfg.get("bands_hz", {}) or {}
    names = sorted(str(k).strip() for k in bands_hz.keys() if str(k).strip())
    if not names:
        raise ValueError("induced_trf requires at least one band definition")
    return names[0]


APERIODIC_SOURCE_BAND = _compute_aperiodic_source_band()


def _induced_subject_optional_inputs(wildcards):
    target = str(wildcards.target).strip().lower()
    if target not in set(_compute_induced_aperiodic_targets()):
        return []
    return [
        f"{OUT_DIR}/induced_trf_{APERIODIC_SOURCE_BAND}_residual/subjects/{wildcards.dyad_id}/{wildcards.subject}/targets/aperiodic/_done.txt"
    ]


rule induced_trf_sliding_fooof_subject:
    input:
        cfg_file="config/config.yaml",
        fooof_cfg="config/fooof.yaml",
        induced_cfg="config/induced_trf.yaml",
        dataset_index=OUT_DIR + "/dataset/index.parquet",
        events=OUT_DIR + "/events/events.parquet",
    output:
        done=f"{OUT_DIR}/induced_trf_{APERIODIC_SOURCE_BAND}_residual/subjects/{{dyad_id}}/{{subject}}/targets/aperiodic/_done.txt"
    log:
        f"{OUT_DIR}/logs/induced_trf/sliding_fooof_{APERIODIC_SOURCE_BAND}" + "_{dyad_id}_{subject}.log"
    shell:
        r"""
        set -euo pipefail
        MNE_TQDM=off MNE_LOGGING_LEVEL=WARNING PYTHONPATH=src python -u -m rate.cli.main induced-trf-sliding-fooof-subject \
            --dyad-id {wildcards.dyad_id} \
            --subject {wildcards.subject} \
            --band """ + APERIODIC_SOURCE_BAND + r""" \
            --config config/config.yaml \
            --schema config/schema.json \
            --out-dir {OUT_DIR} \
            --done-file {output.done} \
            2>&1 | tee {log}
        """


rule induced_trf_subject:
    input:
        events=OUT_DIR + "/events/events.parquet",
        dataset_index=OUT_DIR + "/dataset/index.parquet",
        emg=OUT_DIR + "/features/emg/{dyad_id}/{subject}/emg_env.parquet",
        validate_config=OUT_DIR + "/_meta/config_validated.json",
        write_manifest=OUT_DIR + "/_meta/run_manifest.json",
        aperiodic_done=_induced_subject_optional_inputs,
    output:
        scores=OUT_DIR + "/induced_trf_{target}/subjects/{dyad_id}/{subject}/scores.parquet"
    threads: 1
    log:
        OUT_DIR + "/logs/induced_trf/induced_trf_{target}_{dyad_id}_{subject}.log"
    shell:
        r"""
        set -euo pipefail
        MNE_TQDM=off MNE_LOGGING_LEVEL=WARNING PYTHONPATH=src python -u -m rate.cli.main induced-trf-subject \
            --dyad-id {wildcards.dyad_id} \
            --subject {wildcards.subject} \
            --target {wildcards.target} \
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
        _compute_induced_target_outputs
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
