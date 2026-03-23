# ==================================================================================================
#                                         STATS
# ==================================================================================================

def _stats_out_dir() -> str:
    # Read from stats.yaml (not Snakemake `config`) so compute.yaml changes do not
    # propagate to stats-only targets via implicit config tracking.
    return str(stats_config["paths"]["out_dir"])


def _stats_table_ext() -> str:
    tables_format = str(stats_config.get("reporting", {}).get("tables_format", "parquet")).lower()
    return "csv" if tables_format == "csv" else "parquet"


def _stats_induced_bands():
    ind_cfg = stats_induced_trf_config
    bands = ind_cfg.get("bands")
    if bands:
        return [str(b) for b in bands]
    bands_hz = ind_cfg.get("bands_hz", {}) or {}
    return sorted(str(k) for k in bands_hz.keys())


def _stats_induced_target_kinds():
    ind_cfg = stats_induced_trf_config
    kinds = ind_cfg.get("target_kinds")
    if kinds:
        return [str(k).strip().lower() for k in kinds]
    return ["raw"]


def _stats_induced_bands_for_target_kind(target_kind: str):
    base = _stats_induced_bands()
    mode = str(target_kind).strip().lower()
    if mode != "residual":
        return base
    ind_cfg = stats_induced_trf_config
    residual = ind_cfg.get("residual_bands")
    if residual is None:
        return base
    return [str(b).strip() for b in residual if str(b).strip() and str(b).strip() in set(base)]


def _stats_induced_analysis_name(band: str, target_kind: str) -> str:
    if str(target_kind).strip().lower() == "raw":
        return f"induced_trf_{str(band)}"
    return f"induced_trf_{str(band)}_{str(target_kind).strip().lower()}"


INDUCED_ANALYSES = [
    _stats_induced_analysis_name(band, target_kind)
    for target_kind in _stats_induced_target_kinds()
    for band in _stats_induced_bands_for_target_kind(target_kind)
]


KERNEL0_BEHAVIOR_FEATURES = ["self_rate", "other_rate", "alignment", "convergence"]


def _branch_behavior_feature_indexes(branch_cfg, branch_key):
    block = branch_cfg.get(branch_key, {}) or {}
    specs = block.get("models", []) or []
    out = {}
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        model_name = str(spec.get("name", "")).strip()
        regressors = [str(r).strip() for r in (spec.get("regressors", []) or [])]
        if not model_name:
            continue
        for feature in KERNEL0_BEHAVIOR_FEATURES:
            if feature in regressors and feature not in out:
                out[feature] = {
                    "model_name": model_name,
                    "feature_index": int(regressors.index(feature)),
                }
    return out


EVOKED_KERNEL0_FEATURE_MAP = _branch_behavior_feature_indexes(evoked_trf_config, "evoked_trf")
INDUCED_KERNEL0_FEATURE_MAP = _branch_behavior_feature_indexes(induced_trf_config, "induced_trf")


def _canonical_feature_name(value: str) -> str:
    token = str(value).strip()
    aliases = {
        "self_ipu_onset": "self_onset_impulse",
        "other_ipu_offset": "other_offset_impulse",
        "self_rate^2": "self_rate_sq",
        "other_rate^2": "other_rate_sq",
        "alignment^2": "alignment_sq",
        "convergence^2": "convergence_sq",
    }
    return aliases.get(token, token)


def _model_feature_index_from_cfg(branch_cfg, branch_key: str, model_name: str, feature_name: str):
    block = branch_cfg.get(branch_key, {}) or {}
    specs = block.get("models", []) or []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        if str(spec.get("name", "")).strip() != str(model_name).strip():
            continue
        regressors = [str(r).strip() for r in (spec.get("regressors", []) or [])]
        canonical = [_canonical_feature_name(r) for r in regressors]
        target = _canonical_feature_name(feature_name)
        if target in canonical:
            return int(canonical.index(target))
        return None
    return None


def _kernel0_feature_index(wildcards):
    feature_name = str(wildcards.feature_name)
    analysis = str(wildcards.analysis)
    feature_map = EVOKED_KERNEL0_FEATURE_MAP if analysis == "evoked_trf" else INDUCED_KERNEL0_FEATURE_MAP
    meta = feature_map.get(feature_name)
    if meta is None:
        raise ValueError(
            f"Cannot resolve feature_index for analysis={analysis} feature_name={feature_name}."
        )
    return int(meta["feature_index"])


def _kernel0_feature_name(wildcards):
    return str(wildcards.feature_name)


def _kernel0_scores_input(wildcards):
    analysis = str(wildcards.analysis)
    if analysis == "evoked_trf":
        return f"{_stats_out_dir()}/evoked_trf/scores/scores.parquet"
    if analysis.startswith("induced_trf_"):
        return f"{_stats_out_dir()}/induced_trf/scores/scores.parquet"
    raise ValueError(f"Unsupported analysis for stats_kernel0 rule: {analysis}")


KERNEL0_BEHAVIOR_TARGETS = [
    (
        f"{_stats_out_dir()}/stats/"
        f"evoked_trf/{EVOKED_KERNEL0_FEATURE_MAP[feature_name]['model_name']}/{feature_name}/clusters.{_stats_table_ext()}"
    )
    for feature_name in EVOKED_KERNEL0_FEATURE_MAP
]
KERNEL0_BEHAVIOR_TARGETS += [
    (
        f"{_stats_out_dir()}/stats/"
        f"{analysis}/{INDUCED_KERNEL0_FEATURE_MAP[feature_name]['model_name']}/{feature_name}/clusters.{_stats_table_ext()}"
    )
    for analysis in INDUCED_ANALYSES
    for feature_name in INDUCED_KERNEL0_FEATURE_MAP
]
KERNEL0_BEHAVIOR_SUBJECT_Z_TARGETS = [
    (
        f"{_stats_out_dir()}/stats/"
        "evoked_trf/"
        f"{EVOKED_KERNEL0_FEATURE_MAP[feature_name]['model_name']}/"
        f"{feature_name}/"
        f"subject_level/subject_z_map_index.{_stats_table_ext()}"
    )
    for feature_name in EVOKED_KERNEL0_FEATURE_MAP
]
KERNEL0_BEHAVIOR_SUBJECT_Z_TARGETS += [
    (
        f"{_stats_out_dir()}/stats/"
        f"{analysis}/"
        f"{INDUCED_KERNEL0_FEATURE_MAP[feature_name]['model_name']}/"
        f"{feature_name}/"
        f"subject_level/subject_z_map_index.{_stats_table_ext()}"
    )
    for analysis in INDUCED_ANALYSES
    for feature_name in INDUCED_KERNEL0_FEATURE_MAP
]


def _active_evoked_models(branch_cfg):
    block = branch_cfg.get("evoked_trf", {}) or {}
    specs = block.get("models", []) or []
    out = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name", "")).strip()
        if name:
            out.append(name)
    return out


EMG_ONSET_EVOKED_TARGET = (
    f"{_stats_out_dir()}/features/emg/grand_average_emg_self_onset-ave.fif"
)

rule stats_kernel0:
    input:
        index=f"{_stats_out_dir()}/dataset/index.parquet",
        scores=_kernel0_scores_input,
    output:
        clusters=(
            f"{_stats_out_dir()}/stats/"
            "{analysis}/{model}/{feature_name}/clusters." + _stats_table_ext()
        ),
        cluster_labels=(
            f"{_stats_out_dir()}/stats/"
            "{analysis}/{model}/{feature_name}/cluster_labels.npy"
        ),
        t_obs_map=(
            f"{_stats_out_dir()}/stats/"
            "{analysis}/{model}/{feature_name}/t_obs_map.npy"
        ),
        subject_z_maps=(
            f"{_stats_out_dir()}/stats/"
            "{analysis}/{model}/{feature_name}/subject_z_maps.npy"
        ),
        grand_average=(
            f"{_stats_out_dir()}/stats/"
            "{analysis}/{model}/{feature_name}/grand_average_trfs-ave.fif"
        ),
    params:
        feature_index=_kernel0_feature_index,
    shell:
        """
        python -m rate.cli.main stats-kernel0 \
            --analysis {wildcards.analysis} \
            --model {wildcards.model} \
            --feature-index {params.feature_index} \
            --feature-name {wildcards.feature_name} \
            --config config/stats.yaml \
            --schema config/schema.json
        """


rule stats_kernel0_behavior:
    input:
        KERNEL0_BEHAVIOR_TARGETS


rule stats_kernel0_subject_level:
    input:
        index=f"{_stats_out_dir()}/dataset/index.parquet",
        subject_z_maps=(
            f"{_stats_out_dir()}/stats/"
            "{analysis}/{model}/{feature_name}/subject_z_maps.npy"
        ),
    output:
        subject_z_index=(
            f"{_stats_out_dir()}/stats/"
            "{analysis}/{model}/{feature_name}/subject_level/"
            "subject_z_map_index." + _stats_table_ext()
        ),
    params:
        feature_index=_kernel0_feature_index,
    shell:
        """
        python -m rate.cli.main stats-export-subject-zmaps \
            --analysis {wildcards.analysis} \
            --model {wildcards.model} \
            --feature-index {params.feature_index} \
            --feature-name {wildcards.feature_name} \
            --config config/stats.yaml \
            --schema config/schema.json
        """


rule stats_kernel0_behavior_subject_level:
    input:
        KERNEL0_BEHAVIOR_SUBJECT_Z_TARGETS


rule stats_reconstruction_erp_r2_evoked:
    input:
        scores=f"{_stats_out_dir()}/evoked_trf/scores/scores.parquet",
    output:
        RECONSTRUCTION_ERP_R2_STAT_TARGETS
        + RECONSTRUCTION_ERP_R2_MEAN_TARGETS
        + RECONSTRUCTION_ERP_R2_LABEL_TARGETS
        + RECONSTRUCTION_ERP_R2_CHNAME_TARGETS
    shell:
        """
        python -m rate.cli.main stats-reconstruction-erp-r2 \
            --analysis evoked_trf \
            --config config/stats.yaml \
            --schema config/schema.json
        """


rule stats_evoked_emg_onset_by_subject_emg:
    input:
        dataset_index=f"{_stats_out_dir()}/dataset/index.parquet",
        events=f"{_stats_out_dir()}/events/events.parquet",
        emg_done=f"{_stats_out_dir()}/_meta/features_emg_done.txt",
    output:
        EMG_ONSET_EVOKED_TARGET
    shell:
        """
        python -m rate.cli.main stats-emg-onset-evoked \
            --analysis evoked_trf \
            --model baseline \
            --config config/stats.yaml \
            --schema config/schema.json
        """


rule stats_evoked_emg_onset:
    input:
        EMG_ONSET_EVOKED_TARGET
