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


def _stats_induced_targets():
    ind_cfg = stats_induced_trf_config
    targets = ind_cfg.get("targets")
    if targets:
        return [str(target).strip().lower() for target in targets if str(target).strip()]
    bands = ind_cfg.get("bands")
    if bands:
        return [str(band).strip().lower() for band in bands if str(band).strip()]
    bands_hz = ind_cfg.get("bands_hz", {}) or {}
    return sorted(str(k).strip().lower() for k in bands_hz.keys() if str(k).strip())


INDUCED_ANALYSES = [
    f"induced_trf_{target}"
    for target in _stats_induced_targets()
]


KERNEL0_EXCLUDED_FEATURES = {
    "self_onset_impulse",
    "other_offset_impulse",
    "emg",
    "run",
    "time_within_run",
}


def _branch_behavior_feature_indexes(branch_cfg, branch_key):
    def _normalize_feature(value):
        token = str(value).strip()
        if "*" in token:
            parts = [part.strip() for part in token.split("*", 1)]
            if len(parts) == 2 and all(parts):
                return f"{_normalize_feature(parts[0])}__x__{_normalize_feature(parts[1])}"
        aliases = {
            "self_ipu_onset": "self_onset_impulse",
            "other_ipu_offset": "other_offset_impulse",
            "self_rate^2": "self_rate_sq",
            "other_rate^2": "other_rate_sq",
            "alignment^2": "alignment_sq",
            "convergence^2": "convergence_sq",
            "mean_rate^2": "mean_rate_sq",
        }
        return aliases.get(token, token)

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
        canonical = [_normalize_feature(feature) for feature in regressors]
        for idx, feature in enumerate(canonical):
            if not feature or feature in KERNEL0_EXCLUDED_FEATURES or feature in out:
                continue
            out[feature] = {
                    "model_name": model_name,
                    "feature_index": int(idx),
                }
    return out


EVOKED_KERNEL0_FEATURE_MAP = _branch_behavior_feature_indexes(evoked_trf_config, "evoked_trf")
INDUCED_KERNEL0_FEATURE_MAP = _branch_behavior_feature_indexes(induced_trf_config, "induced_trf")


def _canonical_feature_name(value: str) -> str:
    token = str(value).strip()
    if "*" in token:
        parts = [part.strip() for part in token.split("*", 1)]
        if len(parts) == 2 and all(parts):
            return f"{_canonical_feature_name(parts[0])}__x__{_canonical_feature_name(parts[1])}"
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
    model_name = str(wildcards.model)
    branch_cfg = evoked_trf_config if analysis == "evoked_trf" else induced_trf_config
    branch_key = "evoked_trf" if analysis == "evoked_trf" else "induced_trf"
    direct = _model_feature_index_from_cfg(branch_cfg, branch_key, model_name=model_name, feature_name=feature_name)
    if direct is not None:
        return int(direct)
    feature_map = EVOKED_KERNEL0_FEATURE_MAP if analysis == "evoked_trf" else INDUCED_KERNEL0_FEATURE_MAP
    meta = feature_map.get(feature_name)
    if meta is None:
        raise ValueError(
            f"Cannot resolve feature_index for analysis={analysis} model={model_name} feature_name={feature_name}."
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
