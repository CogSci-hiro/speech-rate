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


def _induced_bands():
    ind_cfg = stats_induced_trf_config
    bands = ind_cfg.get("bands")
    if bands:
        return [str(b) for b in bands]
    bands_hz = ind_cfg.get("bands_hz", {}) or {}
    return sorted(str(k) for k in bands_hz.keys())


def _induced_target_kinds():
    ind_cfg = stats_induced_trf_config
    kinds = ind_cfg.get("target_kinds")
    if kinds:
        return [str(k).strip().lower() for k in kinds]
    return ["raw"]


def _induced_bands_for_target_kind(target_kind: str):
    base = _induced_bands()
    mode = str(target_kind).strip().lower()
    if mode != "residual":
        return base
    ind_cfg = stats_induced_trf_config
    residual = ind_cfg.get("residual_bands")
    if residual is None:
        return base
    return [str(b).strip() for b in residual if str(b).strip() and str(b).strip() in set(base)]


def _induced_analysis_name(band: str, target_kind: str) -> str:
    if str(target_kind).strip().lower() == "raw":
        return f"induced_trf_{str(band)}"
    return f"induced_trf_{str(band)}_{str(target_kind).strip().lower()}"


INDUCED_ANALYSES = [
    _induced_analysis_name(band, target_kind)
    for target_kind in _induced_target_kinds()
    for band in _induced_bands_for_target_kind(target_kind)
]


def _branch_behavior_model_features(branch_cfg, branch_key):
    block = branch_cfg.get(branch_key, {}) or {}
    specs = block.get("models", []) or []
    out = {}
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        model_name = str(spec.get("name", "")).strip()
        if not model_name or model_name == "baseline":
            continue
        regressors = [str(r) for r in (spec.get("regressors", []) or [])]
        if not regressors:
            continue

        target_regressor = model_name
        if model_name.endswith("_sq"):
            base = model_name[:-3]
            squared_name = f"{base}^2"
            if squared_name in regressors:
                target_regressor = squared_name

        if target_regressor not in regressors:
            raise ValueError(
                f"{branch_key} model '{model_name}' is missing expected target regressor "
                f"'{target_regressor}' in config/{branch_key}.yaml-like model spec."
            )

        out[model_name] = {
            "feature_index": int(regressors.index(target_regressor)),
            "feature_name": model_name,
        }
    return out


def _common_behavior_model_features():
    evoked = _branch_behavior_model_features(evoked_trf_config, "evoked_trf")
    induced = _branch_behavior_model_features(induced_trf_config, "induced_trf")
    common = [name for name in evoked.keys() if name in induced]
    if not common:
        raise ValueError("No shared non-baseline behavior models found across evoked_trf and induced_trf configs.")
    mismatched = [name for name in common if evoked[name] != induced[name]]
    if mismatched:
        raise ValueError(
            "Evoked/induced model feature mappings differ for: " + ", ".join(sorted(mismatched))
        )
    return {name: evoked[name] for name in common}


KERNEL0_BEHAVIOR_MODEL_FEATURES = _common_behavior_model_features()
KERNEL0_BEHAVIOR_MODELS = list(KERNEL0_BEHAVIOR_MODEL_FEATURES.keys())


def _canonical_feature_name(value: str) -> str:
    token = str(value).strip()
    aliases = {
        "self_ipu_onset": "self_onset_impulse",
        "other_ipu_offset": "other_offset_impulse",
        "self_rate^2": "self_rate_sq",
        "other_rate^2": "other_rate_sq",
        "abs_diff^2": "abs_diff_sq",
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
    model = str(wildcards.model)
    if model in KERNEL0_BEHAVIOR_MODEL_FEATURES:
        return int(KERNEL0_BEHAVIOR_MODEL_FEATURES[model]["feature_index"])

    analysis = str(wildcards.analysis)
    feature_name = str(wildcards.feature_name)
    if analysis == "evoked_trf":
        idx = _model_feature_index_from_cfg(
            evoked_trf_config,
            "evoked_trf",
            model_name=model,
            feature_name=feature_name,
        )
    elif analysis.startswith("induced_trf_"):
        idx = _model_feature_index_from_cfg(
            induced_trf_config,
            "induced_trf",
            model_name=model,
            feature_name=feature_name,
        )
    else:
        idx = None

    if idx is None:
        raise ValueError(
            f"Cannot resolve feature_index for analysis={analysis}, model={model}, feature_name={feature_name}."
        )
    return int(idx)


def _kernel0_feature_name(wildcards):
    return str(KERNEL0_BEHAVIOR_MODEL_FEATURES[str(wildcards.model)]["feature_name"])


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
        f"evoked_trf/{model}/{KERNEL0_BEHAVIOR_MODEL_FEATURES[model]['feature_name']}/clusters.{_stats_table_ext()}"
    )
    for model in KERNEL0_BEHAVIOR_MODELS
]
KERNEL0_BEHAVIOR_TARGETS += [
    (
        f"{_stats_out_dir()}/stats/"
        f"{analysis}/{model}/{KERNEL0_BEHAVIOR_MODEL_FEATURES[model]['feature_name']}/clusters.{_stats_table_ext()}"
    )
    for analysis in INDUCED_ANALYSES
    for model in KERNEL0_BEHAVIOR_MODELS
]
KERNEL0_BEHAVIOR_SUBJECT_Z_TARGETS = [
    (
        f"{_stats_out_dir()}/stats/"
        "evoked_trf/"
        f"{model}/"
        f"{KERNEL0_BEHAVIOR_MODEL_FEATURES[model]['feature_name']}/"
        f"subject_level/subject_z_map_index.{_stats_table_ext()}"
    )
    for model in KERNEL0_BEHAVIOR_MODELS
]
KERNEL0_BEHAVIOR_SUBJECT_Z_TARGETS += [
    (
        f"{_stats_out_dir()}/stats/"
        f"{analysis}/"
        f"{model}/"
        f"{KERNEL0_BEHAVIOR_MODEL_FEATURES[model]['feature_name']}/"
        f"subject_level/subject_z_map_index.{_stats_table_ext()}"
    )
    for analysis in INDUCED_ANALYSES
    for model in KERNEL0_BEHAVIOR_MODELS
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


EVOKED_RECONSTRUCTION_R2_MODELS = [
    model for model in _active_evoked_models(evoked_trf_config)
    if str(model) != "baseline"
]

RECONSTRUCTION_ERP_R2_STAT_TARGETS = [
    f"{_stats_out_dir()}/stats/evoked_trf/{model}/reconstruction_erp_r2_vs_baseline/clusters.{_stats_table_ext()}"
    for model in EVOKED_RECONSTRUCTION_R2_MODELS
]
RECONSTRUCTION_ERP_R2_MEAN_TARGETS = [
    f"{_stats_out_dir()}/stats/evoked_trf/{model}/reconstruction_erp_r2_vs_baseline/mean_delta_r2.npy"
    for model in EVOKED_RECONSTRUCTION_R2_MODELS
]
RECONSTRUCTION_ERP_R2_LABEL_TARGETS = [
    f"{_stats_out_dir()}/stats/evoked_trf/{model}/reconstruction_erp_r2_vs_baseline/cluster_labels.npy"
    for model in EVOKED_RECONSTRUCTION_R2_MODELS
]
RECONSTRUCTION_ERP_R2_CHNAME_TARGETS = [
    f"{_stats_out_dir()}/stats/evoked_trf/{model}/reconstruction_erp_r2_vs_baseline/channel_names.npy"
    for model in EVOKED_RECONSTRUCTION_R2_MODELS
]
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
