# ==================================================================================================
#                                         STATS
# ==================================================================================================

OUT_DIR = config["paths"]["out_dir"]
TABLES_FORMAT = str(config.get("reporting", {}).get("tables_format", "parquet")).lower()
TABLE_EXT = "csv" if TABLES_FORMAT == "csv" else "parquet"

KERNEL0_BEHAVIOR_MODEL_FEATURES = {
    # baseline regressors: [self_onset_impulse, other_offset_impulse, emg]
    "baseline": {"feature_index": 2, "feature_name": "emg"},
    # extended regressors: [self_onset_impulse, other_offset_impulse, emg, target]
    "self_rate": {"feature_index": 3, "feature_name": "self_rate"},
    "other_rate": {"feature_index": 3, "feature_name": "other_rate"},
    "abs_diff": {"feature_index": 3, "feature_name": "abs_diff"},
    # quadratic extensions: [self_onset_impulse, other_offset_impulse, emg, linear, squared]
    "self_rate_sq": {"feature_index": 4, "feature_name": "self_rate_sq"},
    "other_rate_sq": {"feature_index": 4, "feature_name": "other_rate_sq"},
    "abs_diff_sq": {"feature_index": 4, "feature_name": "abs_diff_sq"},
}
KERNEL0_BEHAVIOR_MODELS = list(KERNEL0_BEHAVIOR_MODEL_FEATURES.keys())


def _kernel0_feature_index(wildcards):
    return int(KERNEL0_BEHAVIOR_MODEL_FEATURES[str(wildcards.model)]["feature_index"])


def _kernel0_feature_name(wildcards):
    return str(KERNEL0_BEHAVIOR_MODEL_FEATURES[str(wildcards.model)]["feature_name"])


KERNEL0_BEHAVIOR_TARGETS = [
    (
        f"{OUT_DIR}/stats/"
        f"evoked_trf/{model}/{KERNEL0_BEHAVIOR_MODEL_FEATURES[model]['feature_name']}/clusters.{TABLE_EXT}"
    )
    for model in KERNEL0_BEHAVIOR_MODELS
]
KERNEL0_BEHAVIOR_SUBJECT_Z_TARGETS = [
    (
        f"{OUT_DIR}/stats/"
        "evoked_trf/"
        f"{model}/"
        f"{KERNEL0_BEHAVIOR_MODEL_FEATURES[model]['feature_name']}/"
        f"subject_level/subject_z_map_index.{TABLE_EXT}"
    )
    for model in KERNEL0_BEHAVIOR_MODELS
]

rule stats_kernel0:
    input:
        index=f"{OUT_DIR}/dataset/index.parquet",
        evoked_scores=f"{OUT_DIR}/evoked_trf/scores/scores.parquet",
    output:
        clusters=(
            f"{OUT_DIR}/stats/"
            "{analysis}/{model}/{feature_name}/clusters." + TABLE_EXT
        ),
        cluster_labels=(
            f"{OUT_DIR}/stats/"
            "{analysis}/{model}/{feature_name}/cluster_labels.npy"
        ),
        t_obs_map=(
            f"{OUT_DIR}/stats/"
            "{analysis}/{model}/{feature_name}/t_obs_map.npy"
        ),
        subject_z_maps=(
            f"{OUT_DIR}/stats/"
            "{analysis}/{model}/{feature_name}/subject_z_maps.npy"
        ),
        grand_average=(
            f"{OUT_DIR}/stats/"
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
            --feature-name {wildcards.feature_name}
        """


rule stats_kernel0_behavior:
    input:
        KERNEL0_BEHAVIOR_TARGETS


rule stats_kernel0_subject_level:
    input:
        index=f"{OUT_DIR}/dataset/index.parquet",
        subject_z_maps=(
            f"{OUT_DIR}/stats/"
            "{analysis}/{model}/{feature_name}/subject_z_maps.npy"
        ),
    output:
        subject_z_index=(
            f"{OUT_DIR}/stats/"
            "{analysis}/{model}/{feature_name}/subject_level/"
            "subject_z_map_index." + TABLE_EXT
        ),
    params:
        feature_index=_kernel0_feature_index,
    shell:
        """
        python -m rate.cli.main stats-export-subject-zmaps \
            --analysis {wildcards.analysis} \
            --model {wildcards.model} \
            --feature-index {params.feature_index} \
            --feature-name {wildcards.feature_name}
        """


rule stats_kernel0_behavior_subject_level:
    input:
        KERNEL0_BEHAVIOR_SUBJECT_Z_TARGETS
