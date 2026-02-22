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

KERNEL0_BEHAVIOR_MODEL_FEATURES = {
    # Linear models: [self_onset_impulse, other_offset_impulse, target]
    "self_rate": {"feature_index": 2, "feature_name": "self_rate"},
    "other_rate": {"feature_index": 2, "feature_name": "other_rate"},
    "abs_diff": {"feature_index": 2, "feature_name": "abs_diff"},
    # Quadratic extensions: [self_onset_impulse, other_offset_impulse, linear, squared]
    "self_rate_sq": {"feature_index": 3, "feature_name": "self_rate_sq"},
    "other_rate_sq": {"feature_index": 3, "feature_name": "other_rate_sq"},
    "abs_diff_sq": {"feature_index": 3, "feature_name": "abs_diff_sq"},
}
KERNEL0_BEHAVIOR_MODELS = list(KERNEL0_BEHAVIOR_MODEL_FEATURES.keys())


def _kernel0_feature_index(wildcards):
    return int(KERNEL0_BEHAVIOR_MODEL_FEATURES[str(wildcards.model)]["feature_index"])


def _kernel0_feature_name(wildcards):
    return str(KERNEL0_BEHAVIOR_MODEL_FEATURES[str(wildcards.model)]["feature_name"])


KERNEL0_BEHAVIOR_TARGETS = [
    (
        f"{_stats_out_dir()}/stats/"
        f"evoked_trf/{model}/{KERNEL0_BEHAVIOR_MODEL_FEATURES[model]['feature_name']}/clusters.{_stats_table_ext()}"
    )
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

rule stats_kernel0:
    input:
        index=f"{_stats_out_dir()}/dataset/index.parquet",
        evoked_scores=f"{_stats_out_dir()}/evoked_trf/scores/scores.parquet",
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
            --feature-name {wildcards.feature_name}
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
            --feature-name {wildcards.feature_name}
        """


rule stats_kernel0_behavior_subject_level:
    input:
        KERNEL0_BEHAVIOR_SUBJECT_Z_TARGETS
