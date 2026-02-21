# ==================================================================================================
#                                         STATS
# ==================================================================================================

OUT_DIR = config["paths"]["out_dir"]
TABLES_FORMAT = str(config.get("reporting", {}).get("tables_format", "parquet")).lower()
TABLE_EXT = "csv" if TABLES_FORMAT == "csv" else "parquet"

# In evoked_trf specs, each of these models has two baseline impulses first,
# so the behavior regressor is at feature index 2.
KERNEL0_BEHAVIOR_MODELS = ["self_rate", "other_rate", "abs_diff"]
KERNEL0_BEHAVIOR_FEATURE_INDEX = 2
KERNEL0_BEHAVIOR_FEATURE_DIR = "base"
KERNEL0_BEHAVIOR_TARGETS = [
    (
        f"{OUT_DIR}/stats/"
        f"evoked_trf/{model}/{KERNEL0_BEHAVIOR_FEATURE_DIR}/clusters.{TABLE_EXT}"
    )
    for model in KERNEL0_BEHAVIOR_MODELS
]
KERNEL0_BEHAVIOR_SUBJECT_Z_TARGETS = [
    (
        f"{OUT_DIR}/stats/"
        f"evoked_trf/{model}/{KERNEL0_BEHAVIOR_FEATURE_DIR}/subject_level/subject_z_map_index.{TABLE_EXT}"
    )
    for model in KERNEL0_BEHAVIOR_MODELS
]

rule stats_kernel0:
    input:
        index=f"{OUT_DIR}/dataset/index.parquet",
    output:
        clusters=(
            f"{OUT_DIR}/stats/"
            "{analysis}/{model}/" + KERNEL0_BEHAVIOR_FEATURE_DIR + "/clusters." + TABLE_EXT
        ),
        subject_z_maps=(
            f"{OUT_DIR}/stats/"
            "{analysis}/{model}/" + KERNEL0_BEHAVIOR_FEATURE_DIR + "/subject_z_maps.npy"
        ),
        grand_average=(
            f"{OUT_DIR}/stats/"
            "{analysis}/{model}/" + KERNEL0_BEHAVIOR_FEATURE_DIR + "/grand_average_trfs-ave.fif"
        ),
    shell:
        """
        python -m rate.cli.main stats-kernel0 \
            --analysis {wildcards.analysis} \
            --model {wildcards.model} \
            --feature-index {KERNEL0_BEHAVIOR_FEATURE_INDEX} \
            --feature-name {KERNEL0_BEHAVIOR_FEATURE_DIR}
        """


rule stats_kernel0_behavior:
    input:
        KERNEL0_BEHAVIOR_TARGETS


rule stats_kernel0_subject_level:
    input:
        index=f"{OUT_DIR}/dataset/index.parquet",
        subject_z_maps=(
            f"{OUT_DIR}/stats/"
            "{analysis}/{model}/" + KERNEL0_BEHAVIOR_FEATURE_DIR + "/subject_z_maps.npy"
        ),
    output:
        subject_z_index=(
            f"{OUT_DIR}/stats/"
            "{analysis}/{model}/" + KERNEL0_BEHAVIOR_FEATURE_DIR + "/subject_level/"
            "subject_z_map_index." + TABLE_EXT
        ),
    shell:
        """
        python -m rate.cli.main stats-export-subject-zmaps \
            --analysis {wildcards.analysis} \
            --model {wildcards.model} \
            --feature-index {KERNEL0_BEHAVIOR_FEATURE_INDEX} \
            --feature-name {KERNEL0_BEHAVIOR_FEATURE_DIR}
        """


rule stats_kernel0_behavior_subject_level:
    input:
        KERNEL0_BEHAVIOR_SUBJECT_Z_TARGETS
