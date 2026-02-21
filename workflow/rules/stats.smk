# ==================================================================================================
#                                         STATS
# ==================================================================================================

OUT_DIR = config["paths"]["out_dir"]

rule stats_kernel0:
    input:
        index=f"{OUT_DIR}/dataset/index.parquet",
    output:
        clusters=(
            f"{OUT_DIR}/final/kernel_stats/"
            "{analysis}/{model}/{feature_name}/feature-{feature_index}/clusters.parquet"
        ),
    shell:
        """
        python -m rate.cli.main stats-kernel0 \
            --analysis {wildcards.analysis} \
            --model {wildcards.model} \
            --feature-index {wildcards.feature_index} \
            --feature-name {wildcards.feature_name}
        """
