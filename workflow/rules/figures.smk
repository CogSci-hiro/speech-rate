BEHAVIOURAL_FIGURES_OUTPUT = f"{OUT_DIR}/figures/behavioural/figure_manifest.json"
SIMILARITY_FIGURES_OUTPUT = str(build_similarity_output_paths(RESOLVED_CONFIG).figure_manifest_output_path)
LMEEEG_QC_FIGURES_OUTPUT = f"{OUT_DIR}/figures/lmeeeg/figure_manifest.json"
LMEEEG_INFERENCE_FIGURES_OUTPUT = f"{OUT_DIR}/figures/lmeeeg_inference/figure_manifest.json"
LMEEEG_MAIN_RESULTS_FIGURES_OUTPUT = f"{OUT_DIR}/figures/main_figures/figure_manifest.json"
LMEEEG_NESTED_SUPPLEMENT_FIGURES_OUTPUT = f"{OUT_DIR}/figures/supplement/figure_manifest.json"
LMEEEG_VARIANCE_PARTITION_FIGURES_OUTPUT = f"{OUT_DIR}/figures/lmeeeg_variance_partition/figure_manifest.json"


rule figures_behavioural:
    input:
        residuals=BEHAVIOURAL_RESIDUALS_OUTPUT,
        random_effects=BEHAVIOURAL_RANDOM_EFFECTS_OUTPUT,
        lmm_summary=R_ANALYSIS_MODEL_SUMMARY_TXT_OUTPUT,
        viz_config="config/viz.yaml",
    output:
        BEHAVIOURAL_FIGURES_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main figures-behavioural "
        "--config config/config.yaml --schema config/schema.json "
        "--viz-config {input.viz_config} --input {input.residuals} "
        "--random-effects-input {input.random_effects} "
        "--lmm-summary-input {input.lmm_summary} --output {output}"


rule figures_lmeeeg:
    input:
        lmeeeg_output=LMEEEG_STATS_OUTPUT,
        viz_config="config/viz.yaml",
    output:
        LMEEEG_QC_FIGURES_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main figures-lmeeeg "
        "--config config/config.yaml --schema config/schema.json "
        "--viz-config {input.viz_config} --output {output}"


rule figures_lmeeeg_inference:
    input:
        lmeeeg_output=LMEEEG_STATS_OUTPUT,
        viz_config="config/viz.yaml",
    output:
        LMEEEG_INFERENCE_FIGURES_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main figures-lmeeeg-inference "
        "--config config/config.yaml --schema config/schema.json "
        "--viz-config {input.viz_config} --output {output}"


rule figures_lmeeeg_main_results:
    input:
        lmeeeg_output=LMEEEG_STATS_OUTPUT,
        similarity_output=SIMILARITY_STATS_OUTPUT,
        behavioural_residuals=BEHAVIOURAL_RESIDUALS_OUTPUT,
        behavioural_random_effects=BEHAVIOURAL_RANDOM_EFFECTS_OUTPUT,
        variance_partition_stats=LMEEEG_VARIANCE_PARTITION_STATS_OUTPUT,
        viz_config="config/viz.yaml",
    output:
        LMEEEG_MAIN_RESULTS_FIGURES_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main figures-lmeeeg-main-results "
        "--config config/config.yaml --schema config/schema.json "
        "--viz-config {input.viz_config} --output {output}"


rule figures_lmeeeg_nested_supplement:
    input:
        nested_stats=LMEEEG_NESTED_GROUP_STATS_OUTPUT,
        viz_config="config/viz.yaml",
    output:
        LMEEEG_NESTED_SUPPLEMENT_FIGURES_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main figures-lmeeeg-nested-supplement "
        "--config config/config.yaml --schema config/schema.json "
        "--viz-config {input.viz_config} --output {output}"


rule figures_similarity:
    input:
        similarity_output=SIMILARITY_STATS_OUTPUT,
        viz_config="config/viz.yaml",
    output:
        SIMILARITY_FIGURES_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main figures-similarity "
        "--config config/config.yaml --schema config/schema.json "
        "--viz-config {input.viz_config} --output {output}"


rule figures_lmeeeg_variance_partition:
    input:
        variance_partition_stats=LMEEEG_VARIANCE_PARTITION_STATS_OUTPUT,
        variance_partition_table=f"{LMEEEG_VARIANCE_PARTITION_GROUP_DIR}/descriptive_maps.parquet",
        viz_config="config/viz.yaml",
    output:
        LMEEEG_VARIANCE_PARTITION_FIGURES_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main figures-lmeeeg-variance-partition "
        "--config config/config.yaml --schema config/schema.json "
        "--viz-config {input.viz_config} --input {input.variance_partition_table} "
        "--output {output}"
