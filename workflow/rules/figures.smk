BEHAVIOURAL_FIGURES_OUTPUT = f"{OUT_DIR}/figures/behavioural/figure_manifest.json"
DECODING_FIGURES_OUTPUT = f"{OUT_DIR}/figures/decoding/figure_manifest.json"
DECODING_RESULTS_FIGURES_OUTPUT = f"{OUT_DIR}/figures/decoding_results/figure_manifest.json"
LMEEEG_QC_FIGURES_OUTPUT = f"{OUT_DIR}/figures/lmeeeg/figure_manifest.json"
LMEEEG_INFERENCE_FIGURES_OUTPUT = f"{OUT_DIR}/figures/lmeeeg_inference/figure_manifest.json"
LMEEEG_MAIN_RESULTS_FIGURES_OUTPUT = f"{OUT_DIR}/figures/lmeeeg_main_results/figure_manifest.json"
LMEEEG_VARIANCE_PARTITION_FIGURES_OUTPUT = f"{OUT_DIR}/figures/lmeeeg_variance_partition/figure_manifest.json"


rule figures_behavioural:
    input:
        residuals=BEHAVIOURAL_RESIDUALS_OUTPUT,
        subject_slopes=BEHAVIOURAL_SUBJECT_SLOPES_OUTPUT,
        viz_config="config/viz.yaml",
    output:
        BEHAVIOURAL_FIGURES_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main figures-behavioural "
        "--config config/config.yaml --schema config/schema.json "
        "--viz-config {input.viz_config} --input {input.residuals} "
        "--slopes-input {input.subject_slopes} --output {output}"


rule figures_decoding:
    input:
        decoding_stats=DECODING_STATS_STAGE,
        viz_config="config/viz.yaml",
        decoding_config="config/decoding.yaml",
    output:
        DECODING_FIGURES_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main figures-decoding "
        "--config config/config.yaml --schema config/schema.json "
        "--viz-config {input.viz_config} --decoding-config {input.decoding_config} "
        "--output {output}"


rule figures_decoding_results:
    input:
        decoding_stats=DECODING_STATS_STAGE,
        viz_config="config/viz.yaml",
        decoding_config="config/decoding.yaml",
    output:
        DECODING_RESULTS_FIGURES_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main figures-decoding-results "
        "--config config/config.yaml --schema config/schema.json "
        "--viz-config {input.viz_config} --decoding-config {input.decoding_config} "
        "--output {output}"


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
        viz_config="config/viz.yaml",
    output:
        LMEEEG_MAIN_RESULTS_FIGURES_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main figures-lmeeeg-main-results "
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
