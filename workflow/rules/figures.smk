BEHAVIOURAL_FIGURES_OUTPUT = f"{OUT_DIR}/figures/behavioural/figure_manifest.json"
LMEEEG_QC_FIGURES_OUTPUT = f"{OUT_DIR}/figures/lmeeeg/figure_manifest.json"
LMEEEG_INFERENCE_FIGURES_OUTPUT = f"{OUT_DIR}/figures/lmeeeg_inference/figure_manifest.json"


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
