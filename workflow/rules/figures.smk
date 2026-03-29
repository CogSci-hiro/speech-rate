GLM_FIGURES_OUTPUT = f"{OUT_DIR}/figures/glm/figure_manifest.json"
MODEL_WEIGHTS_OUTPUT_DIR = f"{OUT_DIR}/figures/glm/model_weights"
FOOOF_FIGURES_OUTPUT = f"{OUT_DIR}/figures/fooof/figure_manifest.json"
LMEEEG_QC_FIGURES_OUTPUT = f"{OUT_DIR}/figures/lmeeeg/figure_manifest.json"
LMEEEG_FOOOF_QC_FIGURES_OUTPUT = f"{OUT_DIR}/figures/lmeeeg_fooof/figure_manifest.json"
LMEEEG_INFERENCE_FIGURES_OUTPUT = f"{OUT_DIR}/figures/lmeeeg_inference/figure_manifest.json"
LMEEEG_FOOOF_INFERENCE_FIGURES_OUTPUT = f"{OUT_DIR}/figures/lmeeeg_fooof_inference/figure_manifest.json"


rule figures_glm:
    input:
        stats_output=GLM_STATS_OUTPUT,
        viz_config="config/viz.yaml",
    output:
        GLM_FIGURES_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main figures-glm "
        "--config config/config.yaml --schema config/schema.json "
        "--viz-config {input.viz_config} --output {output}"


rule figures_fooof:
    input:
        stats_output=FOOOF_MASS_STATS_OUTPUT,
        viz_config="config/viz.yaml",
    output:
        FOOOF_FIGURES_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main figures-fooof "
        "--config config/config.yaml --schema config/schema.json "
        "--viz-config {input.viz_config} --output {output}"


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


rule figures_lmeeeg_fooof:
    input:
        lmeeeg_output=LMEEEG_FOOOF_STAGE,
        viz_config="config/viz.yaml",
    output:
        LMEEEG_FOOOF_QC_FIGURES_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main figures-lmeeeg-fooof "
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


rule figures_lmeeeg_fooof_inference:
    input:
        lmeeeg_output=LMEEEG_FOOOF_STATS_OUTPUT,
        viz_config="config/viz.yaml",
    output:
        LMEEEG_FOOOF_INFERENCE_FIGURES_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main figures-lmeeeg-fooof-inference "
        "--config config/config.yaml --schema config/schema.json "
        "--viz-config {input.viz_config} --output {output}"


rule model_weights_glm:
    input:
        glm_outputs=glm_model_weights_inputs,
        viz_config="config/viz.yaml",
    output:
        directory(MODEL_WEIGHTS_OUTPUT_DIR),
    shell:
        "PYTHONPATH=src python -m rate.cli.main model-weights-glm "
        "--config config/config.yaml --schema config/schema.json "
        "--viz-config {input.viz_config} --output {output}/manifest.json"
