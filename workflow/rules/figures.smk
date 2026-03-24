GLM_FIGURES_OUTPUT = f"{OUT_DIR}/figures/glm/figure_manifest.json"
FOOOF_FIGURES_OUTPUT = f"{OUT_DIR}/figures/fooof/figure_manifest.json"


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
