GLM_STATS_OUTPUT = f"{OUT_DIR}/stats/glm_group_stats.json"
FOOOF_MASS_STATS_OUTPUT = f"{OUT_DIR}/stats/fooof_mass_group_stats.json"
LMEEEG_STATS_OUTPUT = f"{OUT_DIR}/stats/lmeeeg_group_stats.json"


rule glm_group_stats:
    input:
        glm=GLM_STAGE,
        stats_config="config/stats.yaml",
    output:
        GLM_STATS_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main glm-group-stats "
        "--config config/config.yaml --schema config/schema.json "
        "--stats-config {input.stats_config} --output {output}"


rule fooof_mass_group_stats:
    input:
        fooof_mass=FOOOF_MASS_STAGE,
        stats_config="config/stats.yaml",
    output:
        FOOOF_MASS_STATS_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main fooof-mass-group-stats "
        "--config config/config.yaml --schema config/schema.json "
        "--stats-config {input.stats_config} --output {output}"


rule lmeeeg_group_stats:
    input:
        lmeeeg=LMEEEG_STAGE,
        stats_config="config/stats.yaml",
    output:
        LMEEEG_STATS_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main lmeeeg-group-stats "
        "--config config/config.yaml --schema config/schema.json "
        "--stats-config {input.stats_config} --output {output}"
