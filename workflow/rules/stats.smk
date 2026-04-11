GLM_STATS_OUTPUT = f"{OUT_DIR}/stats/glm_group_stats.json"
FOOOF_MASS_STATS_OUTPUT = f"{OUT_DIR}/stats/fooof_mass_group_stats.json"
LMEEEG_STATS_OUTPUT = f"{OUT_DIR}/stats/lmeeeg_group_stats.json"
LMEEEG_FOOOF_STATS_OUTPUT = f"{OUT_DIR}/stats/lmeeeg_fooof_group_stats.json"


if "GLM_STAGE" in globals():
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


if "FOOOF_MASS_STAGE" in globals():
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
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
    output:
        LMEEEG_STATS_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main lmeeeg-group-stats "
        "--config config/config.yaml --schema config/schema.json "
        "--stats-config {input.stats_config} --dataset-index {input.dataset_index} --output {output}"


if "FOOOF_EPOCHS_STAGE" in globals():
    rule lmeeeg_fooof_group_stats:
        input:
            lmeeeg=LMEEEG_FOOOF_STAGE,
            stats_config="config/stats.yaml",
            dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        output:
            LMEEEG_FOOOF_STATS_OUTPUT,
        shell:
            "PYTHONPATH=src python -m rate.cli.main lmeeeg-group-stats "
            "--config config/config.yaml --schema config/schema.json "
            "--stats-config {input.stats_config} --dataset-index {input.dataset_index} "
            "--config-section lmeeeg_fooof --output {output}"
