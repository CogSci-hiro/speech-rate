LMEEEG_STAGE = f"{OUT_DIR}/_stage/lmeeeg_all.done.json"
LMEEEG_INDUCED_STAGE = f"{OUT_DIR}/_stage/lmeeeg_induced_all.done.json"


rule lmeeeg_fit:
    input:
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        config_path="config/config.yaml",
        schema_path="config/schema.json",
        epochs_stage=EPOCHS_STAGE,
        trial_regressors=BEHAVIOURAL_RESIDUALS_OUTPUT,
    output:
        residualised_trials=f"{OUT_DIR}/lmeeeg/{{model_name}}/residualised_trials.npy",
        random_effects=f"{OUT_DIR}/lmeeeg/{{model_name}}/random_effects.npy",
        betas=f"{OUT_DIR}/lmeeeg/{{model_name}}/betas.npy",
        standard_errors=f"{OUT_DIR}/lmeeeg/{{model_name}}/standard_errors.npy",
        t_values=f"{OUT_DIR}/lmeeeg/{{model_name}}/t_values.npy",
        residual_variance=f"{OUT_DIR}/lmeeeg/{{model_name}}/residual_variance.npy",
        row_index=f"{OUT_DIR}/lmeeeg/{{model_name}}/row_index.npy",
        times=f"{OUT_DIR}/lmeeeg/{{model_name}}/times.npy",
        channel_names=f"{OUT_DIR}/lmeeeg/{{model_name}}/channel_names.json",
        column_names=f"{OUT_DIR}/lmeeeg/{{model_name}}/column_names.json",
        test_predictors=f"{OUT_DIR}/lmeeeg/{{model_name}}/test_predictors.json",
        selection=f"{OUT_DIR}/lmeeeg/{{model_name}}/selection.parquet",
        lmm_diagnostics=f"{OUT_DIR}/lmeeeg/{{model_name}}/lmm_diagnostics.json",
        summary=f"{OUT_DIR}/lmeeeg/{{model_name}}/summary.json",
    params:
        model_name=lambda wildcards: str(wildcards.model_name),
    shell:
        "PYTHONPATH=src python -m rate.cli.main lmeeeg-subject "
        "--config {input.config_path} --schema {input.schema_path} --dataset-index {input.dataset_index} "
        "--model-name {params.model_name} --trial-regressors-csv {input.trial_regressors}"


rule lmeeeg_stage:
    input:
        lmeeeg_stage_inputs,
    output:
        LMEEEG_STAGE,
    shell:
        "python -c \"from pathlib import Path; "
        "path = Path(r'{output}'); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text('ok\\n', encoding='utf-8')\""


rule lmeeeg_induced_fit:
    input:
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        config_path="config/config.yaml",
        schema_path="config/schema.json",
        epochs_stage=EPOCHS_STAGE,
        trial_regressors=BEHAVIOURAL_RESIDUALS_OUTPUT,
    output:
        residualised_trials=f"{OUT_DIR}/lmeeeg_induced/{{target_name}}/{{model_name}}/residualised_trials.npy",
        random_effects=f"{OUT_DIR}/lmeeeg_induced/{{target_name}}/{{model_name}}/random_effects.npy",
        betas=f"{OUT_DIR}/lmeeeg_induced/{{target_name}}/{{model_name}}/betas.npy",
        standard_errors=f"{OUT_DIR}/lmeeeg_induced/{{target_name}}/{{model_name}}/standard_errors.npy",
        t_values=f"{OUT_DIR}/lmeeeg_induced/{{target_name}}/{{model_name}}/t_values.npy",
        residual_variance=f"{OUT_DIR}/lmeeeg_induced/{{target_name}}/{{model_name}}/residual_variance.npy",
        row_index=f"{OUT_DIR}/lmeeeg_induced/{{target_name}}/{{model_name}}/row_index.npy",
        times=f"{OUT_DIR}/lmeeeg_induced/{{target_name}}/{{model_name}}/times.npy",
        channel_names=f"{OUT_DIR}/lmeeeg_induced/{{target_name}}/{{model_name}}/channel_names.json",
        column_names=f"{OUT_DIR}/lmeeeg_induced/{{target_name}}/{{model_name}}/column_names.json",
        test_predictors=f"{OUT_DIR}/lmeeeg_induced/{{target_name}}/{{model_name}}/test_predictors.json",
        selection=f"{OUT_DIR}/lmeeeg_induced/{{target_name}}/{{model_name}}/selection.parquet",
        lmm_diagnostics=f"{OUT_DIR}/lmeeeg_induced/{{target_name}}/{{model_name}}/lmm_diagnostics.json",
        summary=f"{OUT_DIR}/lmeeeg_induced/{{target_name}}/{{model_name}}/summary.json",
    params:
        model_name=lambda wildcards: str(wildcards.model_name),
        target_name=lambda wildcards: str(wildcards.target_name),
    shell:
        "PYTHONPATH=src python -m rate.cli.main lmeeeg-subject "
        "--config {input.config_path} --schema {input.schema_path} --dataset-index {input.dataset_index} "
        "--config-section lmeeeg_induced --target {params.target_name} "
        "--model-name {params.model_name} --trial-regressors-csv {input.trial_regressors}"


rule lmeeeg_induced_stage:
    input:
        lmeeeg_induced_stage_inputs,
    output:
        LMEEEG_INDUCED_STAGE,
    shell:
        "python -c \"from pathlib import Path; "
        "path = Path(r'{output}'); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text('ok\\n', encoding='utf-8')\""
