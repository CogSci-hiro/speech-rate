GLM_STAGE = f"{OUT_DIR}/_stage/glm_all.done.json"


rule glm_subject:
    input:
        epochs=lambda wildcards: str(glm_paths(wildcards).epochs_input_path),
        metadata=lambda wildcards: str(glm_paths(wildcards).metadata_input_path),
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        config_path="config/config.yaml",
        schema_path="config/schema.json",
    output:
        betas=f"{OUT_DIR}/glm/{{subject_id}}/{{model_name}}/betas.npy",
        standard_errors=f"{OUT_DIR}/glm/{{subject_id}}/{{model_name}}/standard_errors.npy",
        t_values=f"{OUT_DIR}/glm/{{subject_id}}/{{model_name}}/t_values.npy",
        residual_variance=f"{OUT_DIR}/glm/{{subject_id}}/{{model_name}}/residual_variance.npy",
        design_matrix=f"{OUT_DIR}/glm/{{subject_id}}/{{model_name}}/design_matrix.npy",
        row_index=f"{OUT_DIR}/glm/{{subject_id}}/{{model_name}}/row_index.npy",
        times=f"{OUT_DIR}/glm/{{subject_id}}/{{model_name}}/times.npy",
        channel_names=f"{OUT_DIR}/glm/{{subject_id}}/{{model_name}}/channel_names.json",
        column_names=f"{OUT_DIR}/glm/{{subject_id}}/{{model_name}}/column_names.json",
        test_predictors=f"{OUT_DIR}/glm/{{subject_id}}/{{model_name}}/test_predictors.json",
        predictor_columns=f"{OUT_DIR}/glm/{{subject_id}}/{{model_name}}/predictor_columns.json",
        predictor_betas=f"{OUT_DIR}/glm/{{subject_id}}/{{model_name}}/predictor_betas.npz",
        selection=f"{OUT_DIR}/glm/{{subject_id}}/{{model_name}}/selection.parquet",
        summary=f"{OUT_DIR}/glm/{{subject_id}}/{{model_name}}/summary.json",
    params:
        subject_id=lambda wildcards: str(wildcards.subject_id),
        model_name=lambda wildcards: str(wildcards.model_name),
    shell:
        "PYTHONPATH=src python -m rate.cli.main glm-subject "
        "--config {input.config_path} --schema {input.schema_path} --dataset-index {input.dataset_index} "
        "--subject-id {params.subject_id} "
        "--model-name {params.model_name} --epochs-path {input.epochs} --metadata-path {input.metadata}"


rule glm_stage:
    input:
        glm_stage_inputs,
    output:
        GLM_STAGE,
    shell:
        "python -c \"from pathlib import Path; "
        "path = Path(r'{output}'); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text('ok\\n', encoding='utf-8')\""
