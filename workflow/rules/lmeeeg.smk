LMEEEG_STAGE = f"{OUT_DIR}/_stage/lmeeeg_all.done.json"


rule lmeeeg_subject:
    input:
        epochs=lambda wildcards: str(lmeeeg_paths(wildcards).epochs_input_path),
        metadata=lambda wildcards: str(lmeeeg_paths(wildcards).metadata_input_path),
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        config_path="config/config.yaml",
        schema_path="config/schema.json",
    output:
        residualised_trials=f"{OUT_DIR}/lmeeeg/{{subject_id}}/task-{{task}}/run-{{run}}/{{model_name}}/residualised_trials.npy",
        random_effects=f"{OUT_DIR}/lmeeeg/{{subject_id}}/task-{{task}}/run-{{run}}/{{model_name}}/random_effects.npy",
        betas=f"{OUT_DIR}/lmeeeg/{{subject_id}}/task-{{task}}/run-{{run}}/{{model_name}}/betas.npy",
        standard_errors=f"{OUT_DIR}/lmeeeg/{{subject_id}}/task-{{task}}/run-{{run}}/{{model_name}}/standard_errors.npy",
        t_values=f"{OUT_DIR}/lmeeeg/{{subject_id}}/task-{{task}}/run-{{run}}/{{model_name}}/t_values.npy",
        residual_variance=f"{OUT_DIR}/lmeeeg/{{subject_id}}/task-{{task}}/run-{{run}}/{{model_name}}/residual_variance.npy",
        row_index=f"{OUT_DIR}/lmeeeg/{{subject_id}}/task-{{task}}/run-{{run}}/{{model_name}}/row_index.npy",
        times=f"{OUT_DIR}/lmeeeg/{{subject_id}}/task-{{task}}/run-{{run}}/{{model_name}}/times.npy",
        channel_names=f"{OUT_DIR}/lmeeeg/{{subject_id}}/task-{{task}}/run-{{run}}/{{model_name}}/channel_names.json",
        column_names=f"{OUT_DIR}/lmeeeg/{{subject_id}}/task-{{task}}/run-{{run}}/{{model_name}}/column_names.json",
        test_predictors=f"{OUT_DIR}/lmeeeg/{{subject_id}}/task-{{task}}/run-{{run}}/{{model_name}}/test_predictors.json",
        selection=f"{OUT_DIR}/lmeeeg/{{subject_id}}/task-{{task}}/run-{{run}}/{{model_name}}/selection.parquet",
        lmm_diagnostics=f"{OUT_DIR}/lmeeeg/{{subject_id}}/task-{{task}}/run-{{run}}/{{model_name}}/lmm_diagnostics.json",
        summary=f"{OUT_DIR}/lmeeeg/{{subject_id}}/task-{{task}}/run-{{run}}/{{model_name}}/summary.json",
    params:
        subject_id=lambda wildcards: str(wildcards.subject_id),
        task=lambda wildcards: str(wildcards.task),
        run=lambda wildcards: str(wildcards.run),
        model_name=lambda wildcards: str(wildcards.model_name),
    shell:
        "PYTHONPATH=src python -m rate.cli.main lmeeeg-subject "
        "--config {input.config_path} --schema {input.schema_path} --dataset-index {input.dataset_index} "
        "--subject-id {params.subject_id} --task {params.task} --run {params.run} "
        "--model-name {params.model_name} --epochs-path {input.epochs} --metadata-path {input.metadata}"


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
