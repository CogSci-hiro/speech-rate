R_ANALYSIS_DIR = f"{OUT_DIR}/stats/r_analysis"
R_ANALYSIS_PREPROCESS_CLEANED_OUTPUT = f"{R_ANALYSIS_DIR}/cleaned_events.csv"
R_ANALYSIS_PREPROCESS_PARTNER_OUTPUT = f"{R_ANALYSIS_DIR}/partner_events.csv"
R_ANALYSIS_PREPROCESS_NULL_OUTPUT = f"{R_ANALYSIS_DIR}/null_summary.csv"
R_ANALYSIS_PREPROCESS_SUMMARY_OUTPUT = f"{R_ANALYSIS_DIR}/preprocess_summary.json"
R_ANALYSIS_RESULTS_OUTPUT = f"{R_ANALYSIS_DIR}/analysis_results.csv"
R_ANALYSIS_STAGE = f"{OUT_DIR}/_stage/r_analysis_all.done.json"
LMEEEG_STATS_OUTPUT = f"{OUT_DIR}/stats/lmeeeg_group_stats.json"
SIMILARITY_STATS_OUTPUT = str(build_similarity_output_paths(RESOLVED_CONFIG).summary_output_path)
LMEEEG_VARIANCE_PARTITION_STATS_OUTPUT = f"{OUT_DIR}/stats/lmeeeg_variance_partition_group_stats.json"
LMEEEG_VARIANCE_PARTITION_GROUP_DIR = str(build_variance_partition_group_output_paths(RESOLVED_CONFIG).group_dir)


rule behavioural_residuals:
    input:
        events=EVENTS_OUTPUT,
        preprocess_script=str(WORKFLOW_DIR / "scripts" / "preprocess.R"),
        script=str(WORKFLOW_DIR / "scripts" / "build_behavioural_residuals.R"),
    output:
        residuals=BEHAVIOURAL_RESIDUALS_OUTPUT,
        subject_slopes=BEHAVIOURAL_SUBJECT_SLOPES_OUTPUT,
        model_summary_csv=BEHAVIOURAL_MODEL_SUMMARY_CSV_OUTPUT,
        model_summary_txt=BEHAVIOURAL_MODEL_SUMMARY_TXT_OUTPUT,
    params:
        prediction_mode="in_sample",
    shell:
        "Rscript {input.script} "
        "--input {input.events} "
        "--preprocess_script {input.preprocess_script} "
        "--output {output.residuals} "
        "--subject_slopes_output {output.subject_slopes} "
        "--model_summary_csv {output.model_summary_csv} "
        "--model_summary_txt {output.model_summary_txt} "
        "--prediction_mode {params.prediction_mode}"


rule r_analysis_preprocess:
    input:
        events=EVENTS_OUTPUT,
        script=str(WORKFLOW_DIR / "scripts" / "preprocess.R"),
    output:
        cleaned=R_ANALYSIS_PREPROCESS_CLEANED_OUTPUT,
        partner=R_ANALYSIS_PREPROCESS_PARTNER_OUTPUT,
        null_summary=R_ANALYSIS_PREPROCESS_NULL_OUTPUT,
        summary=R_ANALYSIS_PREPROCESS_SUMMARY_OUTPUT,
    params:
        n_null=1000,
        seed=int((RESOLVED_CONFIG.get("project") or {}).get("random_seed", 1234)),
    shell:
        "Rscript {input.script} "
        "--input {input.events} "
        "--cleaned_output {output.cleaned} "
        "--partner_output {output.partner} "
        "--null_output {output.null_summary} "
        "--summary_output {output.summary} "
        "--n_null {params.n_null} "
        "--seed {params.seed}"


rule r_analysis_fit:
    input:
        events=EVENTS_OUTPUT,
        preprocess_summary=R_ANALYSIS_PREPROCESS_SUMMARY_OUTPUT,
        preprocess_script=str(WORKFLOW_DIR / "scripts" / "preprocess.R"),
        fit_script=str(WORKFLOW_DIR / "scripts" / "fit_lmeeeg.R"),
    output:
        results=R_ANALYSIS_RESULTS_OUTPUT,
    params:
        n_null=1000,
        seed=int((RESOLVED_CONFIG.get("project") or {}).get("random_seed", 1234)),
    shell:
        "mkdir -p $(dirname {output.results}) && "
        "Rscript {input.fit_script} "
        "--input {input.events} "
        "--preprocess_script {input.preprocess_script} "
        "--output {output.results} "
        "--n_null {params.n_null} "
        "--seed {params.seed}"


rule r_analysis_stage:
    input:
        R_ANALYSIS_PREPROCESS_SUMMARY_OUTPUT,
        R_ANALYSIS_RESULTS_OUTPUT,
    output:
        R_ANALYSIS_STAGE,
    shell:
        "python -c \"from pathlib import Path; "
        "path = Path(r'{output}'); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text('ok\\n', encoding='utf-8')\""


rule lmeeeg_group_stats:
    input:
        stats_config="config/stats.yaml",
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        trial_regressors=BEHAVIOURAL_RESIDUALS_OUTPUT,
    output:
        LMEEEG_STATS_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main lmeeeg-group-stats "
        "--config config/config.yaml --schema config/schema.json "
        "--stats-config {input.stats_config} --dataset-index {input.dataset_index} "
        "--trial-regressors-csv {input.trial_regressors} --output {output}"


rule lmeeeg_similarity:
    input:
        lmeeeg=LMEEEG_STAGE,
        config_path="config/config.yaml",
        schema_path="config/schema.json",
        similarity_config="config/similarity.yaml",
    output:
        summary=SIMILARITY_STATS_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main lmeeeg-similarity "
        "--config {input.config_path} --schema {input.schema_path}"

rule lmeeeg_variance_partition_group_stats:
    input:
        lmeeeg=LMEEEG_VARIANCE_PARTITION_STAGE,
        config_path="config/config.yaml",
        schema_path="config/schema.json",
    output:
        summary=LMEEEG_VARIANCE_PARTITION_STATS_OUTPUT,
        table=f"{LMEEEG_VARIANCE_PARTITION_GROUP_DIR}/descriptive_maps.parquet",
        channel_names=f"{LMEEEG_VARIANCE_PARTITION_GROUP_DIR}/channel_names.json",
        times=f"{LMEEEG_VARIANCE_PARTITION_GROUP_DIR}/times.npy",
        total_map=f"{LMEEEG_VARIANCE_PARTITION_GROUP_DIR}/total.npy",
        unique_self_map=f"{LMEEEG_VARIANCE_PARTITION_GROUP_DIR}/unique_self.npy",
        unique_other_map=f"{LMEEEG_VARIANCE_PARTITION_GROUP_DIR}/unique_other.npy",
        shared_map=f"{LMEEEG_VARIANCE_PARTITION_GROUP_DIR}/shared.npy",
        group_summary=f"{LMEEEG_VARIANCE_PARTITION_GROUP_DIR}/summary.json",
    shell:
        "PYTHONPATH=src python -m rate.cli.main lmeeeg-variance-partition-group-stats "
        "--config {input.config_path} --schema {input.schema_path} --output {output.summary}"
