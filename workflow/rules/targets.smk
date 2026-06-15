rule preprocessing_all:
    input:
        PREPROCESSING_STAGE


rule events_all:
    input:
        EVENTS_OUTPUT,
        EVENTS_QC_OUTPUT,
        EVENTS_VALIDATION_OUTPUT


rule epochs_all:
    input:
        epochs_stage_inputs


rule margin_epochs_all:
    input:
        margin_epochs_stage_inputs


rule lmeeeg_all:
    input:
        LMEEEG_STAGE,
        LMEEEG_STATS_OUTPUT


rule lmeeeg_nested_all:
    input:
        LMEEEG_NESTED_GROUP_STATS_OUTPUT


rule lmeeeg_commonality_all:
    input:
        LMEEEG_COMMONALITY_STAGE,
        LMEEEG_COMMONALITY_STATS_OUTPUT


rule similarity_all:
    input:
        SIMILARITY_STATS_OUTPUT


rule lmeeeg_variance_partition_all:
    input:
        LMEEEG_VARIANCE_PARTITION_STAGE,
        LMEEEG_VARIANCE_PARTITION_STATS_OUTPUT

rule r_analysis_all:
    input:
        R_ANALYSIS_PREPROCESS_SUMMARY_OUTPUT,
        R_ANALYSIS_RESULTS_OUTPUT,
        R_ANALYSIS_STAGE


_figures_all_inputs = [
    BEHAVIOURAL_FIGURES_OUTPUT,
    SIMILARITY_FIGURES_OUTPUT,
    LMEEEG_QC_FIGURES_OUTPUT,
    LMEEEG_INFERENCE_FIGURES_OUTPUT,
    LMEEEG_MAIN_RESULTS_FIGURES_OUTPUT,
    LMEEEG_VARIANCE_PARTITION_FIGURES_OUTPUT,
]


rule figures_all:
    input:
        _figures_all_inputs


rule behavioural_figures_all:
    input:
        BEHAVIOURAL_FIGURES_OUTPUT

rule similarity_figures_all:
    input:
        SIMILARITY_FIGURES_OUTPUT


rule lmeeeg_main_results_figures_all:
    input:
        LMEEEG_MAIN_RESULTS_FIGURES_OUTPUT


rule lmeeeg_variance_partition_figures_all:
    input:
        LMEEEG_VARIANCE_PARTITION_FIGURES_OUTPUT


rule qc_lmeeeg_plots_all:
    input:
        LMEEEG_QC_FIGURES_OUTPUT

rule inference_lmeeeg_plots_all:
    input:
        LMEEEG_INFERENCE_FIGURES_OUTPUT


rule all:
    input:
        PREPROCESSING_STAGE,
        EVENTS_STAGE,
        EPOCHS_STAGE,
        MARGIN_EPOCHS_STAGE,
        LMEEEG_STAGE,
        LMEEEG_STATS_OUTPUT,
        LMEEEG_NESTED_GROUP_STATS_OUTPUT,
        LMEEEG_COMMONALITY_STAGE,
        LMEEEG_COMMONALITY_STATS_OUTPUT,
        SIMILARITY_STATS_OUTPUT,
        LMEEEG_VARIANCE_PARTITION_STAGE,
        LMEEEG_VARIANCE_PARTITION_STATS_OUTPUT,
        R_ANALYSIS_PREPROCESS_SUMMARY_OUTPUT,
        R_ANALYSIS_RESULTS_OUTPUT,
        R_ANALYSIS_STAGE
    default_target: True
