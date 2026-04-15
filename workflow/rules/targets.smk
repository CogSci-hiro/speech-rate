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


rule lmeeeg_induced_all:
    input:
        LMEEEG_INDUCED_STAGE,
        LMEEEG_INDUCED_STATS_OUTPUT


rule lmeeeg_variance_partition_all:
    input:
        LMEEEG_VARIANCE_PARTITION_STAGE,
        LMEEEG_VARIANCE_PARTITION_STATS_OUTPUT


rule decoding_all:
    input:
        DECODING_STAGE


rule decoding_stats_all:
    input:
        DECODING_STATS_STAGE


rule r_analysis_all:
    input:
        R_ANALYSIS_PREPROCESS_SUMMARY_OUTPUT,
        R_ANALYSIS_RESULTS_OUTPUT,
        R_ANALYSIS_STAGE


_figures_all_inputs = [
    BEHAVIOURAL_FIGURES_OUTPUT,
    DECODING_FIGURES_OUTPUT,
    LMEEEG_QC_FIGURES_OUTPUT,
    LMEEEG_INFERENCE_FIGURES_OUTPUT,
    LMEEEG_INDUCED_QC_FIGURES_OUTPUT,
    LMEEEG_INDUCED_INFERENCE_FIGURES_OUTPUT,
]


rule figures_all:
    input:
        _figures_all_inputs


rule behavioural_figures_all:
    input:
        BEHAVIOURAL_FIGURES_OUTPUT


rule decoding_figures_all:
    input:
        DECODING_FIGURES_OUTPUT



rule qc_lmeeeg_plots_all:
    input:
        LMEEEG_QC_FIGURES_OUTPUT


rule qc_lmeeeg_induced_plots_all:
    input:
        LMEEEG_INDUCED_QC_FIGURES_OUTPUT


rule inference_lmeeeg_plots_all:
    input:
        LMEEEG_INFERENCE_FIGURES_OUTPUT


rule inference_lmeeeg_induced_plots_all:
    input:
        LMEEEG_INDUCED_INFERENCE_FIGURES_OUTPUT


rule all:
    input:
        PREPROCESSING_STAGE,
        EVENTS_STAGE,
        EPOCHS_STAGE,
        MARGIN_EPOCHS_STAGE,
        LMEEEG_STATS_OUTPUT,
        LMEEEG_INDUCED_STATS_OUTPUT,
        R_ANALYSIS_PREPROCESS_SUMMARY_OUTPUT,
        R_ANALYSIS_RESULTS_OUTPUT,
        R_ANALYSIS_STAGE
    default_target: True
