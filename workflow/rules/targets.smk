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


rule r_analysis_all:
    input:
        R_ANALYSIS_STAGE


_figures_all_inputs = [LMEEEG_QC_FIGURES_OUTPUT]


rule figures_all:
    input:
        _figures_all_inputs


rule qc_lmeeeg_plots_all:
    input:
        LMEEEG_QC_FIGURES_OUTPUT


rule all:
    input:
        PREPROCESSING_STAGE,
        EVENTS_STAGE,
        EPOCHS_STAGE,
        MARGIN_EPOCHS_STAGE,
        LMEEEG_STATS_OUTPUT,
        R_ANALYSIS_STAGE
    default_target: True
