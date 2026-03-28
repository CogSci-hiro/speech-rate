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


rule glm_all:
    input:
        GLM_STAGE,
        GLM_STATS_OUTPUT


rule fooof_mass_all:
    input:
        FOOOF_MASS_STAGE,
        FOOOF_MASS_STATS_OUTPUT


rule fooof_epochs_all:
    input:
        FOOOF_EPOCHS_STAGE


rule lmeeeg_all:
    input:
        LMEEEG_STAGE,
        LMEEEG_STATS_OUTPUT


rule model_weights_all:
    input:
        MODEL_WEIGHTS_OUTPUT_DIR


rule figures_all:
    input:
        GLM_FIGURES_OUTPUT,
        FOOOF_FIGURES_OUTPUT


rule qc_lmeeeg_plots_all:
    input:
        LMEEEG_QC_FIGURES_OUTPUT


rule qc_lmeeeg_inference_plots_all:
    input:
        LMEEEG_INFERENCE_FIGURES_OUTPUT


rule all:
    input:
        PREPROCESSING_STAGE,
        EVENTS_STAGE,
        EPOCHS_STAGE,
        MARGIN_EPOCHS_STAGE,
        FOOOF_EPOCHS_STAGE,
        GLM_STATS_OUTPUT,
        FOOOF_MASS_STATS_OUTPUT,
        LMEEEG_STATS_OUTPUT,
        GLM_FIGURES_OUTPUT,
        FOOOF_FIGURES_OUTPUT
    default_target: True
