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


if "FOOOF_EPOCHS_STAGE" in globals():
    rule lmeeeg_fooof_all:
        input:
            LMEEEG_FOOOF_STAGE,
            LMEEEG_FOOOF_STATS_OUTPUT


if "GLM_STAGE" in globals():
    rule model_weights_all:
        input:
            MODEL_WEIGHTS_OUTPUT_DIR


_figures_all_inputs = [LMEEEG_QC_FIGURES_OUTPUT]
if "GLM_STAGE" in globals():
    _figures_all_inputs.append(GLM_FIGURES_OUTPUT)
if "FOOOF_MASS_STAGE" in globals():
    _figures_all_inputs.append(FOOOF_FIGURES_OUTPUT)
if "FOOOF_EPOCHS_STAGE" in globals():
    _figures_all_inputs.append(LMEEEG_FOOOF_QC_FIGURES_OUTPUT)


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
        LMEEEG_STATS_OUTPUT
    default_target: True
