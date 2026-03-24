rule preprocessing_all:
    input:
        PREPROCESSING_STAGE


rule epochs_all:
    input:
        EPOCHS_STAGE


rule glm_all:
    input:
        GLM_STAGE,
        GLM_STATS_OUTPUT


rule fooof_mass_all:
    input:
        FOOOF_MASS_STAGE,
        FOOOF_MASS_STATS_OUTPUT


rule lmeeeg_all:
    input:
        LMEEEG_STAGE,
        LMEEEG_STATS_OUTPUT


rule figures_all:
    input:
        GLM_FIGURES_OUTPUT,
        FOOOF_FIGURES_OUTPUT


rule all:
    input:
        PREPROCESSING_STAGE,
        EPOCHS_STAGE,
        GLM_STATS_OUTPUT,
        FOOOF_MASS_STATS_OUTPUT,
        LMEEEG_STATS_OUTPUT,
        GLM_FIGURES_OUTPUT,
        FOOOF_FIGURES_OUTPUT
    default_target: True
