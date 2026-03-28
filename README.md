# speech-rate

This repository now uses Snakemake as the active pipeline core for a new mass-univariate EEG scaffold. The active workflow is organized around dataset indexing, preprocessing, epoching, subject-level modeling, thin group-level stats wrappers, and figure generation.

The subject-level `lmeEEG` fitting stage is backed by the external [`lmeeeg`](https://pypi.org/project/lmeeeg/) package rather than an in-repo mixed-model implementation.

## Active Workflow

- Main config entry point: `config/config.yaml`
- Modular analysis configs: `config/dataset.yaml`, `config/preprocessing.yaml`, `config/epochs.yaml`, `config/glm.yaml`, `config/fooof.yaml`, `config/lmeeeg.yaml`, `config/stats.yaml`, `config/viz.yaml`
- CLI entry point: `src/rate/cli/main.py`
- Snakemake entry point: `workflow/Snakefile`

The current scaffold is intentionally light. Commands and rules create valid placeholder outputs so the repository structure, import graph, and pipeline orchestration are ready for later implementation.

## Analysis Directions

- ERP/fMRI-style mass-univariate regression on eligible epochs
- Sliding-window FOOOF feature extraction followed by subject-level regression
- Optional `lmeEEG` trial preparation and group-level inference

## Legacy TRF Code

TRF-era analysis code has been quarantined under `legacy/trf/`. It is preserved for reference, but it is no longer part of the active CLI registration or active Snakemake include path.

## Pipeline Shape

1. Build the dataset index.
2. Run preprocessing placeholders for the active scaffold.
3. Build epochs and mark eligibility through metadata-aware modules.
4. Fit subject-level GLM or feature-extraction stages.
5. Run thin group-level stats wrappers.
6. Generate figure manifests for downstream visualization work.
