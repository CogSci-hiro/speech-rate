# speech-rate

This repository contains the active Snakemake workflow and analysis code for the speech-rate project centered on face-to-face French conversation and dual-EEG hyperscanning. It is the paper-facing codebase for analyses of behavioral speech-rate convergence, pre-speech sensor-space `lmeEEG`, nested model comparisons, topographic similarity, variance partitioning, and figure generation.

The current scientific target is the manuscript:

- *Distinguishable neural correlates of perceived and produced speech rate in conversation*

In the current manuscript version, the central result is that conversational speech-rate alignment is behaviorally present, while the neural correlates of perceived partner rate and produced self rate are partly dissociable in time and scalp topography. This repository implements the analysis workflow behind that claim.

## Scientific Context

This codebase sits alongside the following related outputs and companion resources:

- Manuscript in preparation/submission:
  *Distinguishable neural correlates of perceived and produced speech rate in conversation*
- Related paper:
  [Two Interacting Neural Processes Support Speech Planning during Naturalistic Conversation](https://www.biorxiv.org/content/10.64898/2026.05.06.723165.abstract)
- Dataset paper:
  [A dual EEG hyperscanning dataset of natural French face-to-face conversation](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=OsHJ2bMAAAAJ&citation_for_view=OsHJ2bMAAAAJ:qjMakFHDy7sC)
- Companion repositories:
  [CogSci-hiro/turn-taking](https://github.com/CogSci-hiro/turn-taking)
  [CogSci-hiro/duet](https://github.com/CogSci-hiro/duet)
  [CogSci-hiro/lme-eeg](https://github.com/CogSci-hiro/lme-eeg)

At a high level, the project ecosystem is split across:

- `duet`: dataset-oriented and preprocessing-oriented infrastructure around the DUET hyperscanning recordings
- `lme-eeg`: standalone implementation of the `lmeEEG` method used as the backend for mass mixed-effects modeling
- `speech-rate` (this repo): paper-specific event selection, speech-rate regressors, inferential analyses, and figure assembly

## What This Repository Does

The active retained workflow covers five analysis branches:

1. Behavioral convergence analysis of adjacent conversational turns
2. Sensor-space pooled `lmeEEG` for `self_rate`, `other_rate`, and the combined `both_effect` model
3. Nested-model inference that tests unique contributions of `self_rate` and `other_rate`
4. Topographic similarity between self-rate and other-rate maps
5. Variance partitioning and commonality-style decomposition of unique versus shared variance

The repository has been cleaned down to the active paper-facing surface. Legacy GLM and legacy source-space branches are no longer part of the live workflow.

## Data and Experimental Framing

The analyses operate on conversational events derived from the DUET dual-EEG dataset described in the manuscript and dataset paper above. In the current configuration:

- task: `conversation`
- runs: `1` through `8`
- EEG: BioSemi 64-channel scalp EEG
- analysis window: pre-speech epochs aligned to the onset of the participant's own upcoming IPU
- key trial-wise regressors:
  `self_rate`, `other_rate`, `latency`, `duration_s`, `time_within_run`, `run`

According to the manuscript draft, the associated dataset is intended for release on OpenNeuro as `ds007764`.

## Active Analysis Design

### Behavioral branch

The behavioral model tests whether a participant's upcoming speech rate is predicted by the partner's immediately preceding speech rate while controlling for turn-level covariates.

### `lmeEEG` branch

The pooled EEG analyses fit three main models defined in [config/lmeeeg.yaml](/Users/hiro/Projects/active/speech-rate/config/lmeeeg.yaml:1):

- `other_rate_effect`
- `self_rate_effect`
- `both_effect`

These models operate on trial-wise EEG amplitudes in the pre-speech window and use the external [`lmeeeg`](https://pypi.org/project/lmeeeg/) package as the backend. In this repository, `lmeEEG` is the core inferential engine for sensor-space mass mixed-effects analysis.

### Nested-model branch

Nested inference is configured in [config/lmeeeg_nested.yaml](/Users/hiro/Projects/active/speech-rate/config/lmeeeg_nested.yaml:1) and compares the full `both_effect` model against drop-one-predictor reduced models:

- `drop_self_rate`
- `drop_other_rate`

These outputs feed the supplemental figures currently labeled `S1` and `S2`.

### Similarity branch

Similarity is configured in [config/similarity.yaml](/Users/hiro/Projects/active/speech-rate/config/similarity.yaml:1) and compares the scalp topography of:

- predictor A: `other_rate`
- predictor B: `self_rate`

The current inferential mode uses refit-permutation similarity over time.

### Variance partitioning and commonality branches

These branches quantify:

- unique variance attributable to `self_rate`
- unique variance attributable to `other_rate`
- variance shared by both predictors

They are configured in:

- [config/lmeeeg.yaml](/Users/hiro/Projects/active/speech-rate/config/lmeeeg.yaml:73) for variance partitioning
- [config/lmeeeg_commonality.yaml](/Users/hiro/Projects/active/speech-rate/config/lmeeeg_commonality.yaml:1) for commonality-style decomposition

## Repository Layout

Key entry points:

- main workflow: [workflow/Snakefile](/Users/hiro/Projects/active/speech-rate/workflow/Snakefile:1)
- workflow rules: [workflow/rules](/Users/hiro/Projects/active/speech-rate/workflow/rules:1)
- main package: [src/rate](/Users/hiro/Projects/active/speech-rate/src/rate:1)
- CLI registry: [src/rate/cli/main.py](/Users/hiro/Projects/active/speech-rate/src/rate/cli/main.py:1)
- project config: [config/config.yaml](/Users/hiro/Projects/active/speech-rate/config/config.yaml:1)
- modular analysis configs: [config](/Users/hiro/Projects/active/speech-rate/config:1)

Important subareas:

- `src/rate/events`: event construction and validation
- `src/rate/preprocessing`: preprocessing utilities
- `src/rate/epochs`: epoch construction
- `src/rate/lmeeeg`: pooled fitting, group statistics, nested inference, similarity, variance partitioning
- `src/rate/viz`: behavioral, `lmeEEG`, nested supplement, main-result, similarity, and variance-partition figures
- `tests/unit`: unit coverage for the active surface

## Configuration

The repository uses modular YAML configuration. The primary entry point is:

- [config/config.yaml](/Users/hiro/Projects/active/speech-rate/config/config.yaml:1)

Supporting configs include:

- [config/dataset.yaml](/Users/hiro/Projects/active/speech-rate/config/dataset.yaml:1)
- [config/preprocessing.yaml](/Users/hiro/Projects/active/speech-rate/config/preprocessing.yaml:1)
- [config/epochs.yaml](/Users/hiro/Projects/active/speech-rate/config/epochs.yaml:1)
- [config/margin_epochs.yaml](/Users/hiro/Projects/active/speech-rate/config/margin_epochs.yaml:1)
- [config/lmeeeg.yaml](/Users/hiro/Projects/active/speech-rate/config/lmeeeg.yaml:1)
- [config/lmeeeg_nested.yaml](/Users/hiro/Projects/active/speech-rate/config/lmeeeg_nested.yaml:1)
- [config/lmeeeg_commonality.yaml](/Users/hiro/Projects/active/speech-rate/config/lmeeeg_commonality.yaml:1)
- [config/similarity.yaml](/Users/hiro/Projects/active/speech-rate/config/similarity.yaml:1)
- [config/stats.yaml](/Users/hiro/Projects/active/speech-rate/config/stats.yaml:1)
- [config/viz.yaml](/Users/hiro/Projects/active/speech-rate/config/viz.yaml:1)

Before running the workflow on another machine, you will almost certainly need to edit the machine-specific paths in [config/config.yaml](/Users/hiro/Projects/active/speech-rate/config/config.yaml:1), especially:

- `paths.out_dir`
- `paths.bids_dir`
- `paths.ica_dir`
- `paths.annotations_dir`

## Installation

This package targets Python `>=3.10` and is defined in [pyproject.toml](/Users/hiro/Projects/active/speech-rate/pyproject.toml:1).

Minimal editable install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Core runtime dependencies include:

- `lmeeeg>=0.4.0`
- `mne>=1.11.0`
- `numpy`
- `pandas`
- `patsy`
- `statsmodels`
- `pyarrow`
- `scikit-learn`
- `jsonschema`

## Running the Workflow

### Validate configuration

```bash
PYTHONPATH=src python -m rate.cli.main validate-config \
  --config config/config.yaml \
  --schema config/schema.json
```

### Run the default retained analysis workflow

```bash
snakemake --snakefile workflow/Snakefile --cores 1
```

This default target runs the retained preprocessing, eventing, epoching, pooled `lmeEEG`, nested inference, commonality, similarity, variance partitioning, and behavioral-model outputs. Figure manifests are exposed through separate aggregate targets such as `figures_all`.

### Useful stage targets

The workflow defines several named aggregate targets in [workflow/rules/targets.smk](/Users/hiro/Projects/active/speech-rate/workflow/rules/targets.smk:1):

- `preprocessing_all`
- `events_all`
- `epochs_all`
- `lmeeeg_all`
- `lmeeeg_nested_all`
- `lmeeeg_commonality_all`
- `similarity_all`
- `lmeeeg_variance_partition_all`
- `figures_all`
- `behavioural_figures_all`
- `lmeeeg_main_results_figures_all`
- `supplement_figures_all`
- `qc_lmeeeg_plots_all`
- `inference_lmeeeg_plots_all`

Examples:

```bash
snakemake --snakefile workflow/Snakefile lmeeeg_all --cores 1
snakemake --snakefile workflow/Snakefile supplement_figures_all --cores 1
snakemake --snakefile workflow/Snakefile figures_all --cores 1
```

### Direct CLI entry points

The live CLI surface currently includes:

- `dataset-index`
- `build-events`
- `preprocess-subject`
- `epoch-subject`
- `lmeeeg-subject`
- `lmeeeg-group-stats`
- `lmeeeg-nested-group-stats`
- `lmeeeg-commonality`
- `lmeeeg-similarity`
- `lmeeeg-variance-partition-fit`
- `lmeeeg-variance-partition-group-stats`
- `figures-behavioural`
- `figures-lmeeeg`
- `figures-lmeeeg-inference`
- `figures-lmeeeg-main-results`
- `figures-lmeeeg-nested-supplement`
- `figures-lmeeeg-variance-partition`
- `figures-similarity`

See [src/rate/cli/main.py](/Users/hiro/Projects/active/speech-rate/src/rate/cli/main.py:13) for the authoritative registry.

## High-Level Data Flow

The active pipeline is:

1. Validate config and write a run manifest.
2. Build a dataset index from BIDS EEG data and annotation paths.
3. Preprocess each subject/task/run recording.
4. Build conversational events and epoch EEG around upcoming self-IPU onsets.
5. Fit pooled `lmeEEG` models for self-rate, other-rate, and both-predictor formulations.
6. Run group-level permutation inference.
7. Run nested-model tests, similarity analysis, commonality analysis, and variance partitioning.
8. Render QC, inference, main-result, variance-partition, similarity, behavioral, and supplement figures.

## Main Outputs

Representative outputs include:

- dataset index:
  `dataset/index.parquet`
- event table:
  `events/events.csv`
- pooled `lmeEEG` fits:
  `lmeeeg/<model>/summary.json`
- group inference:
  `stats/lmeeeg_group_stats.json`
- nested inference:
  `stats/lmeeeg_nested_group_stats.json`
- similarity:
  `similarity/<analysis>/summary.json`
- variance partitioning:
  `stats/lmeeeg_variance_partition_group_stats.json`
- behavioral summaries:
  `stats/behavioural_residuals/*`
- figure manifests:
  `figures/**/figure_manifest.json`

Main figure families currently include:

- behavioral summary
- `lmeEEG` QC plots
- `lmeEEG` inference plots
- main paper figures
- nested supplement figures `S1` and `S2`
- similarity summary figures
- variance-partition figures

## Testing

Run the focused unit suite with:

```bash
source .venv/bin/activate
pytest tests/unit -q
```

This repository has especially strong unit coverage around:

- `lmeEEG` trial design and pooled fitting
- group statistics and nested-model inference
- figure manifest generation
- config resolution and CLI registration

## Scope Notes

This repository now reflects the retained active analysis surface. Large legacy branches have been removed from the live package path, and the remaining workflow is intended to match the paper-facing analyses rather than preserve every historical experiment.

There is still archived material at the repository root, especially under `legacy/`, `dev/`, and `dev_outputs/`, but these are not part of the active Python package or current workflow registration.

## Citation and Attribution

If you use this repository, please cite:

- the speech-rate manuscript once publicly available
- the DUET dataset paper
- the `lmeEEG` methods paper:
  Visalli et al. (2024), *Journal of Neuroscience Methods*

Related code and data availability, according to the current manuscript draft:

- code for this paper:
  [CogSci-hiro/speech-rate](https://github.com/CogSci-hiro/speech-rate)
- `lmeEEG` implementation:
  [CogSci-hiro/lme-eeg](https://github.com/CogSci-hiro/lme-eeg)
- dataset release:
  OpenNeuro `ds007764`
