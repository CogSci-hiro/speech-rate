# speech-rate

## Configuration Lifecycle Separation

This repository separates configuration into lifecycle layers so small edits do not
trigger unnecessary recomputation.

### Layers

- Compute layer (heavy rules): `config/config.yaml` plus compute-related modular configs
  (`config/trf.yaml`, `config/evoked_trf.yaml`, `config/induced_trf.yaml`,
  `config/fooof.yaml`, `config/ddc.yaml`).
- Inference layer (stats rules): `config/stats.yaml`
- Visualization layer (figure rules): `config/viz.yaml`

### Edit Impact

- Editing `config/viz.yaml` should affect figures only.
- Editing `config/stats.yaml` should affect inference/stats outputs only.
- Editing compute configs should affect compute outputs (and downstream stats/figures).

### Hard Boundary Rules

- Compute rules must not depend on `config/stats.yaml` or `config/viz.yaml`.
- Stats rules must not depend on `config/viz.yaml`.
- Figure rules depend on `config/viz.yaml` plus frozen stats outputs.
- Rule files must not read the global Snakemake `config` dict directly.

### Why Full-Config Passing Is Forbidden

Passing a full config object (or routing unrelated config files into a rule) causes
accidental cross-layer coupling. In Snakemake, that can make unrelated config edits
appear as rule changes and trigger expensive recomputation. Rules should pass only
the config file or parameters that are relevant to that rule's lifecycle layer.
