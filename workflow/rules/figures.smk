BASELINE_CONTROL_REGRESSORS = {
    "emg",
    "self_onset_impulse",
    "other_offset_impulse",
    "self_ipu_onset",
    "other_ipu_offset",
}
TARGET_BEHAVIOR_REGRESSORS = {"self_rate", "other_rate", "alignment", "convergence"}
DEFAULT_BASELINE_VIZ_REGRESSORS = (
    "self_onset_impulse",
    "other_offset_impulse",
    "emg",
)


def _viz_out_dir() -> str:
    # Read from viz.yaml (loaded manually in Snakefile) so visualization changes do
    # not retrigger compute jobs via the tracked Snakemake `config`.
    return str(viz_config["paths"]["out_dir"])


def _viz_regressors():
    trf_reporting = viz_config.get("reporting", {}).get("trf", {})
    return list(trf_reporting.get("regressors", []))


def _viz_table_ext() -> str:
    tables_format = str(viz_config.get("reporting", {}).get("tables_format", "parquet")).lower()
    return "csv" if tables_format == "csv" else "parquet"


RAW_REGRESSORS = _viz_regressors()

if not RAW_REGRESSORS:
    raise ValueError("config/viz.yaml reporting.trf.regressors must define at least one TRF regressor")


def _induced_bands():
    ind_cfg = viz_induced_trf_config
    bands = ind_cfg.get("bands")
    if bands:
        return [str(b) for b in bands]
    bands_hz = ind_cfg.get("bands_hz", {}) or {}
    return sorted(str(k) for k in bands_hz.keys())


def _induced_target_kinds():
    ind_cfg = viz_induced_trf_config
    kinds = ind_cfg.get("target_kinds")
    if kinds:
        return [str(k).strip().lower() for k in kinds]
    return ["raw"]


def _induced_bands_for_target_kind(target_kind: str):
    base = _induced_bands()
    mode = str(target_kind).strip().lower()
    if mode != "residual":
        return base
    ind_cfg = viz_induced_trf_config
    residual = ind_cfg.get("residual_bands")
    if residual is None:
        return base
    return [str(b).strip() for b in residual if str(b).strip() and str(b).strip() in set(base)]


def _induced_family_name(band: str, target_kind: str) -> str:
    if str(target_kind).strip().lower() == "raw":
        return f"induced_trf_{str(band)}"
    return f"induced_trf_{str(band)}_{str(target_kind).strip().lower()}"


INDUCED_FAMILIES = [
    _induced_family_name(band, target_kind)
    for target_kind in _induced_target_kinds()
    for band in _induced_bands_for_target_kind(target_kind)
]


def _trf_model_for_regressor(regressor: str) -> str:
    value = str(regressor).strip()
    return "baseline" if value in BASELINE_CONTROL_REGRESSORS else value


def _canonical_feature_name(value: str) -> str:
    token = str(value).strip()
    aliases = {
        "self_ipu_onset": "self_onset_impulse",
        "other_ipu_offset": "other_offset_impulse",
        "self_rate^2": "self_rate_sq",
        "other_rate^2": "other_rate_sq",
        "alignment^2": "alignment_sq",
        "convergence^2": "convergence_sq",
    }
    return aliases.get(token, token)


def _active_model_names(branch_cfg, branch_key: str):
    block = branch_cfg.get(branch_key, {}) or {}
    specs = block.get("models", []) or []
    out = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name", "")).strip()
        if name:
            out.append(name)
    return out


def _regressor_model_map(branch_cfg, branch_key: str):
    block = branch_cfg.get(branch_key, {}) or {}
    specs = block.get("models", []) or []
    out = {}
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        model_name = str(spec.get("name", "")).strip()
        regressors = [str(r).strip() for r in (spec.get("regressors", []) or [])]
        if not model_name:
            continue
        for regressor in regressors:
            canonical = _canonical_feature_name(regressor)
            if canonical:
                out.setdefault(canonical, model_name)
    return out


ACTIVE_EVOKED_MODELS = set(_active_model_names(evoked_trf_config, "evoked_trf"))
ACTIVE_INDUCED_MODELS = set(_active_model_names(induced_trf_config, "induced_trf"))
EVOKED_REGRESSOR_MODEL_MAP = _regressor_model_map(evoked_trf_config, "evoked_trf")
INDUCED_REGRESSOR_MODEL_MAP = _regressor_model_map(induced_trf_config, "induced_trf")


def _configured_model_for_regressor(*, family: str, regressor: str) -> str:
    token = _canonical_feature_name(regressor)
    if token in BASELINE_CONTROL_REGRESSORS:
        return "baseline"
    model_map = EVOKED_REGRESSOR_MODEL_MAP if family == "evoked_trf" else INDUCED_REGRESSOR_MODEL_MAP
    return str(model_map.get(token, token))


def _configured_regressors_for_family(family: str):
    active_models = ACTIVE_EVOKED_MODELS if family == "evoked_trf" else ACTIVE_INDUCED_MODELS
    out = []
    for reg in RAW_REGRESSORS:
        model = _configured_model_for_regressor(family=family, regressor=reg)
        if model in active_models:
            out.append(reg)
        elif model == "baseline" and "baseline" in active_models:
            out.append(reg)
    return out


def _baseline_control_dependency(regressor: str) -> str:
    """Return an existing dependency path for baseline controls to avoid DAG-time missing-input errors."""
    model = _configured_model_for_regressor(family="evoked_trf", regressor=regressor)
    stats_kernel = (
        f"{_viz_out_dir()}/stats/evoked_trf/{model}/{regressor}/grand_average_trfs-ave.fif"
    )
    subjects_root = f"{_viz_out_dir()}/evoked_trf/subjects"
    if Path(stats_kernel).exists():
        return stats_kernel
    if Path(subjects_root).exists():
        return subjects_root
    # Stable fallback that should exist for any figure run context.
    return f"{_viz_out_dir()}/dataset/index.parquet"


def _evoked_kernel_input(wildcards):
    regressor = str(wildcards.regressor).strip()
    model = _trf_model_for_regressor(regressor)
    if regressor == "emg":
        del model
        return f"{_viz_out_dir()}/features/emg/grand_average_emg_self_onset-ave.fif"
    if model == "baseline" and regressor in BASELINE_CONTROL_REGRESSORS:
        # Baseline controls may be rendered from subject-level FIFs even when
        # stats-level grand-average files are not present.
        return _baseline_control_dependency(regressor)
    return f"{_viz_out_dir()}/stats/evoked_trf/{model}/{wildcards.regressor}/grand_average_trfs-ave.fif"


def _evoked_cluster_input(wildcards):
    regressor = str(wildcards.regressor).strip()
    model = _trf_model_for_regressor(regressor)
    if regressor == "emg":
        # EMG onset-locked grand average currently has no cluster-table companion.
        del model
        return f"{_viz_out_dir()}/features/emg/grand_average_emg_self_onset-ave.fif"
    if model == "baseline" and regressor in BASELINE_CONTROL_REGRESSORS:
        # No cluster table is required for baseline-control rendering fallback.
        return _baseline_control_dependency(regressor)
    return f"{_viz_out_dir()}/stats/evoked_trf/{model}/{wildcards.regressor}/clusters.{_viz_table_ext()}"


def _evoked_timecourse_kernel_inputs(_wildcards):
    return [
        f"{_viz_out_dir()}/stats/evoked_trf/{_trf_model_for_regressor(reg)}/{reg}/grand_average_trfs-ave.fif"
        for reg in _configured_regressors_for_family("evoked_trf")
    ]


def _family_kernel_input(wildcards):
    model = _trf_model_for_regressor(str(wildcards.regressor))
    family = str(wildcards.family)
    return f"{_viz_out_dir()}/stats/{family}/{model}/{wildcards.regressor}/grand_average_trfs-ave.fif"


def _family_cluster_input(wildcards):
    model = _trf_model_for_regressor(str(wildcards.regressor))
    family = str(wildcards.family)
    return f"{_viz_out_dir()}/stats/{family}/{model}/{wildcards.regressor}/clusters.{_viz_table_ext()}"


def _family_timecourse_kernel_inputs(wildcards):
    family = str(wildcards.family)
    return [
        f"{_viz_out_dir()}/stats/{family}/{_trf_model_for_regressor(reg)}/{reg}/grand_average_trfs-ave.fif"
        for reg in AVAILABLE_INDUCED_REGRESSORS_BY_FAMILY.get(family, [])
    ]


def _has_evoked_models(models):
    return all(str(model) in ACTIVE_EVOKED_MODELS for model in models)


AVAILABLE_EVOKED_REGRESSORS = _configured_regressors_for_family("evoked_trf")
if "baseline" in ACTIVE_EVOKED_MODELS:
    AVAILABLE_EVOKED_REGRESSORS = list(dict.fromkeys([
        *AVAILABLE_EVOKED_REGRESSORS,
        *DEFAULT_BASELINE_VIZ_REGRESSORS,
    ]))
AVAILABLE_EVOKED_RECONSTRUCTION_R2_MODELS = [
    str(model) for model in sorted(ACTIVE_EVOKED_MODELS)
    if str(model) != "baseline"
]
AVAILABLE_INDUCED_REGRESSORS_BY_FAMILY = {
    family: _configured_regressors_for_family(family)
    for family in INDUCED_FAMILIES
}
VIZ_OUT_DIR = _viz_out_dir()

EVOKED_JOINT_TARGETS = [f"{_viz_out_dir()}/final/figures/evoked_trf/joint_{reg}.pdf" for reg in AVAILABLE_EVOKED_REGRESSORS]
EVOKED_TOPO_TARGETS = [f"{_viz_out_dir()}/final/figures/evoked_trf/topomap_{reg}.pdf" for reg in AVAILABLE_EVOKED_REGRESSORS]
INDUCED_JOINT_TARGETS = [
    f"{_viz_out_dir()}/final/figures/{family}/joint_{reg}.pdf"
    for family, regs in AVAILABLE_INDUCED_REGRESSORS_BY_FAMILY.items()
    for reg in regs
]
INDUCED_TOPO_TARGETS = [
    f"{_viz_out_dir()}/final/figures/{family}/topomap_{reg}.pdf"
    for family, regs in AVAILABLE_INDUCED_REGRESSORS_BY_FAMILY.items()
    for reg in regs
]
INDUCED_TIMECOURSE_TARGETS = [
    f"{_viz_out_dir()}/final/figures/{family}/timecourse_panel.pdf"
    for family in INDUCED_FAMILIES
]
EVOKED_RECONSTRUCTION_ERP_TARGETS = [
    f"{_viz_out_dir()}/final/figures/evoked_trf/reconstruction_erp_linear.pdf",
]
if _has_evoked_models(["self_rate_sq", "other_rate_sq", "abs_diff_sq"]):
    EVOKED_RECONSTRUCTION_ERP_TARGETS.append(
        f"{_viz_out_dir()}/final/figures/evoked_trf/reconstruction_erp_quadratic.pdf"
    )
EVOKED_RECONSTRUCTION_ERP_R2_TOPO_TARGETS = [
    f"{_viz_out_dir()}/final/figures/evoked_trf/reconstruction_erp_r2_topomap_{model}_vs_baseline.pdf"
    for model in AVAILABLE_EVOKED_RECONSTRUCTION_R2_MODELS
]
FOOOF_PSD_TARGETS = [
    f"{_viz_out_dir()}/final/figures/fooof/psd_quantile_grid.pdf",
    f"{_viz_out_dir()}/final/figures/fooof/psd_quantile_raw_row.pdf",
]


rule figure_evoked_trf_joint_by_regressor:
    input:
        kernel=_evoked_kernel_input,
        clusters=_evoked_cluster_input,
    output:
        f"{_viz_out_dir()}/final/figures/evoked_trf/joint_{{regressor}}.pdf"
    params:
        out_dir=VIZ_OUT_DIR
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {params.out_dir} \
            --family evoked_trf \
            --figure joint \
            --config config/viz.yaml \
            --regressor {wildcards.regressor}
        """


rule figure_evoked_trf_joint:
    input:
        EVOKED_JOINT_TARGETS


rule figure_evoked_trf_topo_by_regressor:
    input:
        kernel=_evoked_kernel_input,
        clusters=_evoked_cluster_input,
    output:
        f"{_viz_out_dir()}/final/figures/evoked_trf/topomap_{{regressor}}.pdf"
    params:
        out_dir=VIZ_OUT_DIR
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {params.out_dir} \
            --family evoked_trf \
            --figure topo \
            --config config/viz.yaml \
            --regressor {wildcards.regressor}
        """


rule figure_evoked_trf_topo:
    input:
        EVOKED_TOPO_TARGETS


rule figure_induced_trf_joint_by_regressor:
    wildcard_constraints:
        family="induced_trf_.+"
    input:
        kernel=_family_kernel_input,
        clusters=_family_cluster_input,
    output:
        f"{_viz_out_dir()}/final/figures/{{family}}/joint_{{regressor}}.pdf"
    params:
        out_dir=VIZ_OUT_DIR
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {params.out_dir} \
            --family {wildcards.family} \
            --figure joint \
            --config config/viz.yaml \
            --regressor {wildcards.regressor}
        """


rule figure_induced_trf_joint:
    input:
        INDUCED_JOINT_TARGETS


rule figure_induced_trf_topo_by_regressor:
    wildcard_constraints:
        family="induced_trf_.+"
    input:
        kernel=_family_kernel_input,
        clusters=_family_cluster_input,
    output:
        f"{_viz_out_dir()}/final/figures/{{family}}/topomap_{{regressor}}.pdf"
    params:
        out_dir=VIZ_OUT_DIR
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {params.out_dir} \
            --family {wildcards.family} \
            --figure topo \
            --config config/viz.yaml \
            --regressor {wildcards.regressor}
        """


rule figure_induced_trf_topo:
    input:
        INDUCED_TOPO_TARGETS


rule figure_evoked_trf_timecourse:
    input:
        _evoked_timecourse_kernel_inputs
    output:
        f"{_viz_out_dir()}/final/figures/evoked_trf/timecourse_panel.pdf"
    params:
        out_dir=VIZ_OUT_DIR
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {params.out_dir} \
            --family evoked_trf \
            --figure timecourse \
            --config config/viz.yaml
        """


rule figure_induced_trf_timecourse:
    wildcard_constraints:
        family="induced_trf_.+"
    input:
        _family_timecourse_kernel_inputs
    output:
        f"{_viz_out_dir()}/final/figures/{{family}}/timecourse_panel.pdf"
    params:
        out_dir=VIZ_OUT_DIR
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {params.out_dir} \
            --family {wildcards.family} \
            --figure timecourse \
            --config config/viz.yaml
        """


rule figure_evoked_trf_reconstruction_erp_linear:
    input:
        # Stats outputs ensure upstream compute/stats are available before rendering.
        expand(
            f"{_viz_out_dir()}/stats/evoked_trf/{{model}}/{{model}}/grand_average_trfs-ave.fif",
            model=["self_rate", "other_rate", "abs_diff"],
        )
    output:
        f"{_viz_out_dir()}/final/figures/evoked_trf/reconstruction_erp_linear.pdf"
    params:
        out_dir=VIZ_OUT_DIR
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {params.out_dir} \
            --family evoked_trf \
            --figure reconstruction_erp_linear \
            --config config/viz.yaml
        """


rule figure_evoked_trf_reconstruction_erp_quadratic:
    input:
        expand(
            f"{_viz_out_dir()}/stats/evoked_trf/{{model}}/{{model}}/grand_average_trfs-ave.fif",
            model=["self_rate_sq", "other_rate_sq", "abs_diff_sq"],
        )
    output:
        f"{_viz_out_dir()}/final/figures/evoked_trf/reconstruction_erp_quadratic.pdf"
    params:
        out_dir=VIZ_OUT_DIR
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {params.out_dir} \
            --family evoked_trf \
            --figure reconstruction_erp_quadratic \
            --config config/viz.yaml
        """


rule figure_evoked_trf_reconstruction_erp_r2_topo_by_model:
    input:
        mean_delta=(
            f"{_viz_out_dir()}/stats/evoked_trf/{{regressor}}/reconstruction_erp_r2_vs_baseline/mean_delta_r2.npy"
        ),
        labels=(
            f"{_viz_out_dir()}/stats/evoked_trf/{{regressor}}/reconstruction_erp_r2_vs_baseline/cluster_labels.npy"
        ),
        clusters=(
            f"{_viz_out_dir()}/stats/evoked_trf/{{regressor}}/reconstruction_erp_r2_vs_baseline/clusters.{_viz_table_ext()}"
        ),
        ch_names=(
            f"{_viz_out_dir()}/stats/evoked_trf/{{regressor}}/reconstruction_erp_r2_vs_baseline/channel_names.npy"
        ),
    output:
        f"{_viz_out_dir()}/final/figures/evoked_trf/reconstruction_erp_r2_topomap_{{regressor}}_vs_baseline.pdf"
    params:
        out_dir=VIZ_OUT_DIR
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {params.out_dir} \
            --family evoked_trf \
            --figure reconstruction_erp_r2_topo \
            --config config/viz.yaml \
            --regressor {wildcards.regressor}
        """


rule figure_evoked_trf_reconstruction_erp_r2_topo:
    input:
        EVOKED_RECONSTRUCTION_ERP_R2_TOPO_TARGETS


rule figure_fooof_psd_panel:
    input:
        trial_psd_spectra=f"{_viz_out_dir()}/fooof/aggregated/trial_psd_spectra.parquet",
        settings=f"{_viz_out_dir()}/fooof/meta/fooof_settings.json",
    output:
        f"{_viz_out_dir()}/final/figures/fooof/psd_quantile_grid.pdf"
    params:
        out_dir=VIZ_OUT_DIR
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {params.out_dir} \
            --family fooof \
            --figure fooof_psd_panel \
            --config config/viz.yaml
        """


rule figure_fooof_psd_raw_row:
    input:
        trial_psd_spectra=f"{_viz_out_dir()}/fooof/aggregated/trial_psd_spectra.parquet",
        settings=f"{_viz_out_dir()}/fooof/meta/fooof_settings.json",
    output:
        f"{_viz_out_dir()}/final/figures/fooof/psd_quantile_raw_row.pdf"
    params:
        out_dir=VIZ_OUT_DIR
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {params.out_dir} \
            --family fooof \
            --figure fooof_psd_raw_row \
            --config config/viz.yaml
        """


rule figure_fooof_psd:
    input:
        FOOOF_PSD_TARGETS


rule figures_all:
    input:
        EVOKED_JOINT_TARGETS
        + EVOKED_TOPO_TARGETS
        #+ INDUCED_JOINT_TARGETS
        #+ INDUCED_TOPO_TARGETS
        + [
            f"{_viz_out_dir()}/final/figures/evoked_trf/timecourse_panel.pdf",
            *EVOKED_RECONSTRUCTION_ERP_TARGETS,
        ]
        + EVOKED_RECONSTRUCTION_ERP_R2_TOPO_TARGETS
        #+ INDUCED_TIMECOURSE_TARGETS
        + FOOOF_PSD_TARGETS
