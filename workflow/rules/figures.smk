from pathlib import Path


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


REGRESSORS = _viz_regressors()

if not REGRESSORS:
    raise ValueError("config/viz.yaml reporting.trf.regressors must define at least one TRF regressor")


def _induced_bands():
    ind_cfg = viz_induced_trf_config
    bands = ind_cfg.get("bands")
    if bands:
        return [str(b) for b in bands]
    bands_hz = ind_cfg.get("bands_hz", {}) or {}
    return sorted(str(k) for k in bands_hz.keys())


INDUCED_FAMILIES = [f"induced_trf_{band}" for band in _induced_bands()]


def _trf_model_for_regressor(regressor: str) -> str:
    return "baseline" if str(regressor) == "emg" else str(regressor)


def _evoked_kernel_input(wildcards):
    model = _trf_model_for_regressor(str(wildcards.regressor))
    return f"{_viz_out_dir()}/stats/evoked_trf/{model}/{wildcards.regressor}/grand_average_trfs-ave.fif"


def _evoked_cluster_input(wildcards):
    model = _trf_model_for_regressor(str(wildcards.regressor))
    return f"{_viz_out_dir()}/stats/evoked_trf/{model}/{wildcards.regressor}/clusters.{_viz_table_ext()}"


def _evoked_timecourse_kernel_inputs(_wildcards):
    return [
        f"{_viz_out_dir()}/stats/evoked_trf/{_trf_model_for_regressor(reg)}/{reg}/grand_average_trfs-ave.fif"
        for reg in REGRESSORS
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
        for reg in REGRESSORS
    ]


def _available_regressors_for_family(family: str):
    out = []
    for reg in REGRESSORS:
        model = _trf_model_for_regressor(reg)
        kernel = Path(f"{_viz_out_dir()}/stats/{family}/{model}/{reg}/grand_average_trfs-ave.fif")
        clusters = Path(f"{_viz_out_dir()}/stats/{family}/{model}/{reg}/clusters.{_viz_table_ext()}")
        if kernel.exists() and clusters.exists():
            out.append(reg)
    return out


AVAILABLE_EVOKED_REGRESSORS = _available_regressors_for_family("evoked_trf")
AVAILABLE_INDUCED_REGRESSORS_BY_FAMILY = {
    family: _available_regressors_for_family(family)
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
    f"{_viz_out_dir()}/final/figures/evoked_trf/reconstruction_erp_quadratic.pdf",
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


rule figures_all:
    input:
        EVOKED_JOINT_TARGETS
        + EVOKED_TOPO_TARGETS
        + INDUCED_JOINT_TARGETS
        + INDUCED_TOPO_TARGETS
        + [
            f"{_viz_out_dir()}/final/figures/evoked_trf/timecourse_panel.pdf",
            *EVOKED_RECONSTRUCTION_ERP_TARGETS,
        ]
        + INDUCED_TIMECOURSE_TARGETS
