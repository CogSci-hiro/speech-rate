from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from rate.viz.theme import apply_theme


apply_theme()


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    family: str,
    name: str,
    qc: bool = False,
    dpi: int = 300,
    formats: Sequence[str] = ("pdf", "png", "tiff"),
) -> None:
    """
    Save figure to final directory with multi-format export.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure.
    out_dir : Path
        Root output directory.
    family : str
        Analysis family (e.g., "evoked_trf").
    name : str
        Figure base name.
    qc : bool
        If True, save under qc/.
    dpi : int
        Resolution.
    formats : sequence of str
        File formats.
    """
    base_path = (
        out_dir / "final" / ("qc" if qc else "figures") / family
    )
    base_path.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(base_path / f"{name}.{fmt}", dpi=dpi, bbox_inches="tight")

    plt.close(fig)


def add_time_cluster_shading(
    ax: plt.Axes,
    times: np.ndarray,
    clusters: Optional[Iterable[Tuple[float, float]]] = None,
    color: str = "grey",
    alpha: float = 0.2,
) -> None:
    """
    Add shaded regions for significant time clusters.

    Parameters
    ----------
    ax : plt.Axes
        Time course axis.
    times : ndarray
        Time vector in seconds.
    clusters : iterable of (t_start, t_end)
        Significant cluster intervals.
    color : str
        Shading color.
    alpha : float
        Transparency.
    """
    if clusters is None:
        return

    for t_start, t_end in clusters:
        ax.axvspan(t_start, t_end, color=color, alpha=alpha)
