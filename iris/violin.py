"""
Violin plot functionality for Iris — free-spectral model diagnostics.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Reuse the same colour cycle as corner plots
from .corner import _DEFAULT_COLORS


def make_violin(
    datasets: list[np.ndarray],
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    figsize: tuple = (10, 5),
    title: str | None = None,
    xlabel: str = "frequency bin",
    ylabel: str = r"$\log_{10}|\mathrm{PSD}\;\;[\mathrm{s}^2]|$",
    legend_fontsize: int | float = 12,
    widths: float = 0.8,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Make a violin plot for one or more free-spectral sample sets.

    Parameters
    ----------
    datasets : list of ndarray, shape (nsamples, nfreqs)
        One array per dataset.  Each column becomes one violin at that
        frequency-bin position.
    labels : list of str, optional
        Legend label per dataset.  Only shown when more than one dataset
        is present.
    colors : list of str, optional
        Matplotlib colour strings, one per dataset.  Auto-assigned if None.
    figsize : tuple, optional
        Figure size passed to ``plt.figure``.  Ignored when *fig* is given.
    title : str, optional
        Figure title.
    xlabel : str
        x-axis label.
    ylabel : str
        y-axis label.
    legend_fontsize : int or float
        Font size for the legend.  Default 12.
    widths : float
        Width of each violin (passed to ``violinplot``).
    fig : matplotlib Figure, optional
        Existing figure to draw into.  If None a new figure is created.
    ax : matplotlib Axes, optional
        Existing axes to draw into.  If None a new axes is created (or taken
        from *fig*).

    Returns
    -------
    fig : matplotlib Figure
    """
    n = len(datasets)

    # --- colours ---
    if colors is None:
        colors = [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i in range(n)]

    # --- labels / legend ---
    show_legend = n > 1
    if labels is None:
        labels = [f"Run {i + 1}" for i in range(n)]

    # --- figure / axes ---
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.gca() if fig.axes else fig.add_subplot(111)

    # x positions: 0-based frequency-bin indices
    nfreqs = datasets[0].shape[1]
    positions = np.arange(nfreqs)

    # --- draw ---
    if n == 1:
        _draw_violins(ax, datasets[0], positions, colors[0], widths=widths)
    elif n == 2:
        # split violin: first dataset on the left half, second on the right
        _draw_violins(ax, datasets[0], positions, colors[0],
                      widths=widths, side="low")
        _draw_violins(ax, datasets[1], positions, colors[1],
                      widths=widths, side="high")
    else:
        # more than two: overlay with reduced alpha
        for data, color in zip(datasets, colors):
            _draw_violins(ax, data, positions, color, widths=widths, alpha=0.5)

    # --- cosmetics ---
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(i) for i in positions])
    ax.grid(True)
    if title:
        ax.set_title(title)

    if show_legend:
        handles = [
            mpatches.Patch(facecolor=c, label=l, alpha=0.7)
            for c, l in zip(colors, labels)
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=legend_fontsize)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _draw_violins(ax, data, positions, color, widths=0.8, side=None, alpha=0.7):
    """
    Draw violins for *data* (nsamples, nfreqs) on *ax* and colour them.

    Parameters
    ----------
    side : {'low', 'high', None}
        Passed to ``violinplot`` for split-violin rendering.  None draws
        full violins.
    """
    kw = dict(
        dataset=data,
        positions=positions,
        showextrema=False,
        widths=widths,
    )
    if side is not None:
        kw["side"] = side

    parts = ax.violinplot(**kw)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(alpha)
