"""
Trace plot functionality for Plotsar.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def make_trace(
    catalog: dict,
    params: list | None = None,
    thin: int = 1,
    truths: dict | None = None,
    figsize: tuple | None = None,
    ncols: int = 2,
    plot_kwargs: dict | None = None,
) -> plt.Figure:
    """
    Make trace plots (sample index vs parameter value) for selected parameters.

    Parameters
    ----------
    catalog : dict
        Parameter catalog as returned by ``build_param_catalog``.
    params : list of str, optional
        Flat parameter names to plot.  If None, all non-Fourier-coefficient
        parameters are plotted.
    thin : int
        Thinning factor applied to samples before plotting.
    truths : dict, optional
        Mapping from flat parameter name → truth value.  Drawn as a
        horizontal dashed line on each trace panel.
    figsize : tuple, optional
        Figure size ``(width, height)``.  Auto-sized if None.
    ncols : int
        Number of columns in the panel grid.
    plot_kwargs : dict, optional
        Extra keyword arguments forwarded to ``ax.plot``.

    Returns
    -------
    fig : matplotlib Figure
    """
    _SKIP_PREFIXES = ("z", "a", "a_gwb", "a_det")

    if params is not None:
        missing = [p for p in params if p not in catalog]
        if missing:
            raise KeyError(f"Parameter(s) not in catalog: {missing}")
        selected = list(params)
    else:
        selected = [
            k for k in catalog
            if not any(k == pfx or k.startswith(pfx + "_") for pfx in _SKIP_PREFIXES)
        ]

    if not selected:
        raise ValueError("No parameters selected for trace plot.")

    nparams = len(selected)
    nrows = int(np.ceil(nparams / ncols))

    if figsize is None:
        figsize = (5 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    pkw = dict(lw=0.6, alpha=0.8, color="steelblue")
    if plot_kwargs:
        pkw.update(plot_kwargs)

    for idx, key in enumerate(selected):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        s = catalog[key]["samples"][::thin]
        ax.plot(s, **pkw)

        if truths and key in truths:
            ax.axhline(truths[key], color="firebrick", ls="--", lw=1.2, label="truth")
            ax.legend(fontsize=8, loc="upper right")

        ax.set_xlabel("sample index", fontsize=9)
        ax.set_ylabel(catalog[key]["label"], fontsize=10)
        ax.tick_params(labelsize=8)

    # hide empty panels
    for idx in range(nparams, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    return fig
