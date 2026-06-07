"""
Corner plot functionality for Iris.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import corner as corner_lib
    _CORNER_AVAILABLE = True
except ImportError:
    _CORNER_AVAILABLE = False

# Default color cycle for multiple datasets
_DEFAULT_COLORS = [
    "#2196F3",   # blue  (primary)
    "#F44336",   # red
    "#4CAF50",   # green
    "#FF9800",   # orange
    "#9C27B0",   # purple
    "#00BCD4",   # cyan
    "#795548",   # brown
]


def make_corner(
    catalog: dict,
    params: list | None = None,
    thin: int = 1,
    truths: dict | None = None,
    fig: plt.Figure | None = None,
    extra_catalogs: list[dict] | None = None,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    corner_kwargs: dict | None = None,
) -> plt.Figure:
    """
    Make a corner plot, optionally overlaying multiple sample sets.

    Parameters
    ----------
    catalog : dict
        Primary parameter catalog as returned by ``build_param_catalog``.
    params : list of str, optional
        Flat parameter names (catalog keys) to include.  If None, all
        non-Fourier-coefficient parameters are used.
    thin : int
        Thinning factor applied to all sample sets.
    truths : dict, optional
        ``{flat_key: truth_value}`` drawn as dashed lines on the plot.
    fig : matplotlib Figure, optional
        Existing figure to draw into.
    extra_catalogs : list of dict, optional
        Additional parameter catalogs (built from other sample dicts via
        ``build_param_catalog``) to overlay on the same corner plot.
        Each catalog is drawn in a different color.  Catalogs that are
        missing any of the selected parameters for a run are skipped for
        that run with a warning.
    labels : list of str, optional
        Legend labels, one per dataset in order: primary first, then each
        entry in ``extra_catalogs``.  If None, datasets are labelled
        "Run 1", "Run 2", … only when there is more than one dataset.
    colors : list of str, optional
        Colors for each dataset (primary first).  Auto-assigned from a
        built-in cycle if None.
    corner_kwargs : dict, optional
        Additional keyword arguments forwarded to every ``corner.corner``
        call.  Per-dataset color and ``fig`` are set automatically and
        should not be included here.

    Returns
    -------
    fig : matplotlib Figure
    """
    if not _CORNER_AVAILABLE:
        raise ImportError(
            "The 'corner' package is required for corner plots.  "
            "Install it with: pip install corner"
        )

    extra_catalogs = extra_catalogs or []
    all_catalogs = [catalog] + list(extra_catalogs)
    n_datasets = len(all_catalogs)

    # --- parameter selection (based on primary catalog) ---
    selected = _select_params(catalog, params)
    if not selected:
        raise ValueError("No parameters selected for corner plot.")

    # --- labels ---
    if labels is not None:
        if len(labels) != n_datasets:
            raise ValueError(
                f"len(labels)={len(labels)} but there are {n_datasets} dataset(s)."
            )
        show_legend = True
    else:
        labels = [f"Run {i + 1}" for i in range(n_datasets)]
        show_legend = n_datasets > 1

    # --- colors ---
    if colors is not None:
        if len(colors) < n_datasets:
            raise ValueError(
                f"len(colors)={len(colors)} but there are {n_datasets} dataset(s)."
            )
    else:
        colors = [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i in range(n_datasets)]

    # --- truth values ---
    truth_vals = None
    if truths:
        truth_vals = [truths.get(k, None) for k in selected]

    # --- shared corner kwargs ---
    axis_labels = [catalog[k]["label"] for k in selected]
    base_kw = dict(
        labels=axis_labels,
        show_titles=True,
        title_kwargs={"fontsize": 11},
        label_kwargs={"fontsize": 11},
        plot_contours=True,
        plot_datapoints=False,
        fill_contours=True,
        smooth=1.0,
        bins=30,
        quantiles=[0.16, 0.5, 0.84],
    )
    if corner_kwargs:
        base_kw.update(corner_kwargs)

    # --- draw datasets ---
    for i, (cat, color, label) in enumerate(zip(all_catalogs, colors, labels)):
        # check which selected params are available in this catalog
        missing = [k for k in selected if k not in cat]
        if missing:
            warnings.warn(
                f"Dataset '{label}' is missing parameter(s) {missing} — skipping this dataset.",
                stacklevel=3,
            )
            continue

        data = np.column_stack([cat[k]["samples"][::thin] for k in selected])

        kw = dict(**base_kw, color=color)

        # only show truths on the first dataset to avoid clutter
        kw["truths"] = truth_vals if i == 0 else None

        # titles/quantiles only make sense on the first (primary) dataset
        if i > 0:
            kw["show_titles"] = False
            kw["quantiles"] = []

        if fig is not None or i > 0:
            kw["fig"] = fig

        fig = corner_lib.corner(data, **kw)

    # --- legend ---
    if show_legend and fig is not None:
        nparams = len(selected)
        # top-right corner of the grid is the empty axes at position [0, nparams-1]
        legend_ax = fig.axes[nparams - 1]
        handles = [
            mpatches.Patch(facecolor=c, label=l, alpha=0.7)
            for c, l in zip(colors, labels)
        ]
        legend_ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=10,
            framealpha=0.85,
            borderpad=0.8,
        )

    return fig


def _select_params(catalog: dict, params: list | None) -> list:
    """Return an ordered list of catalog keys to plot."""
    _SKIP_PREFIXES = ("z", "a", "a_gwb", "a_det")

    if params is not None:
        missing = [p for p in params if p not in catalog]
        if missing:
            raise KeyError(f"Parameter(s) not found in primary catalog: {missing}")
        return list(params)

    return [
        k for k in catalog
        if not any(k == pfx or k.startswith(pfx + "_") for pfx in _SKIP_PREFIXES)
    ]
