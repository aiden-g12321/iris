"""
Sky map (Mollweide projection) functionality for Nyota.

Convention reminder (Prometheus):
    theta  – polar angle from north pole in [0, π]
    phi    – azimuthal angle in [0, 2π)
    Pulsar positions are stored as data.psr_theta, data.psr_phi.
    CW source sky location is stored as cos_gwtheta (index 5) and
    gwphi (index 6) inside the 8-parameter det_params array.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from .utils import (
    costheta_phi_to_lonlat,
    theta_phi_to_lonlat,
    CW_PARAM_NAMES,
)


def make_skymap(
    model,
    catalog: dict,
    samples: dict,
    show_pulsars: bool = True,
    show_cw_samples: bool = True,
    show_cw_contours: bool = True,
    cw_sample_key: str | None = None,
    sky_params: tuple | None = None,
    thin: int = 1,
    ncontour_bins: int = 50,
    psr_kwargs: dict | None = None,
    cw_kwargs: dict | None = None,
    figsize: tuple = (12, 6),
    title: str | None = None,
    truths: dict | None = None,
) -> plt.Figure:
    """
    Make a Mollweide sky map showing pulsar locations, optional CW sky-location
    samples, and optional sky-parameter samples for any (cos_theta, phi) pair.

    Parameters
    ----------
    model : PTAModel
        Prometheus PTAModel instance used to retrieve pulsar sky locations
        and names.
    catalog : dict
        Parameter catalog from ``build_param_catalog``.
    samples : dict
        Raw samples dictionary (same dict passed to ``Nyota``).
    show_pulsars : bool
        Whether to plot pulsar sky positions.
    show_cw_samples : bool
        Whether to scatter-plot CW sky-location samples when available.
    show_cw_contours : bool
        Whether to overlay 2-D density contours on CW sky-location samples.
    cw_sample_key : str, optional
        Sample key containing the 8-parameter CW parameter array.  If None,
        Nyota auto-detects it by looking at the deterministic model name.
    sky_params : tuple of (str, str), optional
        Pair of flat catalog keys ``(cos_theta_key, phi_key)`` for an
        arbitrary sky location parameter pair to overplot.  Plotted as a
        separate scatter on top of the CW samples.
    thin : int
        Thinning factor applied to samples.
    ncontour_bins : int
        Number of bins along each axis used to build the 2-D density grid
        for contour plotting.
    psr_kwargs : dict, optional
        Keyword arguments forwarded to the pulsar scatter call.
    cw_kwargs : dict, optional
        Keyword arguments forwarded to the CW sample scatter call.
    figsize : tuple
        Figure size.
    title : str, optional
        Figure title.
    truths : dict, optional
        Mapping ``{'cos_gwtheta': val, 'gwphi': val}`` or a pair of
        ``(cos_theta, phi)`` truth values to mark on the map.

    Returns
    -------
    fig : matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True, lw=0.4, alpha=0.4)

    # --- Pulsar positions ---
    if show_pulsars:
        plon, plat = theta_phi_to_lonlat(
            np.array(model.data.psr_theta),
            np.array(model.data.psr_phi),
        )
        pkw = dict(marker="*", s=120, color="gold", edgecolors="k",
                   linewidths=0.5, zorder=5, label="Pulsars")
        if psr_kwargs:
            pkw.update(psr_kwargs)
        ax.scatter(plon, plat, **pkw)

        # annotate each pulsar
        for name, lo, la in zip(model.data.psr_names, plon, plat):
            ax.annotate(
                name,
                xy=(lo, la),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=6,
                color="gold",
                zorder=6,
            )

    # --- CW sky location samples ---
    cw_lon, cw_lat = _get_cw_lonlat(model, samples, catalog, cw_sample_key, thin)
    if cw_lon is not None and (show_cw_samples or show_cw_contours):
        if show_cw_contours:
            _plot_density_contours(ax, cw_lon, cw_lat, ncontour_bins)

        if show_cw_samples:
            ckw = dict(s=1, alpha=0.15, color="royalblue", zorder=3,
                       label="CW sky samples")
            if cw_kwargs:
                ckw.update(cw_kwargs)
            ax.scatter(cw_lon, cw_lat, **ckw)

    # --- Arbitrary sky parameter pair ---
    if sky_params is not None:
        cos_theta_key, phi_key = sky_params
        if cos_theta_key in catalog and phi_key in catalog:
            ct = catalog[cos_theta_key]["samples"][::thin]
            phi = catalog[phi_key]["samples"][::thin]
            slon, slat = costheta_phi_to_lonlat(ct, phi)
            ax.scatter(slon, slat, s=1, alpha=0.15, color="darkorange", zorder=3,
                       label=f"{cos_theta_key} / {phi_key}")

    # --- Truth marker ---
    if truths:
        ct_truth = truths.get("cos_gwtheta")
        phi_truth = truths.get("gwphi")
        if ct_truth is not None and phi_truth is not None:
            tlon, tlat = costheta_phi_to_lonlat(
                np.atleast_1d(ct_truth), np.atleast_1d(phi_truth)
            )
            ax.scatter(tlon, tlat, marker="+", s=200, color="firebrick",
                       linewidths=2, zorder=10, label="CW truth")

    # --- Formatting ---
    ax.set_xlabel("Right Ascension / $\\phi$ (rad)", fontsize=10)
    ax.set_ylabel("Declination / $\\theta$ (rad)", fontsize=10)

    # longitude tick labels in [0, 2π] rather than [-π, π]
    xtick_locs = np.linspace(-np.pi, np.pi, 7)[1:-1]
    xtick_labels = [f"{(x + np.pi):.2f}" for x in xtick_locs]
    ax.set_xticks(xtick_locs)
    ax.set_xticklabels(xtick_labels, fontsize=8)

    handles, labels_leg = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels_leg, loc="lower right",
                  fontsize=9, framealpha=0.7)

    if title:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_cw_lonlat(model, samples, catalog, cw_sample_key, thin):
    """
    Extract CW sky location (lon, lat) from samples if available.

    Returns (None, None) if no CW model / samples are found.
    """
    # try the det_model name if not specified
    if cw_sample_key is None:
        if hasattr(model, "det_model") and model.det_model is not None:
            cw_sample_key = model.det_model.name
        else:
            return None, None

    if cw_sample_key not in samples:
        return None, None

    arr = np.array(samples[cw_sample_key])[::thin]
    if arr.ndim != 2 or arr.shape[1] != len(CW_PARAM_NAMES):
        # not an 8-param CW array; check catalog for named sky params instead
        cos_key = f"{cw_sample_key}_cos_gwtheta"
        phi_key = f"{cw_sample_key}_gwphi"
        if cos_key in catalog and phi_key in catalog:
            ct = catalog[cos_key]["samples"][::thin]
            phi = catalog[phi_key]["samples"][::thin]
            return costheta_phi_to_lonlat(ct, phi)
        return None, None

    cos_theta_idx = CW_PARAM_NAMES.index("cos_gwtheta")
    phi_idx = CW_PARAM_NAMES.index("gwphi")
    cos_theta = arr[:, cos_theta_idx]
    phi = arr[:, phi_idx]
    return costheta_phi_to_lonlat(cos_theta, phi)


def _plot_density_contours(ax, lon, lat, nbins):
    """Overlay a smooth 2-D density contour on the Mollweide axes."""
    lon_bins = np.linspace(-np.pi, np.pi, nbins + 1)
    lat_bins = np.linspace(-np.pi / 2, np.pi / 2, nbins + 1)
    H, xedges, yedges = np.histogram2d(lon, lat, bins=[lon_bins, lat_bins])
    H = H.T  # shape (nlat, nlon)

    # smooth slightly
    from scipy.ndimage import gaussian_filter
    H = gaussian_filter(H, sigma=1.5)

    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(x_centers, y_centers)

    levels_frac = [0.1, 0.5, 0.9]
    H_sorted = np.sort(H.ravel())[::-1]
    H_cumsum = np.cumsum(H_sorted)
    H_total = H_cumsum[-1]
    levels = [
        H_sorted[np.searchsorted(H_cumsum, f * H_total)]
        for f in levels_frac
    ]
    levels = sorted(set(levels))

    ax.contour(X, Y, H, levels=levels, colors="royalblue",
               linewidths=[0.8, 1.2, 1.6], alpha=0.85, zorder=4)
