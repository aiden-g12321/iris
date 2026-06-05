"""
Plotsar — plotting package for Prometheus PTA analysis outputs.

Quick start::

    from plotsar import Plotsar

    plotter = Plotsar(model, samples)
    fig = plotter.corner()
    fig = plotter.trace()
    fig = plotter.skymap()
"""

from .plotter import Plotsar
from .utils import (
    build_param_catalog,
    build_truths_catalog,
    CW_PARAM_NAMES,
    CW_PARAM_LABELS,
    POWERLAW_PARAM_NAMES,
    POWERLAW_PARAM_LABELS,
    costheta_phi_to_lonlat,
    theta_phi_to_lonlat,
)
from .corner import make_corner
from .trace import make_trace
from .skymap import make_skymap

__all__ = [
    "Plotsar",
    "build_param_catalog",
    "build_truths_catalog",
    "make_corner",
    "make_trace",
    "make_skymap",
    "CW_PARAM_NAMES",
    "CW_PARAM_LABELS",
    "POWERLAW_PARAM_NAMES",
    "POWERLAW_PARAM_LABELS",
    "costheta_phi_to_lonlat",
    "theta_phi_to_lonlat",
]
