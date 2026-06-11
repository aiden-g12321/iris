"""
Utility functions for parameter catalog construction, coordinate
transforms, and label formatting used throughout Iris.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Known CW parameter names and LaTeX labels for an 8-parameter CW source
# (matches the ordering in prometheus/deterministic.py)
# ---------------------------------------------------------------------------

CW_PARAM_NAMES = [
    "log10_mc",
    "log10_fgw",
    "cos_inc",
    "psi",
    "log10_h",
    "cos_gwtheta",
    "gwphi",
    "phase0",
]

CW_PARAM_LABELS = [
    r"$\log_{10}\mathcal{M}_c$",
    r"$\log_{10}f_\mathrm{gw}$",
    r"$\cos\iota$",
    r"$\psi$",
    r"$\log_{10}h$",
    r"$\cos\theta_\mathrm{gw}$",
    r"$\phi_\mathrm{gw}$",
    r"$\phi_0$",
]

# Per-pulsar CW deterministic-model keys and their LaTeX labels
PER_PSR_CW_LABELS = {
    "psr_phases": r"$\Phi$",
    "psr_dists": r"$L\;\;[\mathrm{kpc}]$",
}

# Power-law spectrum param names / labels (2-param model)
POWERLAW_PARAM_NAMES = ["log10_A", "gamma"]
POWERLAW_PARAM_LABELS = [r"$\log_{10}A$", r"$\gamma$"]

# ---------------------------------------------------------------------------
# Parameter catalog
# ---------------------------------------------------------------------------


def _latex_label(name: str) -> str:
    """Convert a snake_case parameter name to a passable LaTeX label."""
    replacements = {
        "log10_": r"$\log_{10}$",
        "cos_": r"$\cos$",
        "_": " ",
    }
    label = name
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def build_param_catalog(model, samples: dict, param_names: dict | None = None) -> dict:
    """
    Build a flat catalog of parameter names, LaTeX labels, and 1-D sample
    arrays from the PTAModel and samples dictionary.

    Parameters
    ----------
    model : PTAModel
        A Prometheus PTAModel instance.
    samples : dict
        Dictionary mapping sample-site names to arrays of shape
        ``(nsamples, ...)`` as returned by NumPyro / prometheus utilities.
    param_names : dict, optional
        Override parameter names per sample key.  Keys are sample-site names
        (e.g. ``'cw_source'``); values are lists of parameter name strings.
        For per-pulsar arrays the list should contain the names for **one**
        pulsar (they will be replicated across pulsars with pulsar name tags).

    Returns
    -------
    catalog : dict
        Mapping from flat parameter name → dict with keys:
        ``'label'`` (str), ``'samples'`` (1-D numpy array).
    """
    catalog = {}
    psr_names = list(model.data.psr_names)
    user_names = param_names or {}

    for key, arr in samples.items():
        arr = np.array(arr)
        if arr.ndim == 0:
            continue
        nsamples = arr.shape[0]

        # --- pulsar-indexed 2-D array: (nsamples, npsrs, nparams_per_psr) ---
        if arr.ndim == 3 and arr.shape[1] == len(psr_names):
            npsrs, nparams = arr.shape[1], arr.shape[2]
            per_psr_names = user_names.get(key, [f"p{i}" for i in range(nparams)])
            # fall back to power-law names when shape matches
            if len(per_psr_names) == nparams and nparams == 2 and key not in user_names:
                per_psr_names = POWERLAW_PARAM_NAMES
            free_spectral = nparams > 2 and key not in user_names
            for pi, pname in enumerate(per_psr_names[:nparams]):
                for psi, psrname in enumerate(psr_names):
                    flat_name = f"{key}_{psrname}_{pname}"
                    if free_spectral:
                        label = _free_spectral_label(pi, psrname)
                    else:
                        label = _make_label(key, pname, psrname)
                    catalog[flat_name] = {
                        "label": label,
                        "samples": arr[:, psi, pi],
                        "key": key,
                        "param_idx": pi,
                        "psr_idx": psi,
                        "psr_name": psrname,
                        "free_spectral": free_spectral,
                    }

        # --- per-pulsar 1-D array: (nsamples, npsrs) ---
        elif arr.ndim == 2 and arr.shape[1] == len(psr_names):
            for psi, psrname in enumerate(psr_names):
                flat_name = f"{key}_{psrname}"
                label = _make_label(key, None, psrname)
                catalog[flat_name] = {
                    "label": label,
                    "samples": arr[:, psi],
                    "key": key,
                    "psr_idx": psi,
                    "psr_name": psrname,
                }

        # --- plain parameter vector: (nsamples, nparams) ---
        elif arr.ndim == 2:
            nparams = arr.shape[1]
            names_for_key = _resolve_param_names(key, nparams, user_names)
            # free-spectral only when names were auto-generated (p0, p1, ...)
            # i.e. _resolve_param_names didn't match any known model (CW, power-law, etc.)
            free_spectral = all(n == f"p{i}" for i, n in enumerate(names_for_key))
            for pi, pname in enumerate(names_for_key):
                flat_name = f"{key}_{pname}" if pname != key else pname
                if free_spectral:
                    label = _free_spectral_label(pi, None)
                else:
                    label = _make_label(key, pname, None)
                catalog[flat_name] = {
                    "label": label,
                    "samples": arr[:, pi],
                    "key": key,
                    "param_idx": pi,
                    "free_spectral": free_spectral,
                }

        # --- scalar per sample: (nsamples,) ---
        elif arr.ndim == 1:
            catalog[key] = {
                "label": _latex_label(key),
                "samples": arr,
                "key": key,
            }

    return catalog


def _resolve_param_names(key: str, nparams: int, user_names: dict) -> list:
    """Return a list of length *nparams* of param names for sample key."""
    if key in user_names:
        names = list(user_names[key])
        if len(names) < nparams:
            names += [f"p{i}" for i in range(len(names), nparams)]
        return names[:nparams]
    if nparams == len(CW_PARAM_NAMES):
        return CW_PARAM_NAMES
    if nparams == len(POWERLAW_PARAM_NAMES):
        return POWERLAW_PARAM_NAMES
    return [f"p{i}" for i in range(nparams)]


def _free_spectral_label(pi: int, psrname: str | None) -> str:
    """Return a LaTeX label for free-spectral bin *pi* (1-based)."""
    idx = pi + 1
    if psrname:
        return rf"$\log_{{10}}\rho_{{{idx}}}\;\;[{psrname}]$"
    return rf"$\log_{{10}}\rho_{{{idx}}}\;\;[\mathrm{{GWB}}]$"


def _make_label(key: str, pname: str | None, psrname: str | None) -> str:
    """Build a human-readable / LaTeX label."""
    # prefer known CW labels
    if pname in CW_PARAM_NAMES:
        base = CW_PARAM_LABELS[CW_PARAM_NAMES.index(pname)]
    elif pname in POWERLAW_PARAM_NAMES:
        base = POWERLAW_PARAM_LABELS[POWERLAW_PARAM_NAMES.index(pname)]
    elif pname is not None:
        base = _latex_label(pname)
    elif key in PER_PSR_CW_LABELS:
        base = PER_PSR_CW_LABELS[key]
    else:
        base = _latex_label(key)

    if psrname:
        return f"{base}\n({psrname})"
    return base


def build_truths_catalog(model, truths: dict, param_names: dict | None = None) -> dict:
    """
    Convert a truths dictionary (same nested format as the samples dict) into
    a flat ``{catalog_key: scalar}`` mapping that aligns with the catalog
    produced by ``build_param_catalog``.

    Parameters
    ----------
    model : PTAModel
        Prometheus PTAModel used for pulsar names and shapes.
    truths : dict
        Injected / truth values in the same key-value scheme as the samples
        dict.  Values should be scalars or arrays matching the per-sample
        shape, e.g.::

            {
                'gwb':       np.array([-15.0, 13/3]),
                'psr_noise': np.array([[log10_A_0, gamma_0],
                                       [log10_A_1, gamma_1], ...]),
                'cw_source': np.array([log10_mc, log10_fgw, ...8 values...]),
            }

    param_names : dict, optional
        Same override dict accepted by ``build_param_catalog``.

    Returns
    -------
    flat : dict
        ``{flat_catalog_key: float}`` mapping.
    """
    psr_names = list(model.data.psr_names)
    user_names = param_names or {}
    flat = {}

    for key, val in truths.items():
        arr = np.atleast_1d(np.array(val, dtype=float))

        # (npsrs, nparams) — per-pulsar parameter matrix
        if arr.ndim == 2 and arr.shape[0] == len(psr_names):
            npsrs, nparams = arr.shape
            per_psr_names = user_names.get(key, [f"p{i}" for i in range(nparams)])
            if nparams == 2 and key not in user_names:
                per_psr_names = POWERLAW_PARAM_NAMES
            for pi, pname in enumerate(per_psr_names[:nparams]):
                for psi, psrname in enumerate(psr_names):
                    flat[f"{key}_{psrname}_{pname}"] = float(arr[psi, pi])

        # (npsrs,) — one scalar truth per pulsar
        elif arr.ndim == 1 and len(arr) == len(psr_names):
            # ambiguous: could be per-pulsar scalar OR a plain param vector.
            # Treat as per-pulsar scalar only if the key is in samples with
            # matching shape (we rely on a naming heuristic here).
            for psi, psrname in enumerate(psr_names):
                flat[f"{key}_{psrname}"] = float(arr[psi])

        # (nparams,) — plain parameter vector
        elif arr.ndim == 1:
            nparams = len(arr)
            names_for_key = _resolve_param_names(key, nparams, user_names)
            for pi, pname in enumerate(names_for_key):
                flat_name = f"{key}_{pname}" if pname != key else pname
                flat[flat_name] = float(arr[pi])

        # scalar
        else:
            flat[key] = float(arr.item())

    return flat


# ---------------------------------------------------------------------------
# Sky-coordinate helpers
# ---------------------------------------------------------------------------


def costheta_phi_to_lonlat(cos_theta, phi):
    """
    Convert Prometheus (cos_theta, phi) sky coordinates to Matplotlib
    Mollweide (lon, lat) in radians.

    Prometheus convention:
        theta  — polar angle from north pole, [0, pi]
        phi    — azimuthal angle, [0, 2*pi)
        cos_gwtheta stored as cos(theta) in [-1, 1]

    Mollweide convention (matplotlib):
        lon    — longitude in (-pi, pi]
        lat    — latitude in [-pi/2, pi/2]
    """
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    lat = np.pi / 2.0 - theta           # declination-like
    lon = phi - np.pi                   # centre on 0
    # wrap lon to (-pi, pi]
    lon = (lon + np.pi) % (2 * np.pi) - np.pi
    return lon, lat


def theta_phi_to_lonlat(theta, phi):
    """
    Convert (theta, phi) where theta is polar angle [0, pi] to Mollweide
    (lon, lat) in radians.
    """
    return costheta_phi_to_lonlat(np.cos(theta), phi)
