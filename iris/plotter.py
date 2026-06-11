"""
Main Iris class — entry point for all Prometheus-specific plots.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .utils import build_param_catalog, build_truths_catalog
from .corner import make_corner
from .trace import make_trace
from .skymap import make_skymap
from .violin import make_violin


class Iris:
    """
    Plotting interface for Prometheus PTAModel MCMC outputs.

    Given a PTAModel and a dictionary of MCMC samples Iris automatically
    introspects model structure (pulsar names, sky locations, parameter shapes)
    to build labelled corner plots, trace plots, and sky maps with minimal
    user input.

    Parameters
    ----------
    model : PTAModel
        A Prometheus ``PTAModel`` instance that was used to produce the samples.
    samples : dict
        Dictionary of MCMC samples as produced by NumPyro / Prometheus.
        Keys are parameter-site names; values are arrays of shape
        ``(nsamples, ...)``.
    param_names : dict, optional
        Override the automatically inferred parameter names.  Keys are
        sample-site names (matching keys in *samples*); values are lists of
        strings giving parameter names **for one set** (per-pulsar arrays
        only need names for one pulsar).

        Example::

            param_names = {
                'psr_noise': ['log10_A', 'gamma'],
                'cw_source': ['log10_mc', 'log10_fgw', 'cos_inc',
                               'psi', 'log10_h', 'cos_gwtheta',
                               'gwphi', 'phase0'],
            }

    Attributes
    ----------
    model : PTAModel
    samples : dict
    catalog : dict
        Flat parameter catalog mapping ``'<key>_<param>'`` or
        ``'<key>_<psrname>_<param>'`` strings to dicts with keys
        ``'label'``, ``'samples'``, and book-keeping metadata.
    psr_names : list of str
    npsrs : int

    Examples
    --------
    Minimal usage::

        plotter = Iris(model, samples)

        # Corner plot of all GWB and noise parameters
        fig = plotter.corner()

        # Trace plots for CW source parameters only
        fig = plotter.trace(params=plotter.params_for_key('cw_source'))

        # Sky map with pulsars and CW samples
        fig = plotter.skymap()
    """

    def __init__(
        self,
        model,
        samples: dict,
        param_names: dict | None = None,
        truths: dict | None = None,
    ):
        self.model = model
        self.samples = {k: np.array(v) for k, v in samples.items()}
        self.psr_names = list(model.data.psr_names)
        self.npsrs = model.data.npsrs
        self._param_names = param_names
        self.catalog = build_param_catalog(model, self.samples, param_names)

        # flat truths catalog — populated when truths are supplied
        self.truths_catalog: dict | None = None
        if truths is not None:
            self.truths_catalog = build_truths_catalog(model, truths, param_names)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def params_for_key(self, key: str) -> list[str]:
        """
        Return all flat catalog keys that belong to sample-site *key*.

        Useful for selecting a subset of parameters to pass to
        ``corner()`` or ``trace()``.
        """
        return [k for k, v in self.catalog.items() if v.get("key") == key]

    def params_for_pulsar(self, psr_name: str) -> list[str]:
        """Return all flat catalog keys associated with pulsar *psr_name*."""
        return [k for k, v in self.catalog.items() if v.get("psr_name") == psr_name]

    def params_for_pulsars(self, indices: list[int]) -> list[str]:
        """
        Return flat catalog keys for a subset of pulsars selected by index.

        Parameters
        ----------
        indices : list of int
            Zero-based indices into ``model.data.psr_names``.

        Example
        -------
        >>> plotter.params_for_pulsars([0, 2])   # first and third pulsar
        """
        selected_names = {self.psr_names[i] for i in indices}
        return [k for k, v in self.catalog.items() if v.get("psr_name") in selected_names]

    def psr_model_params(
        self,
        psr_indices: list[int] | None = None,
        psr_index: int | None = None,
    ) -> list[str]:
        """
        Return flat catalog keys for the pulsar noise model (``model.psr_model``).

        Parameters
        ----------
        psr_indices : list of int, optional
            Restrict to these pulsar indices.  If None, returns params for all
            pulsars.
        psr_index : int, optional
            Convenience singular form — equivalent to ``psr_indices=[psr_index]``.
            Ignored when *psr_indices* is also given.
        """
        if psr_indices is None and psr_index is not None:
            psr_indices = [psr_index]
        if not hasattr(self.model, "psr_model") or self.model.psr_model is None:
            return []
        key = self.model.psr_model.name
        keys = self.params_for_key(key)
        if psr_indices is not None:
            selected_names = {self.psr_names[i] for i in psr_indices}
            keys = [k for k in keys if self.catalog[k].get("psr_name") in selected_names]
        # Group by pulsar so params for each pulsar appear together
        psr_order = {name: i for i, name in enumerate(self.psr_names)}
        keys.sort(key=lambda k: psr_order.get(self.catalog[k].get("psr_name"), 0))
        return keys

    def _per_psr_key_params(
        self,
        key: str | tuple[str, ...],
        psr_indices: list[int] | None = None,
        psr_index: int | None = None,
    ) -> list[str]:
        """
        Return flat catalog keys for per-pulsar sample-site *key*, ordered by
        pulsar.  *key* may be a tuple of aliases (e.g. singular and plural
        spellings); the first alias present in the samples is used.
        """
        aliases = (key,) if isinstance(key, str) else key
        key = next((a for a in aliases if a in self.samples), aliases[0])
        if psr_indices is None and psr_index is not None:
            psr_indices = [psr_index]
        if psr_indices is None:
            psr_indices = list(range(self.npsrs))
        keys = []
        for i in psr_indices:
            psrname = self.psr_names[i]
            keys.extend(
                k for k, v in self.catalog.items()
                if v.get("key") == key and v.get("psr_name") == psrname
            )
        return keys

    def psr_phase_params(
        self,
        psr_indices: list[int] | None = None,
        psr_index: int | None = None,
    ) -> list[str]:
        """
        Return flat catalog keys for the per-pulsar CW phase parameters
        (sample key ``'psr_phases'``, shape ``(nsamples, npsrs)``).

        Parameters
        ----------
        psr_indices : list of int, optional
            Restrict to these pulsar indices.  If None, returns params for
            all pulsars.
        psr_index : int, optional
            Convenience singular form — equivalent to
            ``psr_indices=[psr_index]``.  Ignored when *psr_indices* is given.

        Example
        -------
        >>> plotter.corner(plotter.psr_phase_params([0, 3]))
        """
        return self._per_psr_key_params(("psr_phases", "psr_phase"), psr_indices, psr_index)

    def psr_dist_params(
        self,
        psr_indices: list[int] | None = None,
        psr_index: int | None = None,
    ) -> list[str]:
        """
        Return flat catalog keys for the per-pulsar distance parameters
        (sample key ``'psr_dists'``, shape ``(nsamples, npsrs)``).

        Parameters
        ----------
        psr_indices : list of int, optional
            Restrict to these pulsar indices.  If None, returns params for
            all pulsars.
        psr_index : int, optional
            Convenience singular form — equivalent to
            ``psr_indices=[psr_index]``.  Ignored when *psr_indices* is given.

        Example
        -------
        >>> plotter.corner(plotter.psr_dist_params([0, 3]))
        """
        return self._per_psr_key_params(("psr_dists", "psr_dist"), psr_indices, psr_index)

    def psr_dist_phase_params(
        self,
        psr_indices: list[int] | None = None,
        psr_index: int | None = None,
    ) -> list[str]:
        """
        Return flat catalog keys interleaving distance and phase per pulsar:
        ``(dist1, phase1, dist2, phase2, ...)`` for the selected pulsars.

        Parameters
        ----------
        psr_indices : list of int, optional
            Restrict to these pulsar indices.  If None, returns params for
            all pulsars.
        psr_index : int, optional
            Convenience singular form — equivalent to
            ``psr_indices=[psr_index]``.  Ignored when *psr_indices* is given.

        Example
        -------
        >>> plotter.corner(plotter.psr_dist_phase_params(psr_indices=[0, 3]))
        """
        if psr_indices is None and psr_index is not None:
            psr_indices = [psr_index]
        if psr_indices is None:
            psr_indices = list(range(self.npsrs))
        keys = []
        for i in psr_indices:
            keys.extend(self._per_psr_key_params(("psr_dists", "psr_dist"), [i]))
            keys.extend(self._per_psr_key_params(("psr_phases", "psr_phase"), [i]))
        return keys

    def gwb_model_params(self) -> list[str]:
        """Return flat catalog keys for the GWB model (``model.gwb_model``)."""
        if not hasattr(self.model, "gwb_model") or self.model.gwb_model is None:
            return []
        return self.params_for_key(self.model.gwb_model.name)

    def det_model_params(self, param_indices: list[int] | None = None) -> list[str]:
        """
        Return flat catalog keys for the deterministic model (``model.det_model``).

        Parameters
        ----------
        param_indices : list of int, optional
            Restrict to specific parameter indices (e.g. ``[0, 1, 5, 6]`` for
            chirp mass, frequency, and sky location in an 8-param CW model).
            If None, returns all deterministic model parameters.
        """
        if not hasattr(self.model, "det_model") or self.model.det_model is None:
            return []
        key = self.model.det_model.name
        keys = self.params_for_key(key)
        if param_indices is not None:
            keys = [k for k in keys
                    if self.catalog[k].get("param_idx") in param_indices]
        return keys

    def list_params(self) -> list[str]:
        """Return all flat parameter names in the catalog."""
        return list(self.catalog.keys())

    def print_params(self) -> None:
        """Print a formatted table of available parameters and their labels."""
        print(f"{'Flat key':<55}  Label")
        print("-" * 80)
        for k, v in self.catalog.items():
            label_clean = v["label"].replace("\n", " / ")
            print(f"{k:<55}  {label_clean}")

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------

    def corner(
        self,
        params: list[str] | None = None,
        thin: int = 1,
        truths: dict | None = None,
        fig: plt.Figure | None = None,
        extra_samples: list[dict] | None = None,
        labels: list[str] | None = None,
        colors: list[str] | None = None,
        **corner_kwargs,
    ) -> plt.Figure:
        """
        Make a corner plot, optionally overlaying samples from other runs.

        Parameters
        ----------
        params : list of str, optional
            Flat catalog keys to include.  If None, all non-Fourier
            parameters are included automatically.
        thin : int
            Thinning factor applied to all datasets.
        truths : dict, optional
            ``{flat_key: truth_value}`` drawn as dashed lines (uses the
            truths supplied at construction if not given here).
        fig : matplotlib Figure, optional
            Existing figure to draw into.
        extra_samples : list of dict, optional
            Additional sample dictionaries from other runs, each in the
            same raw format as the ``samples`` dict passed to ``Iris``
            (i.e. ``{sample_key: array(nsamples, ...)}``) — they do **not**
            need to have the same keys or shapes as the primary samples.
            Each is built into a temporary catalog using the same model and
            ``param_names`` as the primary ``Iris`` instance.  Datasets
            where any of the selected parameters are absent are skipped with
            a warning.
        labels : list of str, optional
            Legend labels, one per dataset — primary first, then each entry
            in ``extra_samples``.  For example::

                labels=['NUTS', 'HMC', 'prior']

            If None and there is more than one dataset, labels default to
            "Run 1", "Run 2", …
        colors : list[str], optional
            Matplotlib color strings, one per dataset.  Auto-assigned if None.
        legend_fontsize : int or float, optional
            Font size for the legend in the top-right corner.  Default 10.
        **corner_kwargs
            Additional keyword arguments forwarded to ``corner.corner``.

        Returns
        -------
        fig : matplotlib Figure
        """
        # build extra catalogs from raw sample dicts using the same model
        extra_catalogs = None
        if extra_samples:
            from .utils import build_param_catalog
            extra_catalogs = [
                build_param_catalog(self.model, s, self._param_names)
                for s in extra_samples
            ]

        legend_fontsize = corner_kwargs.pop("legend_fontsize", 10)
        return make_corner(
            self.catalog,
            params=params,
            thin=thin,
            truths=truths if truths is not None else self.truths_catalog,
            fig=fig,
            extra_catalogs=extra_catalogs,
            labels=labels,
            colors=colors,
            legend_fontsize=legend_fontsize,
            corner_kwargs=corner_kwargs or None,
        )

    def trace(
        self,
        params: list[str] | None = None,
        thin: int = 1,
        truths: dict | None = None,
        figsize: tuple | None = None,
        ncols: int = 2,
        **plot_kwargs,
    ) -> plt.Figure:
        """
        Make trace plots (sample index vs parameter value).

        Parameters
        ----------
        params : list of str, optional
            Flat catalog keys to plot.  If None, all non-Fourier parameters
            are plotted.
        thin : int
            Thinning factor.
        truths : dict, optional
            ``{flat_key: truth_value}`` drawn as horizontal dashed lines.
        figsize : tuple, optional
            Figure size.  Auto-sized if None.
        ncols : int
            Number of panel columns.
        **plot_kwargs
            Extra keyword arguments forwarded to ``ax.plot``.

        Returns
        -------
        fig : matplotlib Figure
        """
        return make_trace(
            self.catalog,
            params=params,
            thin=thin,
            truths=truths if truths is not None else self.truths_catalog,
            figsize=figsize,
            ncols=ncols,
            plot_kwargs=plot_kwargs or None,
        )

    def violin(
        self,
        params: list[str],
        extra_samples: list[dict] | dict | None = None,
        labels: list[str] | None = None,
        colors: list[str] | None = None,
        figsize: tuple = (10, 5),
        title: str | None = None,
        xlabel: str = "frequency bin",
        ylabel: str | None = None,
        legend_fontsize: int | float = 12,
        widths: float = 0.8,
        fig: plt.Figure | None = None,
        ax=None,
    ) -> plt.Figure:
        """
        Make a violin plot for a free-spectral GWB or pulsar-noise model.

        Pass the list of flat catalog keys returned by
        ``plotter.gwb_model_params()`` or
        ``plotter.psr_model_params(psr_index=5)`` as *params* — the same
        format used by ``corner()``.  Each key corresponds to one frequency
        bin; its posterior samples become one violin.

        Parameters
        ----------
        params : list of str
            Flat catalog keys to plot, one per frequency bin.  Obtain them
            with ``plotter.gwb_model_params()`` or
            ``plotter.psr_model_params(psr_index=N)``.  Only free-spectral
            params (more than 2 bins) are accepted; power-law params raise a
            ``ValueError``.
        extra_samples : list of dict or dict, optional
            Additional raw sample dicts (same format as the ``samples`` dict
            passed to ``Iris``).  A single dict may be passed without wrapping
            it in a list.
        labels : list of str, optional
            Legend labels, one per dataset — primary first, then each entry in
            ``extra_samples``.  Defaults to "Run 1", "Run 2", … when more than
            one dataset is present.
        colors : list of str, optional
            Matplotlib colour strings, one per dataset.  Auto-assigned if None.
        figsize : tuple, optional
            ``(width, height)`` in inches.  Default ``(10, 5)``.
        title : str, optional
            Figure title.
        xlabel : str
            x-axis label.  Default ``'frequency bin'``.
        ylabel : str, optional
            y-axis label.  Auto-inferred from the param metadata
            (``$\\log_{{10}}\\rho_i\\;\\;[\\mathrm{{GWB}}]$`` or the pulsar
            name) when not supplied.
        legend_fontsize : int or float
            Font size of the legend.  Default 12.
        widths : float
            Width of each violin body.  Default 0.8.
        fig : matplotlib Figure, optional
            Existing figure to draw into.
        ax : matplotlib Axes, optional
            Existing axes to draw into.

        Returns
        -------
        fig : matplotlib Figure
        """
        if not params:
            raise ValueError("params must be a non-empty list of flat catalog keys.")
        if len(params) <= 2:
            raise ValueError(
                f"Only {len(params)} param(s) selected — this looks like a "
                "power-law model. Violin plots require free-spectral params "
                "(more than 2 frequency bins)."
            )

        # validate all keys exist in the primary catalog
        missing = [k for k in params if k not in self.catalog]
        if missing:
            raise KeyError(f"Param key(s) not found in catalog: {missing}")

        # --- infer default ylabel from catalog metadata ---
        if ylabel is None:
            meta = self.catalog[params[0]]
            psr_name = meta.get("psr_name")
            if psr_name:
                ylabel = rf"$\log_{{10}}|\mathrm{{PSD}}\;\;[\mathrm{{s}}^2]|\;\;[{psr_name}]$"
            else:
                ylabel = r"$\log_{10}|\mathrm{PSD}\;\;[\mathrm{s}^2]|\;\;[\mathrm{GWB}]$"

        # --- build primary dataset from catalog ---
        primary_data = np.column_stack(
            [self.catalog[k]["samples"] for k in params]
        )  # (nsamples, nfreqs)

        # --- build extra datasets from raw sample dicts ---
        if isinstance(extra_samples, dict):
            extra_samples = [extra_samples]
        extra_samples = extra_samples or []

        datasets = [primary_data]
        for raw in extra_samples:
            tmp_catalog = build_param_catalog(self.model, raw, self._param_names)
            extra_missing = [k for k in params if k not in tmp_catalog]
            if extra_missing:
                import warnings
                warnings.warn(
                    f"Extra sample set is missing param(s) {extra_missing} — skipping.",
                    stacklevel=2,
                )
                continue
            datasets.append(
                np.column_stack([tmp_catalog[k]["samples"] for k in params])
            )

        return make_violin(
            datasets=datasets,
            labels=labels,
            colors=colors,
            figsize=figsize,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend_fontsize=legend_fontsize,
            widths=widths,
            fig=fig,
            ax=ax,
        )

    def skymap(
        self,
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
        Make a Mollweide sky map.

        Pulsar locations are read directly from ``model.data.psr_phi`` and
        ``model.data.psr_theta``.  CW sky-location samples are extracted
        automatically from the deterministic model samples when present.

        Parameters
        ----------
        show_pulsars : bool
            Plot pulsar sky positions (gold stars, labelled by name).
        show_cw_samples : bool
            Scatter-plot CW sky-location posterior samples.
        show_cw_contours : bool
            Overlay 2-D density contours on CW sky-location samples.
        cw_sample_key : str, optional
            Sample-site key for the CW parameter array.  Auto-detected from
            ``model.det_model.name`` if None.
        sky_params : tuple of (str, str), optional
            ``(cos_theta_key, phi_key)`` — a pair of flat catalog keys for
            any additional sky-location parameter pair to overplot.
        thin : int
            Thinning factor.
        ncontour_bins : int
            Bins per axis for density contour estimation.
        psr_kwargs : dict, optional
            Keyword arguments for pulsar scatter.
        cw_kwargs : dict, optional
            Keyword arguments for CW sample scatter.
        figsize : tuple
            Figure size.
        title : str, optional
            Figure suptitle.
        truths : dict, optional
            ``{'cos_gwtheta': val, 'gwphi': val}`` to mark true position.

        Returns
        -------
        fig : matplotlib Figure
        """
        return make_skymap(
            model=self.model,
            catalog=self.catalog,
            samples=self.samples,
            show_pulsars=show_pulsars,
            show_cw_samples=show_cw_samples,
            show_cw_contours=show_cw_contours,
            cw_sample_key=cw_sample_key,
            sky_params=sky_params,
            thin=thin,
            ncontour_bins=ncontour_bins,
            psr_kwargs=psr_kwargs,
            cw_kwargs=cw_kwargs,
            figsize=figsize,
            title=title,
            truths=truths if truths is not None else self.truths_catalog,
        )
