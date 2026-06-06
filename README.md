# Nyota

**Nyota is a plotting package designed exclusively for [Prometheus](https://github.com/XGI-MSU/prometheus) PTA analysis outputs.** It is not a general-purpose plotting library — it is built around Prometheus's `PTAModel` object and the MCMC sample dictionaries that Prometheus produces. Nyota reads your model directly to figure out pulsar names, sky locations, parameter shapes, and constituent sub-models (`psr_model`, `gwb_model`, `det_model`), so you spend your time looking at results rather than wrangling labels and array indices.

> **Requirement:** Nyota expects a Prometheus `PTAModel` instance and a sample dictionary as returned by NumPyro / the Prometheus utility functions. It will not work meaningfully without them.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/aiden-g12321/nyota.git
```

To install an editable local copy (e.g. while developing alongside Prometheus):

```bash
git clone https://github.com/aiden-g12321/nyota.git
cd nyota
pip install -e .
```

**Dependencies:** `numpy`, `matplotlib`, `corner`, `scipy`

## Quick start

```python
from nyota import Nyota

# model   — a Prometheus PTAModel
# samples — dict returned by NumPyro / prometheus.utilities.load_chain()

plotter = Nyota(model, samples)

# Corner plot of all inferred parameters (auto-labelled from the PTAModel)
fig = plotter.corner()

# Trace plots
fig = plotter.trace()

# Mollweide sky map: pulsar positions + CW sky-location posterior
fig = plotter.skymap()
```

## Selecting parameters by model

Because Nyota has access to the `PTAModel`, you never need to type pulsar names or remember parameter orderings. Use the model-level helpers to select parameters by which sub-model they belong to:

```python
# All GWB parameters (reads model.gwb_model.name automatically)
plotter.gwb_model_params()

# Pulsar noise parameters for pulsars 0 and 2 (by index)
plotter.psr_model_params(psr_indices=[0, 2])

# CW source parameters by index in the 8-parameter ordering
# e.g. [0,1,5,6] → log10_mc, log10_fgw, cos_gwtheta, gwphi
plotter.det_model_params(param_indices=[0, 1, 5, 6])
```

Mix and match across models to build the exact corner plot you want:

```python
params = (
    plotter.gwb_model_params()                          # log10_A, gamma
    + plotter.psr_model_params(psr_indices=[0, 1])      # RN for pulsars 0 & 1
    + plotter.det_model_params(param_indices=[0,1,5,6]) # 4 CW params
)

fig = plotter.corner(params=params)
```

Print the full parameter table to see every available flat key and its label:

```python
plotter.print_params()
```

## Corner plots

```python
# All non-Fourier parameters
fig = plotter.corner()

# Selected parameter subset
fig = plotter.corner(params=params)

# Thinning to speed up rendering
fig = plotter.corner(params=params, thin=5)

# Overlay samples from multiple runs with a legend
fig = plotter.corner(
    params=params,
    extra_samples=[samples_run2, samples_run3],
    labels=['NUTS run 1', 'NUTS run 2', 'Prior'],
    colors=['#2196F3', '#F44336', '#4CAF50'],   # optional; auto-assigned if omitted
)
```

`extra_samples` takes raw sample dicts in the same format as the primary `samples` — they do not need to have identical keys or shapes. Parameters missing from an extra run are skipped for that run with a warning; all other runs still render.

## Injected / truth values

Supply a `truths` dict at construction time using the same key-value format as the samples dictionary. Nyota converts it automatically to the flat catalog format and uses it as the default for all subsequent plots:

```python
truths = {
    'gwb':       np.array([-14.5, 13/3]),
    'psr_noise': np.array([[-14.0, 4.0],      # shape (npsrs, nparams)
                            [-14.2, 4.2],
                            [-13.8, 3.8]]),
    'cw_source': np.array([9.0, -8.5, 0.0, 0.5, -14.8, 0.3, 1.2, 0.0]),
}

plotter = Nyota(model, samples, truths=truths)

# Truth lines are drawn automatically on all plots
fig = plotter.corner(params=params)
fig = plotter.trace(params=params)
fig = plotter.skymap()
```

## Trace plots

```python
# All non-Fourier parameters
fig = plotter.trace()

# CW source parameters only
fig = plotter.trace(params=plotter.det_model_params())

# Custom grid layout
fig = plotter.trace(params=params, ncols=3, figsize=(15, 10))
```

## Sky maps

```python
# Pulsars (labelled stars) + CW sky-location posterior (scatter + contours)
# Pulsar positions and CW sample key are read from the PTAModel automatically
fig = plotter.skymap()

# Pulsars only
fig = plotter.skymap(show_cw_samples=False, show_cw_contours=False)

# Arbitrary (cos_theta, phi) parameter pair from the catalog
fig = plotter.skymap(sky_params=('my_model_cos_gwtheta', 'my_model_gwphi'))
```

## Custom parameter names

Nyota infers parameter names from array shape (8-param → standard CW names, 2-param → power-law names). Override with `param_names`:

```python
plotter = Nyota(model, samples, param_names={
    'psr_noise': ['log10_A', 'gamma'],
    'cw_source': ['log10_mc', 'log10_fgw', 'cos_inc', 'psi',
                  'log10_h', 'cos_gwtheta', 'gwphi', 'phase0'],
})
```

## CW parameter conventions

The 8-parameter CW model follows the ordering used by Prometheus (`prometheus/deterministic.py`):

| Index | Name | Description |
|---|---|---|
| 0 | `log10_mc` | $\log_{10}$ chirp mass $[\mathcal{M}_\odot]$ |
| 1 | `log10_fgw` | $\log_{10}$ GW frequency $[\mathrm{Hz}]$ |
| 2 | `cos_inc` | cosine of inclination angle |
| 3 | `psi` | polarization angle $[\mathrm{rad}]$ |
| 4 | `log10_h` | $\log_{10}$ characteristic strain |
| 5 | `cos_gwtheta` | cosine of polar sky angle |
| 6 | `gwphi` | azimuthal sky angle $[\mathrm{rad}]$ |
| 7 | `phase0` | initial GW phase $[\mathrm{rad}]$ |

## License

MIT
