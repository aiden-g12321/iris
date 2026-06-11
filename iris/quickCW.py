'''Convert samples from QuickCW to Prometheus format for comparisons.'''


import h5py
import numpy as np


def convert_QuickCW_samples(filepath, pta_model, prune_domain=True):
    '''
    Convert samples from QuickCW to dictionary in Prometheus format.

    Parameters
    ----------
    filepath : str
        Path to h5 file where QuickCW results are saved.

    pta_model : Prometheus.PTAModel
        A Prometheus.PTAModel instance.
    
    prune_domain : bool
        Whether or not to remove samples which are outside
        bounds of Prometheus CW prior. Defaults to True.

    Returns
    -------
    quickCW_samples_dict : dict
        A dictionary of QuickCW samples in the Prometheus format.
    '''
    
    quickCW_samples_dict = {}
    psr_names = pta_model.data.psr_names

    # load QuickCW samples
    with h5py.File(filepath, 'r') as f:
        par_names = [n.decode() for n in f['par_names'][:]]
        samples = f['samples_cold'][0, :, :]  # cold chain, shape (N, n_par)

    # continuous wave parameters, shape (N, 8)
    cw_par_order = ['log10_mc', 'log10_fgw', 'cos_inc', 'psi',
                    'log10_h', 'cos_gwtheta', 'gwphi', 'phase0']
    cw_params = np.array([samples[:, par_names.index('0_' + p)] for p in cw_par_order]).T
    mask = np.r_[0 : cw_params.shape[0]]
    if pta_model.det_model is not None:
        mask = np.all((cw_params >= pta_model.det_model.param_mins) & 
                  (cw_params <= pta_model.det_model.param_maxs), axis=1)
        quickCW_samples_dict[pta_model.det_model.name] = cw_params[mask]

    # pulsar red noise parameters, ordered as psr_names
    # powerlaw:      psr_noise_params has shape (N, Np, 2), ordered [log10_A, gamma]
    # free spectral: psr_noise_params has shape (N, Np, Nf), ordered [log10_rho_0, ..., log10_rho_{Nf-1}]
    psr_noise_powerlaw = any('_red_noise_gamma' in n for n in par_names)

    if psr_noise_powerlaw:
        psr_noise_params = np.stack(
            [np.array([samples[:, par_names.index(f'{p}_red_noise_log10_A')],
                    samples[:, par_names.index(f'{p}_red_noise_gamma')]]).T
            for p in psr_names], axis=1)  # (N, Np, 2)
    else:
        n_rn = sum(1 for n in par_names if n.startswith(psr_names[0] + '_red_noise_log10_rho'))
        psr_noise_params = np.stack(
            [np.array([samples[:, par_names.index(f'{p}_red_noise_log10_rho_{k}')]
                    for k in range(n_rn)]).T
            for p in psr_names], axis=1)  # (N, Np, Nf)

    quickCW_samples_dict[pta_model.psr_model.name] = psr_noise_params[mask]

    # GWB parameters: powerlaw -> shape (N, 2) ordered [log10_A, gamma],
    #                 free spectral -> shape (N, Nf) ordered [log10_rho_0, ..., log10_rho_{Nf-1}]
    if 'gwb_log10_A' in par_names:
        gwb_params = np.array([samples[:, par_names.index('gwb_log10_A')],
                            samples[:, par_names.index('gwb_gamma')]]).T
    else:
        n_gwb = sum(1 for n in par_names if n.startswith('gwb_log10_rho'))
        gwb_params = np.array([samples[:, par_names.index(f'gwb_log10_rho_{k}')]
                            for k in range(n_gwb)]).T

    quickCW_samples_dict[pta_model.gwb_model.name] = gwb_params[mask]

    # pulsar term phases and distances, each shape (N, Np), ordered as psr_names
    psr_phases = np.array([samples[:, par_names.index(f'{p}_cw0_p_phase')]
                        for p in psr_names]).T

    psr_dists = np.array([samples[:, par_names.index(f'{p}_cw0_p_dist')]
                        for p in psr_names]).T

    quickCW_samples_dict['psr_phases'] = psr_phases[mask]
    quickCW_samples_dict['psr_dists'] = psr_dists[mask]
    
    return quickCW_samples_dict


