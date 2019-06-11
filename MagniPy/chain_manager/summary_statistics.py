import numpy as np

def sample_flux_perturbations(delta_f, fluxes):

    pert = [np.random.normal(0, pert_i) for pert_i in delta_f]

    for i, pi in enumerate(pert):
        fluxes[:,i] += pert
    return fluxes

def quadrature(fluxes_obs, fluxes):

    diff = (fluxes_obs - fluxes)**2

    return np.sqrt(np.sum(diff, axis=1))