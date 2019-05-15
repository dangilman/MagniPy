from MagniPy.ABCsampler.Chain import ChainFromSamples
import numpy as np
import sys

def weight_a0area(values, mean = 0.0, sigma = .02):

    return np.exp(-0.5*(values - mean)**2 * sigma ** -2)

def weight_cross(values, mean = 0.01, sigma = 5):

    return np.exp(-0.5*(values - mean)**2 * sigma ** -2)

def evaluate_denities(chain_name, which_lenses, error, n_pert, nkde_bins,
                      file_ending, tol, bw_scale, weight_funcs=None):

    chain = ChainFromSamples(chain_name, which_lens=which_lenses,
                                   error=error, n_pert=n_pert)

    chain.eval_KDE(bandwidth_scale=bw_scale,tol = tol, nkde_bins=nkde_bins,
                   save_to_file=True, smooth_KDE=True, weights=weight_funcs)

def run():
    #errors = [2,4,6]
    #error = errors[int(sys.argv[1])-1]
    error = 0
    if error == 0:
        n_pert = 1
    if error == 2:
        n_pert = 5
    elif error == 4:
        n_pert = 5
    elif error == 6:
        n_pert = 5

    weight_functions = {'a0_area': weight_a0area}
    #weight_functions = None
    bandwidth_scale = 0.6
    nkde_bins = 10
    tol = 500
    which_lenses = np.arange(1,26)
    evaluate_denities('SIDM_sigma0.0_cross0.01', which_lenses,
                      error, n_pert, nkde_bins, '', tol, bandwidth_scale, weight_funcs=weight_functions)

run()