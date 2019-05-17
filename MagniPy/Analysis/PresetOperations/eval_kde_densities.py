from MagniPy.ABCsampler.Chain import ChainFromSamples
import numpy as np
import sys

def weight_a0area(values, mean = 0.015, sigma = .025):

    return np.exp(-0.5*(values - mean)**2 * sigma ** -2)

def weight_sourcesize(values, mean = 0.02, sigma = 0.005):

    return np.exp(-0.5*(values - mean)**2 * sigma ** -2)

def weight_siegamma(values, mean = 2.08, sigma = 0.04):

    return np.exp(-0.5*(values - mean)**2 * sigma ** -2)

def weight_LOSnorm(values, mean = 1, sigma = 0.1):

    return np.exp(-0.5*(values - mean)**2 * sigma ** -2)

def evaluate_denities(chain_name, which_lenses, error, n_pert, nkde_bins,
                      file_ending, tol, bw_scale, weight_funcs=None):

    chain = ChainFromSamples(chain_name, which_lens=which_lenses,
                                   error=error, n_pert=n_pert)

    chain.eval_KDE(bandwidth_scale=bw_scale,tol = tol, nkde_bins=nkde_bins,
                   save_to_file=True, smooth_KDE=True, weights=weight_funcs)

def run(weight_functions = None):
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

    bandwidth_scale = 0.6
    nkde_bins = 10
    tol = 800
    which_lenses = np.arange(1,21)
    evaluate_denities('SIDM_sigma0.012_cross6.5', which_lenses,
                      error, n_pert, nkde_bins, '', tol, bandwidth_scale, weight_funcs=weight_functions)

#wfunc1 = {'SIE_gamma': weight_siegamma, 'source_size_kpc': weight_sourcesize, 'LOS_normalization': weight_LOSnorm}
#run()
run()