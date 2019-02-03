from MagniPy.ABCsampler.Chain import ChainFromSamples
import numpy as np
import sys

def evaluate_denities(chain_name, which_lenses, error, n_pert, nkde_bins,
                      file_ending, tol, bw_scale):

    chain = ChainFromSamples(chain_name, which_lens=which_lenses,
                                   error=error, n_pert=n_pert)

    chain.eval_KDE(bandwidth_scale=bw_scale,tol = tol, nkde_bins=nkde_bins,
                   save_to_file=True)

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

    bandwidth_scale = 0.6
    nkde_bins = 15
    tol = 1500
    which_lenses = np.arange(1,51)
    evaluate_denities('CDM_sigma0_srcsize35', which_lenses,
                      error, n_pert, nkde_bins, '', tol, bandwidth_scale)

run()