from MagniPy.ABCsampler.Chain import ChainFromSamples
import numpy as np
import sys

def evaluate_denities(chain_name, which_lenses, error, n_pert, nkde_bins,
                      tol, bw_scale, fname, global_weights = None, single_weights = None):

    chain = ChainFromSamples(chain_name, which_lens=which_lenses,
                                   error=error, n_pert=n_pert)

    chain.eval_KDE(bandwidth_scale=bw_scale,tol = tol, nkde_bins=nkde_bins,
                   save_to_file=True, smooth_KDE=True, filename = fname,
                   weights_global=global_weights, weights_single=single_weights)

def run(savename, global_weights = None, single_weights = None):
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

    bandwidth_scale = 0.8
    nkde_bins = 12
    tol = 600
    which_lenses = np.arange(1,26)
    evaluate_denities('SIDM_sigma0.015_cross8', which_lenses,
                      error, n_pert, nkde_bins, tol, bandwidth_scale, savename,
                      global_weights = global_weights, single_weights = single_weights)


run('weightsglobal3', global_weights={'LOS_normalization':[1, 0.01], 'SIE_gamma': [2.08, 0.02]})
#run('weightsglobal2', global_weights={'LOS_normalization':[1, 0.001]})
#run('weightssingle1', single_weights=[{'LOS_normalization':[1, 0.01*25]}]*25)