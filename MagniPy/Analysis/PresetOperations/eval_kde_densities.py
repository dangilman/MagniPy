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

def run(savename, chain_name, nlens, error, global_weights = None, single_weights = None):
    #errors = [2,4,6]
    #error = errors[int(sys.argv[1])-1]

    if error == 0:
        n_pert = 1
    elif error == 1:
        n_pert = 5
    elif error == 2:
        n_pert = 5
    elif error == 4:
        n_pert = 5
    elif error == 6:
        n_pert = 5

    bandwidth_scale = 1
    nkde_bins = 10
    tol = 450
    which_lenses = np.arange(1, 36)
    evaluate_denities(chain_name, which_lenses,
                      error, n_pert, nkde_bins, tol, bandwidth_scale, savename,
                      global_weights = global_weights, single_weights = single_weights)

N = 37

#run('noweights_error0_', 'SIDM_sigma0.025_cross0.1', N, 0)
#run('LOSsrcweights_error0_', 'SIDM_sigma0.025_cross0.1', N, 0, global_weights={'LOS_normalization':[1, 0.05]},
#    single_weights={'source_size_kpc': [0.018, 0.005]})
#exit(1)

for ei in [0]:
    run('noweights_error'+str(ei)+'_', 'SIDM_sigma0.022_cross0.1', N, ei)
    run('LOSsrcweights_error'+str(ei)+'_', 'SIDM_sigma0.022_cross0.1', N, ei, global_weights={'LOS_normalization':[1, 0.05]},
        single_weights={'source_size_kpc': [0.023, 0.005]})

    #run('noweights_error'+str(ei)+'_', 'SIDM_sigma0.015_cross0.25', N, ei)
    #run('LOSsrcweights_error'+str(ei)+'_', 'SIDM_sigma0.015_cross0.25', N, ei, global_weights={'LOS_normalization':[1, 0.05], 'source_size_kpc': [0.02, 0.005]})

    #run('noweights_error'+str(ei)+'_', 'SIDM_sigma0.025_cross0.25', N, ei)
    #run('LOSsrcweights_error'+str(ei)+'_', 'SIDM_sigma0.025_cross0.25', N, ei, global_weights={'LOS_normalization':[1, 0.05], 'source_size_kpc': [0.02, 0.005]})

    #run('noweights_error'+str(ei)+'_', 'SIDM_sigma0.015_cross9', N, ei)
    #run('LOSsrcweights_error'+str(ei)+'_', 'SIDM_sigma0.015_cross9', N, ei, global_weights={'LOS_normalization':[1, 0.05]},
    #    single_weights={'source_size_kpc': [0.02, 0.005]})
