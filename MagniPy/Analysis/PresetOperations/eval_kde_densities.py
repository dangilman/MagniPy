from MagniPy.ABCsampler.Chain import ChainFromSamples
import numpy as np
import sys

def evaluate_denities(chain_name, which_lenses, error, n_pert, nkde_bins,
                      tol, bw_scale, fname, global_weights = None, single_weights = None, compute_inds=None):

    chain = ChainFromSamples(chain_name, which_lens=which_lenses,
                                   error=error, n_pert=n_pert)

    chain.eval_KDE(bandwidth_scale=bw_scale,tol = tol, nkde_bins=nkde_bins,
                   save_to_file=True, smooth_KDE=True, filename = fname,
                   weights_global=global_weights, weights_single=single_weights, compute_inds=compute_inds)

def run(savename, chain_name, nlens, error, compute_index_min, compute_index_max, global_weights = None, single_weights = None):
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

    bandwidth_scale = 0.8
    nkde_bins = 12
    tol = 0.02
    compute_inds = np.arange(compute_index_min, compute_index_max)
    which_lenses = np.arange(1, nlens)
    evaluate_denities(chain_name, which_lenses,
                      error, n_pert, nkde_bins, tol, bandwidth_scale, savename,
                      global_weights = global_weights, single_weights = single_weights, compute_inds=compute_inds)


Nlens = 51

run_1 = False
run_2 = True
err = 2
index_min = int(sys.argv[1])
index_step = int(sys.argv[2])

run('unweighted_error'+str(err)+'_', 'SIDM_sigma0.03_cross9', Nlens, err, index_min, index_min + index_step)

    #run('noweights_error'+str(ei)+'_', 'SIDM_sigma0.015_cross0.25', N, ei)
    #run('LOSsrcweights_error'+str(ei)+'_', 'SIDM_sigma0.015_cross0.25', N, ei, global_weights={'LOS_normalization':[1, 0.05], 'source_size_kpc': [0.02, 0.005]})

    #run('noweights_error'+str(ei)+'_', 'SIDM_sigma0.025_cross0.25', N, ei)
    #run('LOSsrcweights_error'+str(ei)+'_', 'SIDM_sigma0.025_cross0.25', N, ei, global_weights={'LOS_normalization':[1, 0.05], 'source_size_kpc': [0.02, 0.005]})

    #run('noweights_error'+str(ei)+'_', 'SIDM_sigma0.015_cross9', N, ei)
    #run('LOSsrcweights_error'+str(ei)+'_', 'SIDM_sigma0.015_cross9', N, ei, global_weights={'LOS_normalization':[1, 0.05]},
    #    single_weights={'source_size_kpc': [0.02, 0.005]})
