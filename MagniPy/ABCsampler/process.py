from MagniPy.ABCsampler.ChainOps import *
from MagniPy.ABCsampler.Chain import ChainFromChain, ChainFromSamples
from MagniPy.Analysis.Statistics.routines import build_densities, barplothist
from MagniPy.Analysis.Statistics.routines import reweight_posteriors_individually
from MagniPy.Analysis.Statistics.routines import duplicate_with_cuts

def bootstrap_intervals(chain_name, Nlenses, which_lenses, parameter, Nbootstraps, error,
                        tol, param_weights_individual=None, xtrim=None, ytrim=None, bins=40):

    if not isinstance(Nlenses, list):
        Nlenses = [Nlenses]

    chain_master = ChainFromSamples(chain_name, which_lens = which_lenses, error=0, n_pert=1, load=False)
    #params_reject = ['a0_area', 'log_m_break']
    #reject_ranges = [[0, 0.045], [4.8, 10]]

    #chain_master, pranges = duplicate_with_cuts(chain_master, tol, pnames_reject_list=params_reject,
    #                                          keep_ranges_list=reject_ranges)

    low95_interval, low68_interval = [], []
    high95_interval, high68_interval = [], []

    for nlens in Nlenses:

        low95, high95 = [], []
        low68, high68 = [], []

        for i in range(0, Nbootstraps):

            lens_list = np.random.randint(1, len(which_lenses), nlens)
            print(lens_list)
            chain = ChainFromChain(chain_master, lens_list, load_flux=True)

            chain._add_perturbations(error, 2000)

            posteriors = chain.get_posteriors(tol)

            if param_weights_individual is not None:

                weight_param = param_weights_individual['param']
                weight_means = param_weights_individual['means']
                weight_sigmas = param_weights_individual['sigma']

                posteriors = reweight_posteriors_individually(posteriors, weight_param, weight_means, weight_sigmas,
                                                              lens_list, post_to_reweight=[0])[0]

            sims, sim_pranges = build_densities([posteriors], [parameter],
                                                chain.pranges, bandwidth_scale=1,
                                                xtrim=xtrim, ytrim=ytrim, steps=bins,
                                                use_kde_joint=False, use_kde_marginal=True,
                                                reweight=True)

            density = np.ones_like(sims[0][0])

            for di in sims[0]:
                density *= di

            bar_centers, bar_width, bar_heights = barplothist(density,
                              np.linspace(sim_pranges[0][parameter][0],
                                          sim_pranges[0][parameter][1], bins), None)


            h95 = quick_confidence(bar_centers, bar_heights, 0.95)
            l95 = quick_confidence(bar_centers, bar_heights, 0.05)

            low95.append(l95)
            high95.append(h95)

        low95_interval.append(np.mean(low95))
        high95_interval.append(np.mean(high95))

    return {'Nlenses':Nlenses, 'low_95':low95_interval, 'high_95':high95_interval}

def new_chains_withsigma(chain_name, which_lenses, new_chain_name, cut_sigma = 0.05):

    compute_sigma_chains(chain_name, which_lenses, new_chain_name, cut_sigma)

def resample_chain_sigma():

    name = 'WDM_7.7_sigma'
    new_name = 'WDM_7.7_sigma0.3_srcsize35'

    sigma_sub = 0.3
    logmhm = 7.7
    src_size = 0.035
    LOS_norm = 1

    which_lens_indexes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    params_new = {'sigma_sub': [sigma_sub, 0.01], 'log_m_break': [logmhm, 0.1],
                                'source_size_kpc': [src_size, 0.006],
                                'LOS_normalization': [LOS_norm, 0.05]}

    _resample_chain(name = name, new_name = new_name, which_lens_indexes=which_lens_indexes,
                   parameters_new=params_new, SIE_gamma_mean = 2.08,
                       SIE_gamma_sigma = 0.05, transform_fsub = False, sigma_sub = sigma_sub)

def _resample_chain(name=str, new_name = str, which_lens_indexes=int, parameters_new={}, SIE_gamma_mean = 2.08,
                   SIE_gamma_sigma = 0.05, transform_fsub = False, sigma_sub = None):

    new_gamma = []

    if transform_fsub:
        assert sigma_sub is not None

    if not os.path.exists(chainpath_out + 'processed_chains/'+new_name):
        create_directory(chainpath_out + 'processed_chains/'+new_name)

    zd, zs, _ = np.loadtxt(chainpath_out+'/raw_chains/simulation_zRein.txt', unpack = True)

    if transform_fsub:

        from pyHalo.Cosmology.lens_cosmo import LensCosmo
        l = LensCosmo(0.5, 3)

    for which_lens in which_lens_indexes:

        fluxes, fluxes_obs, parameters, header, newgamma = resample(name, which_lens, parameters_new,
                                                                    SIE_gamma_mean=SIE_gamma_mean,
                                                                    SIE_gamma_sigma=SIE_gamma_sigma)

        new_gamma.append(newgamma)

        stack_chain(new_name, which_lens, parameters, fluxes_obs, fluxes, header)

    chain_info_path = chainpath_out + 'processed_chains/'+name + '/simulation_info.txt'

    with open(chain_info_path, 'r') as f:
        lines = f.readlines()

    with open(chainpath_out + 'processed_chains/'+new_name + '/simulation_info.txt', 'w') as f:
        for line in lines:
            f.write(line)
            if line == '# truths\n':
                break

        for pname in parameters_new.keys():

            f.write(pname + ' '+str(parameters_new[pname][0])+'\n')

        f.write('SIE_gamma '+str(SIE_gamma_mean)+'\n')

    with open(chainpath_out + 'processed_chains/'+new_name + '/gamma_values.txt', 'w') as f:

        #f.write('re-sampled gamma '+str(SIE_gamma_mean)+' +\- '+str(SIE_gamma_sigma)+'\n')
        for g in new_gamma:
            f.write(str(g)+'\n')

def process_raw(name=str,which_lenses=[]):

    """
    coverts output from cluster into single files for each lens
    """

    for which_lens in which_lenses:

        fluxes,fluxes_obs,parameters,header = extract_chain(name,which_lens)

        stack_chain(name, which_lens, parameters, np.squeeze(fluxes_obs), fluxes, header)

def process_samples(chain_name, which_lenses, N_pert=1, errors=None):

    for which_lens in which_lenses:

        fluxes, observed_fluxes, parameters, params_header = extract_chain_fromprocessed(chain_name, which_lens)

        add_flux_perturbations(chain_name, which_lens, parameters, observed_fluxes, fluxes, errors = errors, N_pert = N_pert)

def resample_chain(a0_area=None, logmhm=None, src_size=None, LOS_norm=1.0, errors = [0, 0.04],N_pert=1,
                   process_only = False, SIE_gamma_mean=2.08, SIE_gamma_sigma=0.1, logmhm_sigma = 0.015,
                   src_size_sigma = 0.006, a0_area_sigma = 0.001, LOS_norm_sigma = 0.025,
                   name='WDM_sim_7.7_.012', which_lens_indexes = None):

    #name = 'WDM_sim_7.7_.012'

    if logmhm > 6:
        new_name = 'WDM_'+str(logmhm)+'_'
    else:
        new_name = 'CDM_'

    new_name += 'sigma'+str(a0_area)+'_srcsize'+str(src_size)

    if LOS_norm != 1:
        new_name += '_LOS'+str(LOS_norm)

    if process_only is False:
        params_new = {'a0_area': [a0_area, a0_area_sigma], 'log_m_break': [logmhm, logmhm_sigma],
                                    'source_size_kpc': [src_size, src_size_sigma],
                                    'LOS_normalization': [LOS_norm, LOS_norm_sigma]}

        _resample_chain(name = name, new_name = new_name, which_lens_indexes=which_lens_indexes,
                       parameters_new=params_new, SIE_gamma_mean = SIE_gamma_mean,
                           SIE_gamma_sigma = SIE_gamma_sigma)

    process_samples(new_name, which_lens_indexes, errors = errors, N_pert=N_pert)

def resample_sys(num, process_only):
    num = int(num)
    errors = [0, 0.08]
    src_mean = 0.03

    if num == 1:
        resample_chain(a0_area=0.012, logmhm=4.95, src_size=src_mean, LOS_norm=1, errors=errors,
                   N_pert=10, process_only=process_only, name='WDM_sim_7.7_.012',
                   which_lens_indexes=np.arange(1, 47))
    elif num == 2:
        resample_chain(a0_area=0.023, logmhm=4.95, src_size=src_mean, LOS_norm=1, errors=errors,
                   N_pert=10, process_only=process_only, name='WDM_sim_7.7_.012',
                   which_lens_indexes=np.arange(1, 47))
    elif num == 3:
        resample_chain(a0_area=0.0125, logmhm=7.7, src_size=src_mean, LOS_norm=1, errors=errors,
                   N_pert=10, process_only=process_only, name='WDM_sim_7.7_.012',
                   which_lens_indexes=np.arange(1, 42))

    elif num == 4:
        resample_chain(a0_area=0.02, logmhm=7.8, src_size=src_mean, LOS_norm=1, errors=errors,
                   N_pert=15, process_only=process_only, name='WDM_sim_7.7_.012',
                   which_lens_indexes=np.arange(1, 47))
    elif num == 5:
        resample_chain(a0_area=0.015, logmhm=7.7, src_size=src_mean, LOS_norm=1, errors=errors,
                       N_pert=15, process_only=process_only, name='WDM_sim_7.7_.012',
                       which_lens_indexes=np.arange(1, 47))

#process_raw('jpl_sim1', np.arange(1,6))

#resample_sys(5, True)
#resample_sys(5, False)
#resample_sys(5, False)
#resample_sys(7, False)
#import sys
#resample_sys(sys.argv[1], False)
#process_samples('WDM_sim_7.7_.012', np.arange(1,27), N_pert=10, errors=[0,0.04,0.08])
