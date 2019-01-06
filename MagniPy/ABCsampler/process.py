from MagniPy.ABCsampler.ChainOps import *
from MagniPy.ABCsampler.Chain import ChainFromChain, ChainFromSamples
from MagniPy.Analysis.Statistics.routines import build_densities, barplothist
from MagniPy.Analysis.Statistics.routines import reweight_posteriors_individually
from MagniPy.Analysis.Statistics.routines import duplicate_with_cuts


def compute_joint_kde(chain_name, lens_index, nbins, error, n_pert = 15):

    which_lenses = np.arange(lens_index, lens_index + 1)
    chain_master = ChainFromSamples(chain_name, which_lens=which_lenses,
                  error=error, n_pert=n_pert, load=True)
    posteriors = chain_master.get_posteriors(2500)

    group1 = ['a0_area','log_m_break']
    group2 = ['a0_area', 'LOS_normalization']
    group3 = ['a0_area', 'source_size_kpc']
    group4 = ['a0_area', 'SIE_gamma']
    group5 = ['log_m_break', 'LOS_normalization']
    group6 = ['log_m_break', 'source_size_kpc']
    group7 = ['log_m_break', 'SIE_gamma']
    group8 = ['LOS_normalization', 'source_size_kpc']
    group9 = ['LOS_normalization', 'SIE_gamma']
    group10 = ['source_size_kpc','SIE_gamma']

    param_names = [group1, group2, group3, group4, group5, group6, group7,
                   group8, group9, group10]

    output = {}
    output_path_base = prefix + 'data/sims/densities/' + chain_name

    for pnames in param_names:
        sims, sim_pranges = build_densities([posteriors], pnames,
                                        chain_master.pranges, bandwidth_scale=1,
                                        xtrim=None, ytrim=None, steps=nbins,
                                        use_kde_joint=True, use_kde_marginal=True,
                                        reweight=True)
        if not os.path.exists(output_path_base+ '/lens' + str(lens_index) + '/'):
            create_directory(output_path_base+ '/lens' + str(lens_index) + '/')
        np.savetxt(output_path_base+ '/lens' + str(lens_index) + '/'+pnames[0]+'_'+pnames[1]+'_error_'+str(error)+'.txt',
                   X = np.array(sims[0][0]))

def bootstrap_intervals(chain_name, Nlenses, which_lenses, parameter, Nbootstraps, error,
                        tol, param_weights_individual=None, xtrim=None, ytrim=None, bins=40):

    if not isinstance(Nlenses, list):
        Nlenses = [Nlenses]

    #print('loading master.... ')
    chain_master = ChainFromSamples(chain_name, which_lens = which_lenses, error=0, n_pert=1, load=False)
    #params_reject = ['a0_area', 'log_m_break']
    #reject_ranges = [[0, 0.045], [4.8, 10]]

    #chain_master, pranges = duplicate_with_cuts(chain_master, tol, pnames_reject_list=params_reject,
    #                                          keep_ranges_list=reject_ranges)

    low95_interval, low68_interval = [], []
    high95_interval, high68_interval = [], []

    for nlens in Nlenses:
        print('computing '+str(nlens)+' lenses... ')
        low95, high95 = [], []
        low68, high68 = [], []

        for i in range(0, Nbootstraps):

            lens_list = np.random.randint(1, len(which_lenses), nlens)

            chain = ChainFromChain(chain_master, lens_list, load_flux=True)

            print('adding perturbations.... ')
            chain._add_perturbations(error, tol)

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

        stack_chain(new_name, which_lens, parameters, fluxes_obs, fluxes, header, counter_start=0)

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

def process_raw(name=str,which_lenses=[], counter_start = 0):

    """
    coverts output from cluster into single files for each lens
    """

    for which_lens in which_lenses:

        fluxes,fluxes_obs,parameters,header = extract_chain(name,which_lens)

        stack_chain(name, which_lens, parameters, np.squeeze(fluxes_obs), fluxes, header, counter_start)

def process_samples(chain_name, which_lenses, N_pert=1, errors=None):

    for which_lens in which_lenses:

        fluxes, observed_fluxes, parameters, params_header = extract_chain_fromprocessed(chain_name, which_lens)

        add_flux_perturbations(chain_name, which_lens, parameters, observed_fluxes, fluxes, errors = errors, N_pert = N_pert)

def resample_chain(a0_area=None, logmhm=None, src_size=None, LOS_norm=1.0, errors = [0, 0.04],N_pert=1,
                   process_only = False, SIE_gamma_mean=2.08, SIE_gamma_variance = 0.05,
                   SIE_gamma_sigma=0.03, logmhm_sigma = 0.05,
                   src_size_sigma = 0.007, a0_area_sigma = 0.001, LOS_norm_sigma = 0.02,
                   name='WDM_sim_7.7_.012', which_lens_indexes = None, ending = ''):

    #name = 'WDM_sim_7.7_.012'

    if logmhm > 6:
        new_name = 'WDM_'+str(logmhm)+'_'
    else:
        new_name = 'CDM_'

    new_name += 'sigma'+str(a0_area)+'_srcsize'+str(src_size) + ending

    if LOS_norm != 1:
        new_name += '_LOS'+str(LOS_norm)

    if process_only is False:

        SIE_mean = np.random.normal(SIE_gamma_mean, SIE_gamma_variance)

        params_new = {'a0_area': [a0_area, a0_area_sigma], 'log_m_break': [logmhm, logmhm_sigma],
                                    'source_size_kpc': [src_size, src_size_sigma],
                                    'LOS_normalization': [LOS_norm, LOS_norm_sigma],
                                    'SIE_gamma': [SIE_mean, SIE_gamma_sigma]}

        _resample_chain(name = name, new_name = new_name, which_lens_indexes=which_lens_indexes,
                       parameters_new=params_new, SIE_gamma_mean = SIE_gamma_mean,
                           SIE_gamma_sigma = SIE_gamma_sigma)

    process_samples(new_name, which_lens_indexes, errors = errors, N_pert=N_pert)

def resample_sys(num, process_only):
    num = int(num)
    errors = [0, 0.02, 0.04, 0.06, 0.08]
    src_mean = 35

    if num == 1:
        resample_chain(a0_area=1.6, logmhm=4.95, src_size=src_mean, LOS_norm=1, errors=errors,
                   N_pert=15, process_only=process_only, name='WDM_7.7_sigma0.012_srcsize35',
                   which_lens_indexes=np.arange(1, 51), ending = '_exact', SIE_gamma_variance=0.001)
    elif num == 2:
        resample_chain(a0_area=1.6, logmhm=4.95, src_size=src_mean, LOS_norm=1, errors=errors,
                   N_pert=15, process_only=process_only, name='WDM_7.7_sigma0.012_srcsize35',
                   which_lens_indexes=np.arange(1, 51), ending = '')
    elif num == 3:
        resample_chain(a0_area=1.8, logmhm=4.95, src_size=src_mean, LOS_norm=1, errors=errors,
                       N_pert=15, process_only=process_only, name='WDM_7.7_sigma0.012_srcsize35',
                       which_lens_indexes=np.arange(1, 51), ending = '_exact', SIE_gamma_variance=0.001)

    elif num == 4:
        resample_chain(a0_area=1.8,logmhm=4.95, src_size=src_mean, LOS_norm=1, errors=errors,
                   N_pert=15, process_only=process_only, name='WDM_7.7_sigma0.012_srcsize35',
                   which_lens_indexes=np.arange(1, 51), ending = '')

    elif num == 5:
        resample_chain(a0_area=1.25, logmhm=7.7, src_size=src_mean, LOS_norm=1., errors=errors,
                       N_pert=15, process_only=process_only, name='WDM_7.7_sigma0.012_srcsize35',
                       which_lens_indexes=np.arange(1, 51), ending = '_exact', SIE_gamma_variance=0.001)
    elif num == 6:
        resample_chain(a0_area=1.5, logmhm=7.7, src_size=src_mean, LOS_norm=1, errors=errors,
                       N_pert=15, process_only=process_only, name='WDM_7.7_sigma0.012_srcsize35',
                       which_lens_indexes=np.arange(1, 51), ending = '')


#process_raw('hoffman_run1', np.arange(1,6), counter_start=0)
#process_raw('hoffman_run2', np.arange(1,11), counter_start=5)
#process_raw('hoffman_run3', np.arange(1,6), counter_start=15)
#process_raw('hoffman_run4', np.arange(1,6), counter_start = 20)
#process_raw('hoffman_run5', np.arange(1,11), counter_start = 25)
#process_raw('jpl_sim1', np.arange(1,6), counter_start = 35)
#process_raw('jpl_sim2', np.arange(1,6), counter_start = 40)
#process_raw('jpl_sim3', np.arange(1,6), counter_start = 45)
#exit(1)
#process_samples('WDM_7.7_sigma0.012_srcsize35', np.arange(1,51), 15, [0,0.02,0.04,0.06,0.08])
#resample_sys(1, False)
#resample_sys(2, False)
#resample_sys(3, False)
#resample_sys(4, False)
#resample_sys(5, False)
#resample_sys(6, False)
#resample_sys(2, False)
#resample_sys(4, False)
#resample_sys(5, False)
#resample_sys(5, False)
#resample_sys(7, False)
#import sys
#resample_sys(sys.argv[1], False)
#process_samples('WDM_sim_7.7_corrected', np.arange(1,31), N_pert=10, errors=[0,0.02,0.04,0.08])
