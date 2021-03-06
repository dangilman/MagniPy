from MagniPy.ABCsampler.ChainOps import *
from MagniPy.ABCsampler.Chain import ChainFromChain, ChainFromSamples
from MagniPy.Analysis.Statistics.routines import build_densities, barplothist

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


def CI(centers, heights, percentile):
    total = np.sum(heights)
    summ, index = 0, 0
    while summ < total * percentile:
        summ += heights[index]
        index += 1

    return centers[index - 1]

def bootstrap_intervals(chain_name, Nlenses, which_lenses, Nbootstraps, error,
                        tol, bandwidth_scale=0.6, bins=20):

    if not isinstance(Nlenses, list):
        Nlenses = [Nlenses]

    chain_master = ChainFromSamples(chain_name, which_lens = which_lenses, error=0, n_pert=1, load=False)

    low95_interval, low68_interval = [], []
    high95_interval, high68_interval = [], []

    for nlens in Nlenses:
        print('computing '+str(nlens)+' lenses... ')
        low95, high95 = [], []

        for i in range(0, Nbootstraps):

            lens_list = np.random.randint(1, len(which_lenses), nlens)
            print(lens_list)
            parent_chain = ChainFromChain(chain_master, lens_list, load_flux=True)

            print('adding perturbations.... ')
            parent_chain._add_perturbations(error, tol)

            new_chain = ChainFromSamples(chain_name, np.arange(0,nlens),
                                         error=0, n_pert=1, load=False,
                                         from_parent = parent_chain)
            print('evaluading KDE... ')
            new_chain.eval_KDE(tol = tol, nkde_bins=bins, bandwidth_scale=bandwidth_scale,
                               save_to_file=False)

            marginalized = new_chain.get_projection(['log_m_break'], load_from_file=False)

            coords = np.linspace(chain_master.pranges['log_m_break'][0], chain_master.pranges['log_m_break'][1], len(marginalized))
            bar_centers, bar_width, bar_heights = barplothist(marginalized, coords, None)

            h95 = CI(bar_centers, bar_heights, 0.95)
            l95 = CI(bar_centers, bar_heights, 0.05)

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

    for which_lens in which_lens_indexes:

        fluxes, fluxes_obs, parameters, header, newgamma = resample(name, which_lens, parameters_new,
                                                                    SIE_gamma_mean=SIE_gamma_mean,
                                                                    SIE_gamma_sigma=SIE_gamma_sigma)
        print(header)

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

    #with open(chainpath_out + 'processed_chains/'+new_name + '/gamma_values.txt', 'w') as f:

        #f.write('re-sampled gamma '+str(SIE_gamma_mean)+' +\- '+str(SIE_gamma_sigma)+'\n')
    #    for g in new_gamma:
    #        f.write(str(g)+'\n')

def process_raw(name=str,which_lenses=[], counter_start = 0, swap_cols = None):

    """
    coverts output from cluster into single files for each lens
    """

    for which_lens in which_lenses:

        fluxes,fluxes_obs,parameters,header = extract_chain(name,which_lens)
        if swap_cols is not None:
            parameters = parameters[:, np.array(swap_cols)]
            header_split = header.split(' ')
            new_head = ''

            for col in swap_cols:
                new_head += header_split[col] + ' '
            header = new_head

        stack_chain(name, which_lens, parameters, np.squeeze(fluxes_obs), fluxes, header, counter_start)

def process_samples(chain_name, which_lenses, N_pert=1, errors=None, swap_cols = [4, 2, 1, 3, 0]):

    for k, which_lens in enumerate(which_lenses):

        fluxes, observed_fluxes, parameters, params_header = extract_chain_fromprocessed(chain_name, which_lens)
        print('number of samples in lens '+str(k+1)+': ', np.shape(fluxes)[0])
        add_flux_perturbations(chain_name, which_lens, parameters, observed_fluxes, fluxes, errors = errors, N_pert = N_pert)

def resample_chain(a0_area=None, logmhm=None, src_size=None, LOS_norm=1.0, errors = [0, 0.04],N_pert=1,
                   process_only = False, SIE_gamma_mean=2.06,
                   SIE_gamma_sigma=0.06, logmhm_sigma = 0.025,
                   src_size_sigma = 5, a0_area_sigma = 0.1, LOS_norm_sigma = 0.02,
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

        params_new = {'a0_area': [a0_area, a0_area_sigma], 'log_m_break': [logmhm, logmhm_sigma],
                                    'source_size_kpc': [src_size, src_size_sigma],
                                    'LOS_normalization': [LOS_norm, LOS_norm_sigma],
                                    'SIE_gamma': [SIE_gamma_mean, SIE_gamma_sigma]}

        _resample_chain(name = name, new_name = new_name, which_lens_indexes=which_lens_indexes,
                       parameters_new=params_new, SIE_gamma_mean = SIE_gamma_mean,
                           SIE_gamma_sigma = SIE_gamma_sigma)

    process_samples(new_name, which_lens_indexes, errors = errors, N_pert=N_pert)

    return new_name

def resample_chain_sidm(a0_area=None, SIDMcross=None, errors = [0, 0.02],N_pert=1,
                   process_only = False, SIDMcross_sigma = 0.1, SIE_gamma_mean=2.08,
                   SIE_gamma_sigma=0.03, a0_area_sigma = 0.003, name='WDM_sim_7.7_.012',
                        which_lens_indexes = None, ending = '', src_size = 0.02, src_size_sigma = 0.005,
                        LOS_norm = 1, LOS_norm_sigma = 0.1):

    new_name = 'SIDM_'

    new_name += 'sigma'+str(a0_area)+'_cross'+str(SIDMcross) + ending

    if process_only is False:
        params_new = {'a0_area': [a0_area, a0_area_sigma],
                      'source_size_kpc': [src_size, src_size_sigma],
                      'LOS_normalization': [LOS_norm, LOS_norm_sigma],
                      'SIE_gamma': [SIE_gamma_mean, SIE_gamma_sigma], 'SIDMcross':
                          [SIDMcross, SIDMcross_sigma]}

        _resample_chain(name = name, new_name = new_name, which_lens_indexes=which_lens_indexes,
                       parameters_new=params_new)
    process_samples(new_name, which_lens_indexes, errors = errors, N_pert=N_pert)

    return new_name

def resample_sys(num, process_only):
    num = int(num)
    errors = [0.02]
    if errors[0] == 0:
        Npert = 1
    else:
        Npert = 5

    src_mean = 0.025

    if num == 1:
        resample_chain_sidm(a0_area=0.03, SIDMcross=0.3, SIDMcross_sigma=0.2, src_size = src_mean,
                            errors=errors,
                   N_pert=Npert, process_only=process_only, name='SIDMsim', LOS_norm=1.,
                   which_lens_indexes=np.arange(1, 51))
    elif num == 2:
        resample_chain_sidm(a0_area=0.03, SIDMcross=9, SIDMcross_sigma=0.2, src_size=src_mean,
                            errors=errors,
                            N_pert=Npert, process_only=process_only, name='SIDMsim', LOS_norm=1.,
                            which_lens_indexes=np.arange(1, 51))
    elif num == 3:
        resample_chain_sidm(a0_area=0.025, SIDMcross=0.25, SIDMcross_sigma=0.2, src_size=src_mean,
                            errors=errors,
                            N_pert=1, process_only=process_only, name='coldSIDM_full', LOS_norm=1.,
                            which_lens_indexes=np.arange(1, 37))
    elif num == 4:
        resample_chain(a0_area=1.8,logmhm=4.95, src_size=src_mean, LOS_norm=1.1, errors=errors,
                   N_pert=25, process_only=process_only, name='WDM_7.7_sigma0.012_srcsize35',
                   which_lens_indexes=np.arange(1, 51))

    elif num == 5:
        resample_chain(a0_area=1.25, logmhm=7.7, src_size=src_mean, LOS_norm=1., errors=errors,
                       N_pert=25, process_only=process_only, name='WDM_7.7_sigma0.012_srcsize35',
                       which_lens_indexes=np.arange(1, 51))
    elif num == 6:
        resample_chain(a0_area=1.5, logmhm=7.7, src_size=src_mean, LOS_norm=1.1, errors=errors,
                       N_pert=25, process_only=process_only, name='WDM_7.7_sigma0.012_srcsize35',
                       which_lens_indexes=np.arange(1, 51))
    elif num == 7:
        resample_chain(a0_area=0, logmhm=4.9, src_size=src_mean, LOS_norm=1, errors=errors,
                       N_pert=25, process_only=process_only, name='WDM_7.7_sigma0.012_srcsize35',
                       which_lens_indexes=np.arange(1, 51))

process_raw('forecast_benson', np.arange(1,2))
#process_samples('SIDMsim', np.arange(1,51), 5, [0.02])
#resample_sys(1, True)
#resample_sys(2, False)
#resample_sys(1, False)
#resample_sys(1, False)
#process_samples('SIDM_sigma0.03_cross0.2', np.arange(1,51), errors = [0.02, 0.04], N_pert=5)
#process_samples('SIDM_sigma0.03_cross9', np.arange(1,51), errors = [0.02, 0.04], N_pert=5)
#process_samples('SIDM_sigma0.025_cross0.25', np.arange(1,37), errors = [0.02, 0.04], N_pert=5)
#process_samples('SIDM_sigma0.015_cross9', np.arange(1,37), errors = [0.02, 0.04], N_pert=5)
#process_samples('SIDM_sigma0.015_cross5', np.arange(1,37), errors = [0.02, 0.04], N_pert=5)
#resample_sys(2, False)
#resample_sys(1, False)
#process_raw('hoffman_run2', np.arange(1,11), counter_start=5)
#process_raw('hoffman_run3', np.arange(1,6), counter_start=15)
#process_raw('hoffman_run4', np.arange(1,6), counter_start = 20)
#process_raw('hoffman_run5', np.arange(1,11), counter_start = 25)
#process_raw('jpl_sim1', np.arange(1,6), counter_start = 35)
#process_raw('jpl_sim2', np.arange(1,6), counter_start = 40)
#process_raw('SIDM_run', np.arange(1,12), counter_start = 0)
#exit(1)
#process_samples('coldSIDM', np.arange(1,26), 1, [0])
#resample_sys(1, False)
#resample_sys(2, False)
#resample_sys(3, False)
#resample_sys(4, False)
#resample_sys(5, False)
#resample_sys(6, False)
#resample_sys(7, False)
#resample_sys(4, False)
#resample_sys(5, False)
#resample_sys(5, False)
#resample_sys(7, False)
#import sys
#resample_sys(sys.argv[1], False)
#process_samples('WDM_sim_7.7_corrected', np.arange(1,31), N_pert=10, errors=[0,0.02,0.04,0.08])
