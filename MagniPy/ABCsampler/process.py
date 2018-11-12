from MagniPy.ABCsampler.ChainOps import *

def new_chains_withsigma(chain_name, which_lenses, new_chain_name):

    compute_sigma_chains(chain_name, which_lenses, new_chain_name)
    return
    for which_lens in which_lenses:
        fluxes, fluxes_obs, parameters, header = extract_chain_fromprocessed(new_chain_name, which_lens)
        stack_chain(chain_name, which_lens, parameters, fluxes_obs, fluxes, header)

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

        if transform_fsub:

            new_fsub = l.fsub_from_sigmasub(sigma_sub, zd[which_lens-1], zs[which_lens-1])
            parameters_new.update({'fsub': [new_fsub, parameters_new['fsub'][1]]})

            parameters_new.update({'sigma_sub': [sigma_sub, parameters_new['fsub'][1]]})

            print(l.sigmasub_from_fsub(new_fsub, zd[which_lens-1], zs[which_lens-1]))

        fluxes, fluxes_obs, parameters, header, newgamma = resample(name, which_lens, parameters_new,
                                                                    SIE_gamma_mean=SIE_gamma_mean,
                                                                    SIE_gamma_sigma=SIE_gamma_sigma)

        if transform_fsub:
            pnames = list(filter(None, header.split(' ')))

            col = pnames.index('fsub')
            pnames[col] = 'sigma_sub'
            header = ''
            for pname in pnames:
                header += pname +' '

        if transform_fsub:

            parameters[:, -1] = l.sigmasub_from_fsub(parameters[:, -1], zd[which_lens-1], zs[which_lens-1])

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
            if transform_fsub and pname == 'fsub':
                continue
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

def resample_chain():

    name = 'WDM_run_7.7_tier2'
    new_name = 'WDM_7.7_sigma0.3_srcsize35'

    fsub = 0.015
    logmhm = 7.7
    src_size = 0.035
    LOS_norm = 1

    transform_fsub = True
    sigma_sub = 0.3

    which_lens_indexes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    params_new = {'fsub': [fsub, 0.001], 'log_m_break': [logmhm, 0.1],
                                'source_size_kpc': [src_size, 0.006],
                                'LOS_normalization': [LOS_norm, 0.05]}

    _resample_chain(name = name, new_name = new_name, which_lens_indexes=which_lens_indexes,
                   parameters_new=params_new, SIE_gamma_mean = 2.08,
                       SIE_gamma_sigma = 0.05, transform_fsub = transform_fsub, sigma_sub = sigma_sub)

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

#process_raw('WDM_run_7.7_tier2', which_lenses=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
#process_samples('CDM_fsub0.015_srcsize35', [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
#resample_chain()
#process_samples('WDM_7.7_sigma0.3_srcsize35', [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
#new_chains_withsigma('WDM_run_7.7_tier2', [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], 'WDM_7.7_sigma')
#process_samples('WDM_7.7_sigma', [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])

#resample_chain_sigma()
process_samples('WDM_7.7_sigma0.3_srcsize35', [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])