import numpy as np
from MagniPy.paths import *
from MagniPy.util import create_directory, copy_directory, read_data
from copy import deepcopy, copy

def make_samples_histogram(data, ranges, nbins, weights):
    density = np.histogramdd(data, range=ranges, density=True, bins=nbins,
                                weights=weights)[0]
    return density


def make_histograms(data_list, ranges, nbins, weights_list):
    density = 0
    for (data, weights) in zip(data_list, weights_list):
        density += make_samples_histogram(data, ranges, nbins, weights)

    return density

def transform_ellip(ellip):

    return ellip * (2 - ellip) ** -1

def read_chain_info(fname):

    with open(fname,'r') as f:
        lines = f.read().splitlines()

    Ncores, cores_per_lens = read_run_partition(fname)

    params_varied = []
    varyparams_info = {}
    nextline = False

    for line in lines:

        if line == '# params_varied':
            nextline = True
            continue

        if nextline:
            if len(line)==0:
                nextline=False
                break
            params_varied.append(line)

    for pname in params_varied:

        args = {}

        for line in lines:

            if line == pname+':':
                nextline=True
                continue

            if nextline:

                if len(line)==0:
                    nextline=False
                    break
                args[line.split(' ')[0]] = line.split(' ')[1]

            varyparams_info[pname] = args

    truth_dic = {}
    for line in lines:

        if line == '# truths':
            nextline = True
            continue

        if nextline:
            if len(line)==0:
                nextline=False
                break

            truth_dic[line.split(' ')[0]] = float(line.split(' ')[1])

    return params_varied, varyparams_info, Ncores, cores_per_lens

def read_run_partition(fname):

    with open(fname, 'r') as f:
        lines = f.readlines()

    Ncores = int(lines[1])
    cores_per_lens = int(lines[4])

    return Ncores, cores_per_lens

def extract_chain(names, sim_name, start_idx=1, zlens=None, sigmasubmax=None, observed_fluxes=None,
                  mhalomin=None, keep_p=None, mhmmax=None, mhm_flags=None):

    if not isinstance(names, list):
        names = [names]

    fluxes_out, obs, params_out, full_params_out, head = extract_chain_single(names[0], sim_name, zlens=zlens,
                                                                              sigmasubmax=sigmasubmax,
                                                                              observed_fluxes=observed_fluxes,
                                                                              mhalomin=mhalomin,mhmmax=mhmmax,mhm_flag=mhm_flags[0])

    if keep_p is True:
        extended_shape0 = float(params_out.shape[0])

    for count, name in enumerate(names):

        if count==0:
            continue
        if count > 1:
            keep_p=False

        f, _, p, fp, _ = extract_chain_single(name, sim_name, zlens=zlens, sigmasubmax=sigmasubmax,
                                                  observed_fluxes=observed_fluxes,mhalomin=mhalomin,mhmmax=mhmmax,
                                              mhm_flag=mhm_flags[count])
        if keep_p is True:
            keep_percent = extended_shape0 * ((3./7) * p.shape[0])**-1

            nmax = int(keep_percent * p.shape[0])
        else:
            nmax = int(p.shape[0])

        fluxes_out = np.vstack((fluxes_out, f[0:nmax,:]))
        params_out = np.vstack((params_out, p[0:nmax,:]))
        full_params_out = np.vstack((full_params_out, fp[0:nmax,:]))

    return fluxes_out, obs, params_out, full_params_out, head

def extract_chain_single(name, sim_name, start_idx=1, zlens=None, sigmasubmax=None,
                         observed_fluxes=None, mhalomin=None, mhmmax=None, mhm_flag=None):

    chain_info_path = chainpath_out + 'raw_chains_CDM/' + name + '/simulation_info.txt'

    params_varied, varyparams_info, Ncores, cores_per_lens = read_chain_info(chain_info_path)

    #lens_config, lens_R_index = read_R_index(chainpath_out+chain_name+'/R_index_config.txt',0)

    chain_file_path = chainpath_out + 'raw_chains_CDM/' + name +'/chain'

    params_header = None
    order = None

    if ~os.path.exists(chainpath_out+'processed_chains/' + sim_name + '/'):
        create_directory(chainpath_out+'processed_chains/' + sim_name + '/')

    if ~os.path.exists(chainpath_out + 'processed_chains/' + sim_name + '/'+ name):
        create_directory(chainpath_out + 'processed_chains/' + sim_name + '/' + name)

    copy_directory(chain_info_path,chainpath_out+'processed_chains/' + sim_name + '/' + name + '/')

    end = int(start_idx + cores_per_lens)

    init = True

    for i in range(start_idx+1, end):
        folder_name = chain_file_path + str(i)+'/'
        #print(folder_name)
        try:

            fluxes = np.loadtxt(folder_name + '/fluxes.txt')
            if observed_fluxes is None:
                obs_data = read_data(folder_name + '/lensdata.txt')
                observed_fluxes = obs_data[0].m
            params = np.loadtxt(folder_name + '/parameters.txt', skiprows=1)
            zd, sat=None, None
            if 'lens_redshift' in params_varied:
                col = params_varied.index('lens_redshift')
                zd = params[:, col]
            else:
                assert zlens is not None
                zd = np.array([zlens]*np.shape(params)[0])

            if 'satellite_thetaE' in params_varied:
                col = params_varied.index('satellite_thetaE')
                sat = params[:,col:(col+3)]
            if 'satellite_thetaE_1' in params_varied:
                col = params_varied.index('satellite_thetaE_1')
                sat = params[:,col:(col+6)]

            # shear
            params = np.delete(params, 7, 1)
            # SIE_gamma
            params = np.delete(params, 1, 1)


            condition = np.logical_and(params[:, 4] < -1.85, params[:, 4] > -1.95)

            condition = False

            params = params[:,0:6]

            macro_model = np.loadtxt(folder_name + '/macro.txt')

            macro_model[:,3] = transform_ellip(macro_model[:,3])

            macro_normed = np.empty((np.shape(macro_model)[0], 8))

            for col in range(0, 8):

                macro_normed[:,col] = macro_model[:,col] - np.mean(macro_model[:,col])

            macro_model = np.column_stack((macro_model[:,0:8], macro_normed))

            if zd is not None:
                macro_model = np.column_stack((macro_model, zd))
            if sat is not None:
                macro_model = np.column_stack((macro_model, sat))

            if condition is not False:

                params = params[condition,:]
                macro_model = macro_model[condition,:]
                fluxes = fluxes[condition,:]

            assert fluxes.shape[0] == params.shape[0]

        except:
            #print('didnt find a file... '+str(chain_file_path + str(i)+'/'))
            continue

        if params_header is None:
            with open(folder_name + '/parameters.txt', 'r') as f:
                lines = f.read().splitlines()
            head = lines[0].split(' ')
            params_header = ''
            for word in head:
                if word not in ['#', '']:
                    params_header += word + ' '

        if init:

            lens_fluxes = fluxes
            lens_params = params

            lens_all = np.column_stack((params, macro_model))

            init = False
        else:
            lens_fluxes = np.vstack((lens_fluxes,fluxes))
            lens_params = np.vstack((lens_params,params))
            new = np.column_stack((params, macro_model))

            lens_all = np.vstack((lens_all, new))

    return lens_fluxes,observed_fluxes.reshape(1,4),lens_params,lens_all,params_header

def add_flux_perturbations(fluxes, fluxes_obs, sigmas, N_pert, keep_inds, uncertainty_in_ratios):

    sample_inds = []
    statistics = []

    for k in range(1, N_pert + 1):

        if uncertainty_in_ratios:

            ncols = len(keep_inds)
            nrows = np.shape(fluxes)[0]
            perturbed_ratios = np.empty((nrows, ncols))

            norm = deepcopy(fluxes[:, 0])
            pr = np.empty((nrows, 3))

            for i in range(0, 3):
                r = fluxes[:, i + 1] * norm ** -1
                delta = np.random.normal(0, sigmas[i]*r)
                pr[:, i] = r + delta
            for i, idx in enumerate(keep_inds):
                perturbed_ratios[:, i] = pr[:, idx]

            ratios_obs = fluxes_obs[1:]/fluxes_obs[0]

            obs_ratios = [ratios_obs[k] for k in keep_inds]
            diff = (perturbed_ratios - np.array(obs_ratios)) ** 2

        else:
            ncols = len(keep_inds) - 1
            nrows = np.shape(fluxes)[0]
            perturbed_ratios = np.empty((nrows, ncols))
            perturbed_fluxes = np.empty((np.shape(fluxes)[0], len(keep_inds)))
            for i, ind in enumerate(keep_inds):
                f = deepcopy(fluxes[:,ind])
                delta_f = np.random.normal(0, sigmas[ind]*f)
                perturbed_fluxes[:,i] = f+delta_f

            norm = deepcopy(perturbed_fluxes[:,0])
            for col in range(0,len(keep_inds)-1):
                perturbed_ratios[:,col] = fluxes[:,col+1] * norm ** -1

            obs_flux = [fluxes_obs[index] for index in keep_inds]
            obs_normed = np.array(obs_flux)/obs_flux[0]
            obs_ratios = obs_normed[1:]

            diff = (perturbed_ratios - obs_ratios)**2

        summary_statistic = np.sqrt(np.sum(diff, 1))

        ordered_inds = np.argsort(summary_statistic)

        if k == 1:

            print('N < 0.01: ', np.sum(summary_statistic < 0.01*np.sqrt(3)))
            print('N < 0.02: ', np.sum(summary_statistic < 0.02*np.sqrt(3)))
            print('N < 0.03: ', np.sum(summary_statistic < 0.03*np.sqrt(3)))

        sample_inds.append(ordered_inds)
        statistics.append(summary_statistic[ordered_inds])

    return sample_inds, statistics

def process_raw(name, Npert, sim_name='grism_quads',keep_N=3000,sigmasubmax=None,mhalomin=None,mhmmax=None,
                deplete=False):

    """
    coverts output from cluster into single files for each lens
    """

    header = 'f1 f2 f3 f4 stat srcsize sigmasub deltalos mparent alpha mhm re gx gy eps epstheta shear sheartheta gmacro'
    header += ' re_normed gx_normed gy_normed eps_normed epstheta_normed shear_normed sheartheta_normed gmacro_normed'
    header += ' lens_redshift'
    keep_inds = [0,1,2,3]
    uncertainty_in_ratios=False
    zlens = None
    observed_fluxes = None
    run_name = deepcopy(name)
    keep_p = True
    samples_completed = 500000

    if name[0:8] == 'lens1422':
        run_name = ['lens1422_CDM_test']
        mhm_flags = [True] + [False] * (len(run_name) - 1)
        zlens=0.36
        sigmas = [0.011, 0.01, 0.013]
        keep_inds = [0,1,2]

    sigmas = np.array(sigmas)

    if mhm_flags is None:
        mhm_flags = [False]*len(run_name)
    else:
        print(mhm_flags, run_name)
        assert len(mhm_flags) == len(run_name)

    print('loading chains... ')
    fluxes,fluxes_obs,parameters,all,_ = extract_chain(run_name, sim_name, zlens=zlens, sigmasubmax=sigmasubmax,
                                                       observed_fluxes=observed_fluxes,
                                                       mhalomin=mhalomin, keep_p=keep_p, mhmmax=mhmmax, mhm_flags=mhm_flags)
    print('done.')

    all = np.squeeze(all)
    fluxes, fluxes_obs = np.squeeze(fluxes), np.squeeze(fluxes_obs)

    chain_file_path = chainpath_out + 'processed_chains/grism_quads_CDM/lens1422_CDM_test/'

    print('nrealizations: ', fluxes.shape[0])
    if ~os.path.exists(chain_file_path):
        create_directory(chain_file_path)

    print('sampling flux uncertainties, lens '+name[0:8]+'... ')
    inds_to_keep_list, statistics = add_flux_perturbations(fluxes, fluxes_obs, sigmas, Npert,
                                               keep_inds, uncertainty_in_ratios)
    print('done.')

    for i, indexes in enumerate(inds_to_keep_list):

        f = fluxes[indexes[0:keep_N], :]
        final_fluxes = np.column_stack((f, np.array(statistics[i][0:keep_N])))
        x = np.column_stack((final_fluxes, all[indexes[0:keep_N],:]))
        np.savetxt(chain_file_path + 'samples'+str(i+1)+'.txt', x, fmt='%.5f', header=header)

#if False:
#    process_raw('lens1115', 10, keep_N=2000)
#    process_raw('lens1115', 10, mhmmax=5.1)
#    process_raw('lens0414', 10, keep_N=2000)
#    process_raw('lens0414', 10, mhmmax=5.1)
#process_raw('lens2033', 10, keep_N=1000)
#process_raw('lens1413', 10, keep_N=1000)

if True:
    for lensname in ['1422']:
        process_raw('lens'+lensname, 10, keep_N=500)

    #for lensname in ['1606','2026','0128', '0414', '1115']:
    #    process_raw('lens' + lensname, 10, mhmmax=5.1, keep_N=500)

    #for lensname in ['2033', '0911', '0405']:
    #   process_raw('lens' + lensname, 10, mhmmax=5.1, keep_N=500)

#process_raw('lens0414', 10)
#process_raw('lens0414', 10, mhmmax=5.5)
#process_raw('lens0911', 10, keep_N=1000, mhmmax=5.5)
#for lensname in ['1422','2038','0435']:
#    process_raw('lens'+lensname, 10, keep_N=1500, mhmmax=5.3)

#for lensname in ['1606','2026','0128', '0414', '1115']:
#    process_raw('lens' + lensname, 10, keep_N=1500, deplete=True)

#for lensname in ['2033', '0911', '0405']:
#    process_raw('lens' + lensname, 10, keep_N=1500, deplete=True)

#for lensname in ['1422','2038','0435','0405','1606','2026','2033','0128', '0414', '1115', '0911']:
#    process_raw('lens' + lensname, 10, keep_N=3000, mhmmax=5.6)
#process_raw('lens0405', 10, mhmmax=5.8)
#process_raw('lens0405', 10)

#for lensname in ['1422','2038','0435','0405','1606','2026','2033','0128', '0414', '1115', '0911']:
#    process_raw('lens'+lensname, 10)

#process_raw('lens0911', 10)
#process_raw('lens0128', 10)
#process_raw('lens0128', 10)
#process_raw('lens'+'0911', 10)
#process_raw('lens'+'0405', 10)
#process_raw('lens'+'1422', 10, mhalomin=13)
#process_raw('lens2038_highnorm', 10)
#process_raw('lens1422_highnorm_fixshear', 10)
