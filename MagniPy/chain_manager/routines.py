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

def extract_chain(names, sim_name, start_idx=1, zlens=None, sigmasubmax=None, observed_fluxes=None, mhalomin=None):

    if not isinstance(names, list):
        names = [names]

    fluxes_out, obs, params_out, full_params_out, head = extract_chain_single(names[0], sim_name, zlens=zlens,
                                                                              sigmasubmax=sigmasubmax,
                                                                              observed_fluxes=observed_fluxes,
                                                                              mhalomin=mhalomin)

    for count, name in enumerate(names):

        if count==0:
            continue

        f, _, p, fp, _ = extract_chain_single(name, sim_name, zlens=zlens,
                                                                           sigmasubmax=sigmasubmax,
                                                                           observed_fluxes=observed_fluxes,mhalomin=mhalomin)
        fluxes_out = np.vstack((fluxes_out, f))
        params_out = np.vstack((params_out, p))
        full_params_out = np.vstack((full_params_out, fp))

    return fluxes_out, obs, params_out, full_params_out, head

def extract_chain_single(name, sim_name, start_idx=1, zlens=None, sigmasubmax=None,
                         observed_fluxes=None, mhalomin=None):

    chain_info_path = chainpath_out + 'raw_chains/' + name + '/simulation_info.txt'

    params_varied, varyparams_info, Ncores, cores_per_lens = read_chain_info(chain_info_path)

    #lens_config, lens_R_index = read_R_index(chainpath_out+chain_name+'/R_index_config.txt',0)

    chain_file_path = chainpath_out + 'raw_chains/' + name +'/chain'

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

            if sigmasubmax is not None:
                keep = np.where(params[:,1] < sigmasubmax)[0]

            else:
                keep = np.where(params[:,1]<10000)[0]

            fluxes = fluxes[keep,:]
            params=params[keep,:]
            macro_model=macro_model[keep,:]

            if mhalomin is not None:
                keep = np.where(params[:, 3] > mhalomin)[0]
            else:
                keep = np.where(params[:,3]<10000)[0]

            fluxes = fluxes[keep, :]
            params = params[keep, :]
            macro_model = macro_model[keep, :]

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

def process_raw(name, Npert, sim_name='grism_quads',keep_N=5000,sigmasubmax=None,mhalomin=None):

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

    if name[0:8] == 'lens1422':
        zlens=0.36
        sigmas = [0.011, 0.01, 0.013]
        keep_inds = [0,1,2]
    elif name[0:8]=='lens0435':
        zlens=0.45
        header += ' satellite_thetaE satellite_x satellite_y'
        sigmas = [0.05, 0.049, 0.048, 0.056]
    elif name[0:8]=='lens2038':
        zlens=0.23
        sigmas = [0.01, 0.017, 0.022, 0.022]
    elif name[0:8]=='lens1606':
        run_name = ['lens1606', 'lens1606_extended']
        header +=' satellite_thetaE satellite_x satellite_y'
        sigmas = [0.03, 0.03, 0.02/0.59, 0.02/0.79]
        #sigmas = [0.00001]*4
    elif name[0:8]=='lens0405':
        run_name = ['lens0405', 'lens0405_extended']
        sigmas = [0.04, 0.03/0.7, 0.04/1.28, 0.05/0.94]
    elif name[0:8] == 'lens2033':
        header += ' satellite_thetaE_1 satellite_x_1 satellite_y_1 satellite_thetaE_2 satellite_x_2 satellite_y_2'
        zlens=0.66
        sigmas = [0.03, 0.03/0.65, 0.04, 0.02/0.53]
    elif name[0:8]=='lens2026':
        sigmas = [0.02, 0.02/0.75, 0.01/0.31, 0.01/0.28]
    elif name[0:8]=='lens0911':
        header += ' satellite_thetaE satellite_x satellite_y'
        zlens=0.77
        sigmas = [0.04/0.56, 0.05, 0.04/0.53, 0.04/0.24]
    elif name[0:8] == 'lens0128':
        zlens=1.145
        uncertainty_in_ratios=True
        keep_inds = [0,1,2]
        sigmas = [0.029, 0.029, 0.032]
    elif name[0:8] == 'lens1115':
        zlens=0.31
        uncertainty_in_ratios=True
        keep_inds = [0,1,2]
        sigmas = [0.1, 0.1, 0.1]
    elif name[0:8] == 'lens0414':
        #observed_fluxes = np.array([1, 0.83, 0.36, 0.16])
        sigmas = [0.1, 0.1, 0.1]

        observed_fluxes = np.array([1, 0.903, 0.389, 0.145])
        #sigmas = [0.05, 0.04, 0.04]

        zlens=0.96
        uncertainty_in_ratios=True
        keep_inds = [0,1,2]

    sigmas = np.array(sigmas)

    print('loading chains... ')
    fluxes,fluxes_obs,parameters,all,_ = extract_chain(run_name, sim_name, zlens=zlens, sigmasubmax=sigmasubmax,
                                                       observed_fluxes=observed_fluxes, mhalomin=mhalomin)
    print('done.')

    all = np.squeeze(all)
    fluxes, fluxes_obs = np.squeeze(fluxes), np.squeeze(fluxes_obs)
    chain_file_path = chainpath_out + 'processed_chains/grism_quads/' + name + '/'
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

#for lensname in ['1422','2038','0435','0405','1606','2026','2033','0128', '0414', '1115', '0911']:
#    process_raw('lens'+lensname, 10, mhalomin=13)
process_raw('lens'+'0911', 10)
#process_raw('lens'+'0405', 10)
#process_raw('lens'+'1422', 10, mhalomin=13)
#process_raw('lens2038_highnorm', 10)
#process_raw('lens1422_highnorm_fixshear', 10)
