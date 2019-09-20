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
                  mhalomin=None, keep_p=None, mhmmax=None):

    if not isinstance(names, list):
        names = [names]

    fluxes_out, obs, params_out, full_params_out, head = extract_chain_single(names[0], sim_name, zlens=zlens,
                                                                              sigmasubmax=sigmasubmax,
                                                                              observed_fluxes=observed_fluxes,
                                                                              mhalomin=mhalomin,mhmmax=mhmmax)

    if keep_p is True:
        extended_shape0 = float(params_out.shape[0])

    for count, name in enumerate(names):

        if count==0:
            continue
        if count > 1:
            keep_p=False

        f, _, p, fp, _ = extract_chain_single(name, sim_name, zlens=zlens, sigmasubmax=sigmasubmax,
                                                  observed_fluxes=observed_fluxes,mhalomin=mhalomin,mhmmax=mhmmax)

        fluxes_out = np.vstack((fluxes_out, f))
        params_out = np.vstack((params_out, p))
        full_params_out = np.vstack((full_params_out, fp))

    return fluxes_out, obs, params_out, full_params_out, head

def extract_chain_single(name, sim_name, start_idx=1, zlens=None, sigmasubmax=None,
                         observed_fluxes=None, mhalomin=None, mhmmax=None):

    chain_info_path = chainpath_out + 'raw_chains_CDM/' + name + '/simulation_info.txt'

    params_varied, varyparams_info, Ncores, cores_per_lens = read_chain_info(chain_info_path)

    #lens_config, lens_R_index = read_R_index(chainpath_out+chain_name+'/R_index_config.txt',0)

    chain_file_path = chainpath_out + 'raw_chains_CDM/' + name +'/chain'

    params_header = None
    order = None
    print(observed_fluxes)
    if ~os.path.exists(chainpath_out+'processed_chains/' + sim_name + '/'):
        create_directory(chainpath_out+'processed_chains/' + sim_name + '/')

    if ~os.path.exists(chainpath_out + 'processed_chains/' + sim_name + '/'+ name):
        create_directory(chainpath_out + 'processed_chains/' + sim_name + '/' + name)

    copy_directory(chain_info_path,chainpath_out+'processed_chains/' + sim_name + '/' + name + '/')

    end = int(start_idx + cores_per_lens)

    init = True

    for i in range(start_idx+1, end):
        folder_name = chain_file_path + str(i)+'/'

        print(folder_name)

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
                params = np.delete(params, col, 1)
            else:
                assert zlens is not None
                zd = np.array([zlens]*np.shape(params)[0])

            if 'satellite_thetaE' in params_varied:
                col = params_varied.index('satellite_thetaE')
                sat = params[:,col:(col+3)]
            if 'satellite_thetaE_1' in params_varied:
                col = params_varied.index('satellite_thetaE_1')
                sat = params[:,col:(col+6)]

            params = params[:,0:6]

            # SIE_gamma
            params = np.delete(params, 1, 1)
            condition = np.logical_and(params[:,4]<-1.85, params[:,4]>-1.95)
            condition = False

            macro_model = np.loadtxt(folder_name + '/macro.txt')

            macro_model[:,3] = transform_ellip(macro_model[:,3])

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
                r[np.where(np.isnan(r))] = 0
                r[np.where(r <= 0)] = 1e-5

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

def process_raw(name, Npert, sim_name='grism_quads_CDM',keep_N=3000,sigmasubmax=None,mhalomin=None,
                deplete=False):

    """
    coverts output from cluster into single files for each lens
    """

    header = 'f1 f2 f3 f4 stat srcsize sigmasub deltalos mparent alpha c0 beta zeta re gx gy eps epstheta shear sheartheta gmacro'
    header += ' lens_redshift'
    keep_inds = [0,1,2,3]
    uncertainty_in_ratios=False
    zlens = None
    observed_fluxes = None
    run_name = deepcopy(name)
    keep_p = True

    if name[0:8] == 'lens1422':
        run_name = ['lens1422_CDM']
        zlens=0.36
        sigmas = [0.011, 0.01, 0.013]
        keep_inds = [0,1,2]
    elif name[0:8]=='lens0435':
        run_name = ['lens0435_CDM']
        zlens=0.45
        header += ' satellite_thetaE satellite_x satellite_y'
        sigmas = [0.05, 0.049, 0.048, 0.056]
    elif name[0:8]=='lens2038':
        run_name = ['lens2038_CDM']
        keep_p = False
        zlens=0.23
        sigmas = [0.01, 0.017, 0.022, 0.022]
    elif name[0:8]=='lens1606':
        run_name = ['lens1606_CDM']
        header +=' satellite_thetaE satellite_x satellite_y'
        sigmas = [0.03, 0.03, 0.02/0.59, 0.02/0.79]
        #sigmas = [0.00001]*4
    elif name[0:8]=='lens0405':
        run_name = ['lens0405_CDM']

        #sigmas = [0.04, 0.03/0.7, 0.04/1.28, 0.05/0.94]
        observed_fluxes = np.array([1., 0.65, 1.25, 1.17]) * 1.25 ** -1
        sigmas = [0.04*observed_fluxes[0] ** -1, 0.04*observed_fluxes[1] ** -1,
                  0.03**observed_fluxes[2] ** -1, 0.04*observed_fluxes[3] ** -1]

    elif name[0:8] == 'lens2033':
        run_name = ['lens2033_CDM']

        header += ' satellite_thetaE_1 satellite_x_1 satellite_y_1 satellite_thetaE_2 satellite_x_2 satellite_y_2'
        zlens=0.66
        sigmas = np.array([0.03, 0.03/0.65, 0.02/0.5, 0.02/0.53])
        #sigmas = np.array([0.2, 0.2/0.65, 0.2/0.5, 0.2/0.53])

    elif name[0:8]=='lens2026':
        run_name = ['lens2026_CDM']
        sigmas = [0.02, 0.02/0.75, 0.01/0.31, 0.01/0.28]
    elif name[0:8]=='lens0911':
        keep_p = False
        #run_name = ['lens0911_varyshear_extended',]
        #run_name += ['lens0911_old', 'lens0911_aurora']
        run_name = ['lens0911_CDM']

        #run_name = ['lens0911_varyshear_hoffman_2']
        header += ' satellite_thetaE satellite_x satellite_y'
        zlens=0.77
        sigmas = [0.04/0.56, 0.05, 0.04/0.53, 0.04/0.24]
    elif name[0:8] == 'lens0128':
        run_name = ['lens0128_CDM']
        zlens=1.145
        uncertainty_in_ratios=True
        keep_inds = [0,1,2]
        sigmas = [0.029, 0.029, 0.032]
    elif name[0:8] == 'lens1115':
        run_name = ['lens1115_CDM']
        zlens=0.31
        uncertainty_in_ratios=True
        keep_inds = [0,1,2]
        #sigmas = [0.1, 0.1, 0.1]
        sigmas = [0.1] * 3

    elif name[0:8] == 'lens0414':
        run_name = ['lens0414_CDM']
        #observed_fluxes = np.array([1, 0.83, 0.36, 0.16])
        #sigmas = [0.1, 0.1, 0.1]
        sigmas = [0.1] * 3
        observed_fluxes = np.array([1, 0.903, 0.389, 0.145])
        #sigmas = [0.05, 0.04, 0.04]

        zlens=0.96
        uncertainty_in_ratios=True
        keep_inds = [0,1,2]

    sigmas = np.array(sigmas)

    print('loading chains... ')
    fluxes,fluxes_obs,parameters,all,_ = extract_chain(run_name, sim_name, zlens=zlens, sigmasubmax=sigmasubmax,
                                                       observed_fluxes=observed_fluxes,
                                                       mhalomin=mhalomin, keep_p=keep_p)
    print('done.')

    all = np.squeeze(all)
    fluxes, fluxes_obs = np.squeeze(fluxes), np.squeeze(fluxes_obs)

    chain_file_path = chainpath_out + 'processed_chains/grism_quads_CDM/' + name + '/'

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

for lensname in ['lens2038','lens2026','lens0435','lens0911','lens0405','lens1422','lens1606','lens2033']:

    process_raw(lensname, 10, keep_N=1000)
