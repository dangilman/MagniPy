import numpy as np
from MagniPy.paths import *
from MagniPy.util import create_directory, copy_directory, read_data
from copy import deepcopy

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

def read_run_partition(fname):

    with open(fname, 'r') as f:
        lines = f.readlines()

    Ncores = int(lines[1])
    cores_per_lens = int(lines[4])

    return Ncores, cores_per_lens, int(Ncores * cores_per_lens ** -1)

def extract_chain(name, sim_name, start_idx=1):

    chain_info_path = chainpath_out + 'raw_chains/' + name + '/simulation_info.txt'
    Ncores, cores_per_lens, Nlens = read_run_partition(chain_info_path)

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
    #for i in range(start,end):
    for i in range(start_idx+1, end):
        folder_name = chain_file_path + str(i)+'/'
        #print(folder_name)
        try:

            fluxes = np.loadtxt(folder_name + '/fluxes.txt')
            obs_data = read_data(folder_name + '/lensdata.txt')
            observed_fluxes = obs_data[0].m
            params = np.loadtxt(folder_name + '/parameters.txt', skiprows=1)
            params = np.delete(params, 1, 1)
            params = params[:,0:6]

            macro_model = np.loadtxt(folder_name + '/macro.txt')

            macro_model[:,3] = transform_ellip(macro_model[:,3])
            macro_model = macro_model[:,0:8]

            macro_normed = deepcopy(macro_model)
            for col in range(0, macro_normed.shape[1]):
                macro_normed[:,col] -= np.mean(macro_normed[:,col])

            macro_model = np.column_stack((macro_model, macro_normed))

            assert fluxes.shape[0] == params.shape[0]

        except:
            print('didnt find a file... '+str(chain_file_path + str(i)+'/'))
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

    observed_fluxes = observed_fluxes[order]

    return lens_fluxes[:,order],observed_fluxes.reshape(1,4),lens_params,lens_all,params_header

def add_flux_perturbations(fluxes, fluxes_obs, sigmas, N_pert, keep_inds):

    sample_inds = []
    statistics = []

    for k in range(1, N_pert + 1):

        ncols = len(keep_inds) - 1
        nrows = np.shape(fluxes)[0]
        perturbed_ratios = np.empty((nrows, ncols))

        perturbed_fluxes = np.empty((np.shape(fluxes)[0], len(keep_inds)))

        for i, ind in enumerate(keep_inds):
            perturbed_fluxes[:,i] += deepcopy(fluxes[:,ind])+\
                                     np.random.normal(0, sigmas[ind])

        norm = deepcopy(perturbed_fluxes[:,0])
        for col in range(0,len(keep_inds)-1):
            perturbed_ratios[:,col] = fluxes[:,col+1] * norm ** -1

        obs_flux = [fluxes_obs[index] for index in keep_inds]
        obs_normed = np.array(obs_flux)/obs_flux[0]
        obs_ratios = obs_normed[1:]

        diff = (perturbed_ratios - obs_ratios)**2
        summary_statistic = np.sqrt(np.sum(diff, 1))

        ordered_inds = np.argsort(summary_statistic)

        sample_inds.append(ordered_inds)
        statistics.append(summary_statistic[ordered_inds])

    return sample_inds, statistics

def process_raw(name, Npert, sim_name='grism_quads',keep_N=5000):

    """
    coverts output from cluster into single files for each lens
    """

    header = 'f1 f2 f3 f4 stat srcsize sigmasub deltalos mparent alpha mhm re gx gy eps epstheta shear sheartheta gmacro'
    header += ' re_normed gx_normed gy_normed eps_normed epstheta_normed shear_normed sheartheta_normed gmacro_normed'
    keep_inds = [0,1,2,3]
    if name=='lens1422' or name=='lens1422_highnorm':
        sigmas = [0.01, 0.01, 0.006]
        keep_inds = [0,1,2]
    elif name=='lens0435':
        sigmas = [0.02]*4
    elif name =='lens2038' or name=='lens2038_highnorm':
        #sigmas = [0.01, 0.02, 0.02, 0.01]
        sigmas = [0.01, 0.02, 0.02, 0.01]

    sigmas = np.array(sigmas)

    print('loading chains... ')
    fluxes,fluxes_obs,parameters,all,_ = extract_chain(name, sim_name)
    print('done.')

    all = np.squeeze(all)
    fluxes, fluxes_obs = np.squeeze(fluxes), np.squeeze(fluxes_obs)
    chain_file_path = chainpath_out + 'processed_chains/grism_quads/' + name + '/'
    print('nrealizations: ', fluxes.shape[0])
    if ~os.path.exists(chain_file_path):
        create_directory(chain_file_path)

    print('sampling flux uncertainties... ')
    inds_to_keep_list, statistics = add_flux_perturbations(fluxes, fluxes_obs, sigmas, Npert,
                                               keep_inds)
    print('done.')

    for i, indexes in enumerate(inds_to_keep_list):

        f = fluxes[indexes[0:keep_N], :]
        final_fluxes = np.column_stack((f, np.array(statistics[i][indexes[0:keep_N]])))
        x = np.column_stack((final_fluxes, all[indexes[0:keep_N],:]))
        np.savetxt(chain_file_path + 'samples'+str(i+1)+'.txt', x, fmt='%.5f', header=header)

process_raw('lens0435', 10)
