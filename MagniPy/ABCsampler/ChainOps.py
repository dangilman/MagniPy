from MagniPy.paths import *
from MagniPy.Analysis.Statistics.summary_statistics import *
import sys
from copy import copy
import numpy as np
from MagniPy.util import *
import ast

def read_chain_info(fname):

    with open(fname,'r') as f:
        lines = f.read().splitlines()

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

    return params_varied,truth_dic,varyparams_info


def read_run_partition(fname):

    with open(fname, 'r') as f:
        lines = f.readlines()

    Ncores = int(lines[1])
    cores_per_lens = int(lines[4])

    return Ncores, cores_per_lens, int(Ncores * cores_per_lens ** -1)

def read_R_index(fname,lens_index):

    with open(fname,'r') as f:
        lines = f.readlines()

    [config,R_index] = lines[lens_index].split(' ')

    return config,int(R_index)

def stack_chain(chain_name='', which_lens = None, parameters=None,fluxes_obs=None,
                           fluxes=None,header=str):

    fluxes, fluxes_obs = np.squeeze(fluxes), np.squeeze(fluxes_obs)

    chain_file_path = chainpath_out + 'processed_chains/' + chain_name + '/lens' + str(which_lens) + '/'

    if ~os.path.exists(chain_file_path):
        create_directory(chain_file_path)

    np.savetxt(chain_file_path + 'modelfluxes' + '.txt', fluxes, fmt='%.6f')
    np.savetxt(chain_file_path + 'observedfluxes' + '.txt',fluxes_obs, fmt='%.6f')
    np.savetxt(chain_file_path + 'samples.txt',parameters,fmt='%.6f',header=header)

def add_flux_perturbations(name, which_lens, parameters, fluxes_obs, fluxes, errors = None, N_pert = 1):

    if errors is None:
        errors = []

    if isinstance(errors,int) or isinstance(errors,float):
        errors = [errors]

    errors = [0]+errors

    for error in errors:

        for k in range(1, N_pert + 1):

            if error != 0:
                flux_perturbations_obs = np.random.normal(0, error * fluxes_obs)
            else:
                flux_perturbations_obs = np.zeros_like(fluxes_obs)

            perturbed_obs = fluxes_obs + flux_perturbations_obs

            perturbed_ratios_obs = perturbed_obs*perturbed_obs[0]**-1

            #np.savetxt(perturbed_fname_obs, perturbed_ratios_obs.reshape(1, 3), fmt='%.6f')

            ############################################################################

            perturbed_path = chainpath_out + 'chain_stats/' + name + '/lens' + str(which_lens) + '/'

            if not os.path.exists(chainpath_out + 'chain_stats/' + name):
                create_directory(chainpath_out + 'chain_stats/' + name)

            if not os.path.exists(perturbed_path):
                create_directory(chainpath_out + 'chain_stats/' + name + '/lens' + str(which_lens) + '/')

            if error == 0:

                flux_perturbations = np.zeros_like(fluxes)

            else:
                sigma = float(error) * fluxes
                flux_perturbations = np.random.normal(0, sigma)

            perturbed_fluxes = fluxes + flux_perturbations

            for i in range(0,int(np.shape(perturbed_fluxes)[0])):
                perturbed_fluxes[i,:] = perturbed_fluxes[i,:]*perturbed_fluxes[i,0]**-1

            summary_statistic = np.sqrt(np.sum((perturbed_fluxes-perturbed_ratios_obs)**2,axis=1))
            #elif stat == 'R':
            #    summary_statistic = R(perturbed_fluxes,perturbed_obs,config=lens_config)

            ordered_inds = np.argsort(summary_statistic)

            np.savetxt(perturbed_path + 'statistic_' + str(int(error * 100)) + 'error_' + str(k) + '.txt',X=summary_statistic[ordered_inds])
            np.savetxt(perturbed_path + 'params_'+str(int(error * 100)) + 'error_' + str(k) + '.txt',X=parameters[ordered_inds])

            if error == 0:
                break

def extract_chain_fromprocessed(chain_name = '', which_lens = None):

    route = chainpath_out + 'processed_chains/' + chain_name + '/lens'+str(which_lens)+'/'

    #lens_config, _ = read_R_index(chainpath_out + chain_name + '/R_index_config.txt', 0)

    fluxes = np.loadtxt(route+'modelfluxes.txt')

    with open(route + '/samples.txt', 'r') as f:
        lines = f.read().splitlines()
    head = lines[0].split(' ')
    params_header = ''
    for word in head:
        if word not in ['#', '']:
            params_header += word + ' '

    parameters = np.loadtxt(route + '/samples.txt')

    observed_fluxes = np.squeeze(np.loadtxt(route + 'observedfluxes.txt'))

    return fluxes, observed_fluxes, parameters, params_header

def extract_chain(chain_name='',which_lens = None):

    chain_info_path = chainpath_out + 'raw_chains/' + chain_name + '/simulation_info.txt'
    Ncores, cores_per_lens, Nlens = read_run_partition(chain_info_path)

    #lens_config, lens_R_index = read_R_index(chainpath_out+chain_name+'/R_index_config.txt',0)

    chain_file_path = chainpath_out + 'raw_chains/' + chain_name +'/chain'

    params_header = None
    order = None

    if ~os.path.exists(chainpath_out+'processed_chains/' + chain_name + '/'):
        create_directory(chainpath_out+'processed_chains/' + chain_name + '/')

    copy_directory(chain_info_path,chainpath_out+'processed_chains/' + chain_name + '/')

    if ~os.path.exists(chainpath_out + 'processed_chains/' + chain_name + '/lens' + str(which_lens) + '/'):
        create_directory(chainpath_out + 'processed_chains/' + chain_name + '/lens' + str(which_lens) + '/')

    start = int((which_lens-1)*cores_per_lens)
    end = int(start + cores_per_lens)
    init = True
    #for i in range(start,end):
    for i in range(start, end):
        folder_name = chain_file_path + str(i)+'/'
        #print(folder_name)
        try:

            fluxes = np.loadtxt(folder_name + '/fluxes.txt')
            obs_data = read_data(folder_name + '/lensdata.txt')
            observed_fluxes = obs_data[0].m
            params = np.loadtxt(folder_name + '/parameters.txt', skiprows=1)

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
            init = False
        else:
            lens_fluxes = np.vstack((lens_fluxes,fluxes))
            lens_params = np.vstack((lens_params,params))

    observed_fluxes = observed_fluxes[order]

    return lens_fluxes[:,order],observed_fluxes.reshape(1,4),lens_params,params_header


def resample(name, which_lens, parameter_vals_new, SIE_gamma_mean = 2.08,
             SIE_gamma_sigma = 0.05):

    fluxes, observedfluxes, parameters, header = extract_chain_fromprocessed(name, which_lens)

    params_new = copy(parameter_vals_new)
    params_new.update({'SIE_gamma': [np.random.normal(SIE_gamma_mean, SIE_gamma_sigma), 0.02]})
    parameter_names = list(filter(None, header.split(' ')))

    newparams = np.ones_like(parameters)
    sigmas = np.ones_like(parameters)

    for i, pname in enumerate(parameter_names):

        newparams[:,i] = params_new[pname][0]
        sigmas[:,i] = params_new[pname][1]

    delta = np.sum(np.absolute(newparams - parameters) * sigmas ** -1, axis=1)
    index = np.argmin(delta)

    fluxes_obs = fluxes[index, :]

    print(parameters[index])

    return fluxes, fluxes_obs, parameters, header, params_new['SIE_gamma'][0]

def compute_sigma_chains(chain_name, which_lenses, new_chain_name):

    from pyHalo.Cosmology.lens_cosmo import LensCosmo
    l = LensCosmo(0.5, 3)
    zd, zs, _ = np.loadtxt(chainpath_out + '/processed_chains/simulation_zRein.txt', unpack=True)

    if not os.path.exists(chainpath_out + '/processed_chains/'+new_chain_name):
        create_directory(chainpath_out + '/processed_chains/'+new_chain_name)

    chain_info_path = chainpath_out + 'raw_chains/'+chain_name + '/simulation_info.txt'
    copy_directory(chain_info_path, chainpath_out + '/processed_chains/' + new_chain_name)

    for which_lens in which_lenses:
        fluxes, observedfluxes, lens_params, params_header = extract_chain_fromprocessed(chain_name, which_lens)

        pnames = list(filter(None, params_header.split(' ')))

        col = pnames.index('fsub')
        pnames[col] = 'sigma_sub'

        params_header = ''
        for name in pnames:
            params_header += name + ' '

        sigma = l.sigmasub_from_fsub(lens_params[:, col], zd[which_lens - 1], zs[which_lens - 1])
        new_parameters = copy(lens_params)
        new_parameters[:, col] = sigma

        create_directory(chainpath_out + '/processed_chains/'+new_chain_name+'/lens'+str(which_lens))

        f_to_copy = chainpath_out + 'processed_chains/'+chain_name+'/lens'+str(which_lens)+'/modelfluxes.txt'
        loc = chainpath_out + '/processed_chains/'+new_chain_name+'/lens'+str(which_lens)+'/modelfluxes.txt'
        copy_directory(f_to_copy, loc)

        f_to_copy = chainpath_out + '/processed_chains/' + chain_name + '/lens' + str(which_lens) + '/observedfluxes.txt'
        loc = chainpath_out + '/processed_chains/' + new_chain_name + '/lens' + str(which_lens) + '/observedfluxes.txt'
        copy_directory(f_to_copy, loc)
        np.savetxt(chainpath_out + '/processed_chains/'+new_chain_name+'/lens' + str(which_lens) + '/samples.txt', X = new_parameters, header=params_header)

#new_chains_withsigma('WDM_run_7.7_tier2',[1,2],'WDM_run_7.7_sigma')



"""
if False:
    for i in range(1, 11):
        process_chain_i('WDM_run_7.7_tier2', which_lens=i, errors=[0])


if False:
    fsub = 0.007
    logmhm = 4.8
    src_size = 0.035
    LOS_norm = 1
    new_name = 'CDM_run_sigma23_src35'
    which_lens_indexes = np.arange(1, 11)

    resample_chain('WDM_run_7.7_tier2', new_name, which_lens_indexes=which_lens_indexes, errors=[0],
                   parameters_new={'fsub': [fsub, 0.002], 'log_m_break': [logmhm, 0.1],
                                   'source_size_kpc': [src_size, 0.005],
                                   'LOS_normalization': [LOS_norm, 0.05]},
                   SIE_gamma_mean=2.08, SIE_gamma_sigma=0.04, transform_fsub = True)


    fsub = 0.007
    logmhm = 7.5
    src_size = 0.035
    LOS_norm = 1
    new_name = 'WDM_run_sigma23_7.5_src35'
    which_lens_indexes = np.arange(1, 11)

    resample_chain('WDM_run_7.7_tier2', new_name, which_lens_indexes=which_lens_indexes, errors= [0],
    parameters_new={'fsub': [fsub, 0.001], 'log_m_break': [logmhm, 0.1],
                            'source_size_kpc': [src_size, 0.006],
                            'LOS_normalization': [LOS_norm, 0.05]},
    SIE_gamma_mean = 2.08, SIE_gamma_sigma = 0.04)


"""
