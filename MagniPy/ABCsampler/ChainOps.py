from MagniPy.paths import *
from MagniPy.Analysis.Statistics.singledensity import SingleDensity
import sys
from copy import deepcopy, copy
import numpy as np
from MagniPy.util import *
import pandas

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
                           fluxes=None,header=str, counter_start = int):

    fluxes, fluxes_obs = np.squeeze(fluxes), np.squeeze(fluxes_obs)
    lens_idx = which_lens + counter_start
    chain_file_path = chainpath_out + 'processed_chains/' + chain_name + '/lens' + str(lens_idx) + '/'

    if ~os.path.exists(chain_file_path):
        create_directory(chain_file_path)

    np.savetxt(chain_file_path + 'modelfluxes' + '.txt', fluxes, fmt='%.6f')
    np.savetxt(chain_file_path + 'observedfluxes' + '.txt',fluxes_obs, fmt='%.6f')
    np.savetxt(chain_file_path + 'samples.txt',parameters,fmt='%.5f',header=header)

def add_flux_perturbations(name, which_lens, parameters, fluxes_obs, fluxes, errors = None, N_pert = 1,
                           keep_n=15000):

    if errors is None:
        errors = []

    if isinstance(errors,int) or isinstance(errors,float):
        errors = [errors]

    for error in errors:

        for k in range(1, N_pert + 1):

            perturbed_path = chainpath_out + 'chain_stats/' + name + '/lens' + str(which_lens) + '/'

            if not os.path.exists(chainpath_out + 'chain_stats/' + name):
                create_directory(chainpath_out + 'chain_stats/' + name)

            if not os.path.exists(perturbed_path):
                create_directory(chainpath_out + 'chain_stats/' + name + '/lens' + str(which_lens) + '/')

            if error == 0:

                perturbed_ratios_obs = fluxes_obs[1:]
                perturbed_ratios = fluxes[:,1:]

            else:
                flux_perturbations_obs = np.random.normal(0, float(error)*fluxes_obs)
                flux_perturbations = np.random.normal(0, float(error)*fluxes)

                perturbed_fluxes_obs = fluxes_obs + flux_perturbations_obs

                perturbed_fluxes = fluxes + flux_perturbations

                perturbed_ratios_obs = perturbed_fluxes_obs[1:]*perturbed_fluxes_obs[0]**-1

                norm = deepcopy(perturbed_fluxes[:,0])

                for col in range(0,4):

                    perturbed_fluxes[:,col] *= norm ** -1

                perturbed_ratios = perturbed_fluxes[:,1:]
                #perturbed_ratios = perturbed_fluxes

            diff = np.array((perturbed_ratios - perturbed_ratios_obs)**2)
            summary_statistic = np.sqrt(np.sum(diff, 1))

            #print('warning: not sorting summary statistics')
            #ordered_inds = np.arange(0,keep_n)
            ordered_inds = np.argsort(summary_statistic)[0:keep_n]
            print('lens # ', which_lens)
            print('N < 0.01: ', np.sum(summary_statistic < 0.01))
            print('N < 0.02: ', np.sum(summary_statistic < 0.02))
            print('N < 0.03: ', np.sum(summary_statistic < 0.03))
            np.savetxt(perturbed_path + 'statistic_' + str(int(error * 100)) + 'error_' + str(k) + '.txt',
                       X=summary_statistic[ordered_inds], fmt=('%.4f'))
            if parameters.shape[1] == 5:
                np.savetxt(perturbed_path + 'params_'+str(int(error * 100)) + 'error_' + str(k) + '.txt',X=parameters[ordered_inds,:],
                       fmt=('%.3f', '%.4f', '%.4f', '%.4f', '%.3f'))
            elif parameters.shape[1] == 3:
                np.savetxt(perturbed_path + 'params_' + str(int(error * 100)) + 'error_' + str(k) + '.txt',
                           X=parameters[ordered_inds, :],
                           fmt=('%.5f', '%.5f', '%.5f'))
            elif parameters.shape[1] == 2:
                np.savetxt(perturbed_path + 'params_' + str(int(error * 100)) + 'error_' + str(k) + '.txt',
                           X=parameters[ordered_inds, :],
                           fmt=('%.5f', '%.5f'))
            if error == 0:
                break

def extract_chain_fromprocessed(chain_name = '', which_lens = None):

    route = chainpath_out + 'processed_chains/' + chain_name + '/lens'+str(which_lens)+'/'

    #lens_config, _ = read_R_index(chainpath_out + chain_name + '/R_index_config.txt', 0)

    #fluxes = np.loadtxt(route+'modelfluxes.txt')
    fluxes = np.squeeze(pandas.read_csv(route+'modelfluxes.txt',header=None,
                                        sep=" ", index_col=None)).values

    with open(route + '/samples.txt', 'r') as f:
        lines = f.read().splitlines()
    head = lines[0].split(' ')
    params_header = ''

    for word in head:
        if word not in ['#', '']:
            params_header += word + ' '

    parameters = np.loadtxt(route + '/samples.txt')
    assert np.shape(parameters)[0] == np.shape(fluxes)[0]
    observed_fluxes = np.squeeze(np.loadtxt(route + 'observedfluxes.txt'))

    return fluxes, observed_fluxes, parameters, params_header

def extract_chain(chain_name='',which_lens = None):

    chain_info_path = chainpath_out + 'raw_chains_sidm/' + chain_name + '/simulation_info.txt'
    Ncores, cores_per_lens, Nlens = read_run_partition(chain_info_path)

    #lens_config, lens_R_index = read_R_index(chainpath_out+chain_name+'/R_index_config.txt',0)

    chain_file_path = chainpath_out + 'raw_chains_sidm/' + chain_name +'/chain'

    params_header = None
    order = None

    if ~os.path.exists(chainpath_out+'processed_chains/' + chain_name + '/'):
        create_directory(chainpath_out+'processed_chains/' + chain_name + '/')

    copy_directory(chain_info_path,chainpath_out+'processed_chains/' + chain_name + '/')

    if ~os.path.exists(chainpath_out + 'processed_chains/' + chain_name + '/lens' + str(which_lens) + '/'):
        create_directory(chainpath_out + 'processed_chains/' + chain_name + '/lens' + str(which_lens) + '/')

    start = int((which_lens-1)*cores_per_lens)
    end = int(start + cores_per_lens)
    print(start, end)
    init = True
    #for i in range(start,end):
    for i in range(start+1, end+1):
        folder_name = chain_file_path + str(i)+'/'
        #print(folder_name)
        try:

            fluxes = np.loadtxt(folder_name + '/fluxes.txt')
            obs_data = read_data(folder_name + '/lensdata.txt')
            observed_fluxes = obs_data[0].m
            params = np.loadtxt(folder_name + '/parameters.txt', skiprows=1)

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

    return fluxes, fluxes_obs, parameters, header, parameters[index][1]

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
