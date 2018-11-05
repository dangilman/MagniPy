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

def add_flux_perturbations(chain_name='',errors=None,N_pert=1,which_lens = None, parameters=None,fluxes_obs=None,
                           fluxes=None,header=str,stat='quad'):

    #Ncores, cores_per_lens, Nlenses = read_run_partition(chainpath + '/processed_chains/' + chain_name + '/simulation_info.txt')

    flux_ratio_index = 1

    if errors is None:
        errors = []

    if isinstance(errors,int) or isinstance(errors,float):
        errors = [errors]

    errors = [0]+errors

    fluxes, fluxes_obs = np.squeeze(fluxes), np.squeeze(fluxes_obs)

    chain_file_path = chainpath_out + 'processed_chains/' + chain_name + '/lens' + str(which_lens) + '/'
    perturbed_path = chain_file_path + 'fluxratios/'

    if ~os.path.exists(perturbed_path):
        create_directory(perturbed_path)

    if ~os.path.exists(chain_file_path):
        create_directory(chain_file_path)

    if ~os.path.exists(perturbed_path):
        create_directory(perturbed_path)

    np.savetxt(chain_file_path + 'modelfluxes' + '.txt', fluxes, fmt='%.6f')
    np.savetxt(chain_file_path + 'observedfluxes' + '.txt',fluxes_obs, fmt='%.6f')
    np.savetxt(chain_file_path + 'samples.txt',parameters,fmt='%.6f',header=header)

    for error in errors:

        for k in range(1, N_pert + 1):

            if error != 0:
                flux_perturbations_obs = np.random.normal(0, error * fluxes_obs)
            else:
                flux_perturbations_obs = np.zeros_like(fluxes_obs)

            perturbed_obs = fluxes_obs + flux_perturbations_obs

            perturbed_ratios_obs = perturbed_obs*perturbed_obs[flux_ratio_index]**-1

            #np.savetxt(perturbed_fname_obs, perturbed_ratios_obs.reshape(1, 3), fmt='%.6f')

            ############################################################################

            perturbed_fname = perturbed_path + 'model_' + str(int(error * 100)) + 'error_' + str(k) + '.txt'

            if error == 0:

                flux_perturbations = np.zeros_like(fluxes)

            else:
                sigma = float(error) * fluxes
                flux_perturbations = np.random.normal(0, sigma)

            perturbed_fluxes = fluxes + flux_perturbations

            for i in range(0,int(np.shape(perturbed_fluxes)[0])):
                perturbed_fluxes[i,:] = perturbed_fluxes[i,:]*perturbed_fluxes[i,flux_ratio_index]**-1

            if stat == 'quad':
                summary_statistic = np.sqrt(np.sum((perturbed_fluxes-perturbed_ratios_obs)**2,axis=1))
            #elif stat == 'R':
            #    summary_statistic = R(perturbed_fluxes,perturbed_obs,config=lens_config)

            ordered_inds = np.argsort(summary_statistic)

            np.savetxt(perturbed_path+ 'statistic_' + str(int(error * 100)) + 'error_' + str(k) + '.txt',X=summary_statistic[ordered_inds])
            np.savetxt(perturbed_path + 'params_'+str(int(error * 100)) + 'error_' + str(k) + '.txt',X=parameters[ordered_inds])

            #ordered_parameters = parameters[ordered_inds]

            #if tol<1:
            #    return_params = ordered_parameters[0:int(tol*len(summary_statistic))]
            #else:
            #    return_params = ordered_parameters[0:int(tol)]

            #create_directory(chainpath + 'processed_chains/' + chain_name + '/lens'+str(which_lens)+'_final/')

            #np.savetxt(chainpath + 'processed_chains/' + chain_name + '/lens'+str(which_lens)+'_final/parameters.txt',
            #           X=return_params,header=header)

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

    return fluxes, parameters, params_header

def extract_chain(chain_name='',which_lens = None, position_tol = 0.003):

    chain_info_path = chainpath_out + chain_name + '/simulation_info.txt'

    Ncores,cores_per_lens,Nlens = read_run_partition(chain_info_path)

    #lens_config, lens_R_index = read_R_index(chainpath_out+chain_name+'/R_index_config.txt',0)

    chain_file_path = chainpath_out + chain_name +'/chain'

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
            #astrometric_errors = np.loadtxt(folder_name + 'astrometric_errors.txt')

            obs_data = read_data(folder_name + '/lensdata.txt')
            observed_fluxes = obs_data[0].m

            observed_pos_x = obs_data[0].x
            observed_pos_y = obs_data[0].y

            if order is None:
                lens_config = [0,1,2,3]
                #continue
                #if lens_config == 'cross':

                #    order = [0,1,2,3]


                #else:

                #    reference_x,reference_y = observed_pos_x[lens_R_index],observed_pos_y[lens_R_index]
                #    order = find_closest_xy(observed_pos_x,observed_pos_y,reference_x,reference_y)

            params = np.loadtxt(folder_name + '/parameters.txt', skiprows=1)

        except:
            print('didnt find a file...')
            continue

        if params_header is None:
            with open(folder_name + '/parameters.txt', 'r') as f:
                lines = f.read().splitlines()
            head = lines[0].split(' ')
            params_header = ''
            for word in head:
                if word not in ['#', '']:
                    params_header += word + ' '

        #inds = np.where(astrometric_errors > (2 * position_tol))
        #fluxes[inds,:] = 1000*np.ones_like(4)

        if init:

            lens_fluxes = fluxes
            lens_params = params
            init = False
        else:
            lens_fluxes = np.vstack((lens_fluxes,fluxes))
            lens_params = np.vstack((lens_params,params))

    observed_fluxes = observed_fluxes[order]

    return lens_fluxes[:,order],observed_fluxes.reshape(1,4),lens_params,params_header

    #np.savetxt(savename + 'modelfluxes' + '.txt', lens_fluxes[inds_to_keep], fmt='%.6f')
    #np.savetxt(savename + 'observedfluxes' + '.txt', observed_fluxes.reshape(1, 4), fmt='%.6f')
    #np.savetxt(savename + 'samples.txt', lens_params[inds_to_keep], fmt='%.6f', header=params_header)

def process_chain_i(name=str,which_lens=int,N_pert=1,errors=None):

    fluxes,fluxes_obs,parameters,header = extract_chain(name,which_lens)

    add_flux_perturbations(name,errors=errors,N_pert=N_pert,which_lens=which_lens,parameters=parameters,
                           fluxes_obs=np.squeeze(fluxes_obs),fluxes=fluxes,header=header)

def resample_chain(name=str, new_name = str, which_lens_indexes=int, N_pert=1, errors = None, parameters_new={}, SIE_gamma_mean = 2.08,
                   SIE_gamma_sigma = 0.05):

    new_gamma = []
    if not os.path.exists(chainpath_out + 'processed_chains/'+new_name):
        create_directory(chainpath_out + 'processed_chains/'+new_name)

    for which_lens in which_lens_indexes:

        fluxes, fluxes_obs, parameters, header, newgamma = resample(name, which_lens, parameters_new, SIE_gamma_mean = SIE_gamma_mean,
                                                                       SIE_gamma_sigma = SIE_gamma_sigma)

        new_gamma.append(newgamma)
        add_flux_perturbations(new_name, errors=errors, N_pert=N_pert,which_lens=which_lens,parameters=parameters,
                               fluxes_obs=np.squeeze(fluxes_obs),fluxes=fluxes,header=header)

    chain_info_path = chainpath_out + name + '/simulation_info.txt'
    with open(chain_info_path, 'r') as f:
        lines = f.readlines()

    with open(chainpath_out + 'processed_chains/'+new_name + '/simulation_info.txt', 'w') as f:
        for line in lines:
            f.write(line)
            if line == '# truths\n':
                break

        for pname in parameters_new.keys():
            print(pname)
            f.write(pname + ' '+str(parameters_new[pname][0])+'\n')

        f.write('SIE_gamma '+str(SIE_gamma_mean)+'\n')

    with open(chainpath_out + 'processed_chains/'+new_name + '/gamma_values.txt', 'w') as f:

        f.write('re-sampled gamma '+str(SIE_gamma_mean)+' +\- '+str(SIE_gamma_sigma)+'\n')
        for g in new_gamma:
            f.write(str(g)+' ')

def resample(name, which_lens, parameter_vals_new, SIE_gamma_mean = 2.08,
             SIE_gamma_sigma = 0.05):

    #fluxes, _, parameters, header, lens_config = extract_chain(name, which_lens, from_processe=True)

    fluxes, parameters, header = extract_chain_fromprocessed(name, which_lens)

    params_new = copy(parameter_vals_new)
    params_new.update({'SIE_gamma': [np.random.normal(SIE_gamma_mean, SIE_gamma_sigma), 0.02]})
    parameter_names = list(filter(None, header.split(' ')))

    N = len(parameters)

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

if False:
    for i in range(1, 11):
        process_chain_i('WDM_run_7.7_tier2', which_lens=i, errors=[0])

if False:
    fsub = 0.013
    logmhm = 4.8
    src_size = 0.035
    LOS_norm = 1
    new_name = 'CDM_run_fsub0.013_src35'
    which_lens_indexes = np.arange(1, 11)

    resample_chain('WDM_run_7.7_tier2', new_name, which_lens_indexes=which_lens_indexes, errors=[0],
                   parameters_new={'fsub': [fsub, 0.002], 'log_m_break': [logmhm, 0.1],
                                   'source_size_kpc': [src_size, 0.005],
                                   'LOS_normalization': [LOS_norm, 0.05]},
                   SIE_gamma_mean=2.08, SIE_gamma_sigma=0.04)


    fsub = 0.013
    logmhm = 7.2
    src_size = 0.035
    LOS_norm = 1
    new_name = 'WDM_run_7.2_src35'
    which_lens_indexes = np.arange(1, 11)

    resample_chain('WDM_run_7.7_tier2', new_name, which_lens_indexes=which_lens_indexes, errors= [0],
    parameters_new={'fsub': [fsub, 0.001], 'log_m_break': [logmhm, 0.1],
                            'source_size_kpc': [src_size, 0.006],
                            'LOS_normalization': [LOS_norm, 0.05]},
    SIE_gamma_mean = 2.08, SIE_gamma_sigma = 0.04)

