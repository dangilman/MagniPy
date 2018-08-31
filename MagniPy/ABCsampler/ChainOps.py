from MagniPy.paths import *
from MagniPy.Analysis.Statistics.summary_statistics import *
import sys
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
                           fluxes=None,header=str,tol=None,stat='quad',lens_config=None):

    #Ncores, cores_per_lens, Nlenses = read_run_partition(chainpath + '/processed_chains/' + chain_name + '/simulation_info.txt')

    flux_ratio_index = 1

    if errors is None:
        errors = []

    if isinstance(errors,int) or isinstance(errors,float):
        errors = [errors]

    errors = [0]+errors

    chain_file_path = chainpath_out + 'processed_chains/' + chain_name + '/lens' + str(which_lens) + '/'
    perturbed_path = chain_file_path + 'fluxratios/'

    if ~os.path.exists(perturbed_path):
        create_directory(perturbed_path)

    np.savetxt(chain_file_path + 'modelfluxes' + '.txt',fluxes, fmt='%.6f')
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
            elif stat == 'R':
                summary_statistic = R(perturbed_fluxes,perturbed_obs,config=lens_config)

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
                
def extract_chain(chain_name='',which_lens = None, position_tol = 0.003):

    chain_info_path = chainpath_out + chain_name + '/simulation_info.txt'

    Ncores,cores_per_lens,Nlens = read_run_partition(chain_info_path,'R_index_config.txt')

    lens_config, lens_R_index = read_R_index(chainpath_out+chain_name+'/R_index_config.txt',0)

    chain_file_path = chainpath_out + chain_name + '/chain'

    params_header = None
    order = None

    if ~os.path.exists(chainpath_out+'processed_chains/' + chain_name + '/'):
        create_directory(chainpath_out+'processed_chains/' + chain_name + '/')

    copy_directory(chain_info_path,chainpath_out+'processed_chains/' + chain_name + '/')

    if ~os.path.exists(chainpath_out + 'processed_chains/' + chain_name + '/lens' + str(which_lens) + '/'):
        create_directory(chainpath_out + 'processed_chains/' + chain_name + '/lens' + str(which_lens) + '/')

    start = int((which_lens-1)*cores_per_lens)
    end = int(start + cores_per_lens)

    for i in range(start,end):

        folder_name = chain_file_path + str(i+1) + '/'

        try:

            fluxes = np.loadtxt(folder_name + 'fluxes.txt')
            #astrometric_errors = np.loadtxt(folder_name + 'astrometric_errors.txt')

            obs_data = read_data(folder_name + 'lensdata.txt')
            observed_fluxes = obs_data[0].m

            observed_pos_x = obs_data[0].x
            observed_pos_y = obs_data[0].y

            if order is None:

                if lens_config == 'cross':

                    order = [0,1,2,3]


                else:

                    reference_x,reference_y = observed_pos_x[lens_R_index],observed_pos_y[lens_R_index]
                    order = find_closest_xy(observed_pos_x,observed_pos_y,reference_x,reference_y)

            params = np.loadtxt(folder_name + 'parameters.txt', skiprows=1)

        except:
            #print('didnt find a file...')
            continue

        if params_header is None:
            with open(folder_name + 'parameters.txt', 'r') as f:
                lines = f.read().splitlines()
            head = lines[0].split(' ')
            params_header = ''
            for word in head:
                if word not in ['#', '']:
                    params_header += word + ' '

        #inds = np.where(astrometric_errors > (2 * position_tol))
        #fluxes[inds,:] = 1000*np.ones_like(4)

        if i==start:
            lens_fluxes = fluxes
            lens_params = params
        else:
            lens_fluxes = np.vstack((lens_fluxes,fluxes))
            lens_params = np.vstack((lens_params,params))

    observed_fluxes = observed_fluxes[order]

    return lens_fluxes[:,order],observed_fluxes.reshape(1,4),lens_params,params_header,lens_config

    #np.savetxt(savename + 'modelfluxes' + '.txt', lens_fluxes[inds_to_keep], fmt='%.6f')
    #np.savetxt(savename + 'observedfluxes' + '.txt', observed_fluxes.reshape(1, 4), fmt='%.6f')
    #np.savetxt(savename + 'samples.txt', lens_params[inds_to_keep], fmt='%.6f', header=params_header)

def process_chain_i(name=str,which_lens=int,N_pert=1,errors=None,tol=None):

    fluxes,fluxes_obs,parameters,header,lens_config = extract_chain(name,which_lens)
    
    add_flux_perturbations(name,errors=errors,N_pert=N_pert,which_lens=which_lens,parameters=parameters,
                           fluxes_obs=np.squeeze(fluxes_obs),fluxes=fluxes,header=header,tol=tol,lens_config=lens_config)

#process_chain_i('he0435_LOS', which_lens=1, tol = 1000)
