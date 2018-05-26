from MagniPy.paths import *
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

def add_flux_perturbations(chain_name='',errors=None,N_pert=1,which_lens = None, parameters=None,fluxes_obs=None,
                           fluxes=None,header=str,tol=None):

    #Ncores, cores_per_lens, Nlenses = read_run_partition(chainpath + '/processed_chains/' + chain_name + '/simulation_info.txt')

    if errors is None:
        errors = []

    if isinstance(errors,int) or isinstance(errors,float):
        errors = [errors]

    errors = [0]+errors

    chain_file_path = chainpath + 'processed_chains/' + chain_name + '/lens' + str(which_lens) + '/'
    perturbed_path = chain_file_path + 'fluxratios/'

    if ~os.path.exists(perturbed_path):
        create_directory(perturbed_path)

    #fluxes_obs = np.loadtxt(chain_file_path + 'observedfluxes.txt')
    #fluxes = np.loadtxt(chain_file_path + 'modelfluxes.txt')
    inds = np.where(np.sum(fluxes,axis=1)==4000)

    np.savetxt(chain_file_path + 'modelfluxes' + '.txt',fluxes, fmt='%.6f')
    np.savetxt(chain_file_path + 'observedfluxes' + '.txt',fluxes_obs, fmt='%.6f')
    np.savetxt(chain_file_path + 'samples.txt',parameters,fmt='%.6f',header=header)

    for error in errors:

        for k in range(1, N_pert + 1):

            perturbed_fname_obs = perturbed_path + 'observed_' + str(int(error * 100)) + 'error_' + str(k) + '.txt'

            if error != 0:
                flux_perturbations_obs = np.random.normal(0, error * fluxes_obs)
            else:
                flux_perturbations_obs = np.zeros_like(fluxes_obs)

            perturbed_obs = fluxes_obs + flux_perturbations_obs

            perturbed_ratios_obs = np.delete(perturbed_obs * perturbed_obs[1] ** -1, 1)

            #np.savetxt(perturbed_fname_obs, perturbed_ratios_obs.reshape(1, 3), fmt='%.6f')

            ############################################################################

            perturbed_fname = perturbed_path + 'model_' + str(int(error * 100)) + 'error_' + str(k) + '.txt'

            if error == 0:

                flux_perturbations = np.zeros_like(fluxes)

            else:
                sigma = float(error) * fluxes
                flux_perturbations = np.random.normal(0, sigma)

            perturbed_fluxes = fluxes + flux_perturbations

            perturbed_ratios = np.delete(perturbed_fluxes * perturbed_fluxes[:, 1, np.newaxis] ** -1, 1, axis=1)

            perturbed_ratios[inds, :] = 1000 * np.ones_like(3)

            #np.savetxt(perturbed_fname, perturbed_ratios, fmt='%.6f')

            summary_statistic = np.sqrt(np.sum((perturbed_ratios-perturbed_ratios_obs)**2,axis=1))

            ordered_inds = np.argsort(summary_statistic)

            np.savetxt(perturbed_path+ 'statistic_' + str(int(error * 100)) + 'error_' + str(k) + '.txt',X=summary_statistic[ordered_inds])
            np.savetxt(perturbed_path + 'params_'+str(int(error * 100)) + 'error_' + str(k) + '.txt',X=parameters[ordered_inds])

            ordered_statistics = summary_statistic[ordered_inds]
            ordered_parameters = parameters[ordered_inds]

            if tol<1:
                return_params = ordered_parameters[0:int(tol*len(summary_statistic))]
            else:
                return_params = ordered_parameters[0:int(tol)]

            create_directory(chainpath + 'processed_chains/' + chain_name + '/lens'+str(which_lens)+'_final/')

            np.savetxt(chainpath + 'processed_chains/' + chain_name + '/lens'+str(which_lens)+'_final/parameters.txt',
                       X=return_params,header=header)

            if error == 0:
                break


def extract_chain(chain_name='',which_lens = None, position_tol = 0.003):

    chain_info_path = chainpath + chain_name + '/simulation_info.txt'

    Ncores,cores_per_lens,Nlens = read_run_partition(chain_info_path)

    chain_file_path = chainpath + chain_name + '/chain'

    params_header = None

    if ~os.path.exists(chainpath+'processed_chains/' + chain_name + '/'):
        create_directory(chainpath+'processed_chains/' + chain_name + '/')

    copy_directory(chain_info_path,chainpath+'processed_chains/' + chain_name + '/')

    if ~os.path.exists(chainpath + 'processed_chains/' + chain_name + '/lens' + str(which_lens) + '/'):
        create_directory(chainpath + 'processed_chains/' + chain_name + '/lens' + str(which_lens) + '/')

    start = int((which_lens-1)*cores_per_lens)
    end = int(start + cores_per_lens)

    for i in range(start,end):

        folder_name = chain_file_path + str(i+1) + '/'
        fluxes = np.loadtxt(folder_name + 'fluxes.txt')
        astrometric_errors = np.loadtxt(folder_name + 'astrometric_errors.txt')

        observed_fluxes = read_data(folder_name + 'lensdata.txt')[0].m
        params = np.loadtxt(folder_name + 'parameters.txt')
        if params_header is None:
            with open(folder_name + 'parameters.txt', 'r') as f:
                lines = f.read().splitlines()
            head = lines[0].split(' ')
            params_header = ''
            for word in head:
                if word not in ['#', '']:
                    params_header += word + ' '

        inds = np.where(astrometric_errors > (2 * position_tol))
        fluxes[inds,:] = 1000*np.ones_like(4)

        if i==start:
            lens_fluxes = fluxes
            lens_params = params
        else:
            lens_fluxes = np.vstack((lens_fluxes,fluxes))
            lens_params = np.vstack((lens_params,params))

    #savename = chainpath + 'processed_chains/' + chain_name + '/' + 'lens' + str(which_lens) + '/'

    inds_to_keep = np.where(lens_fluxes[:,0]!=1000)

    return lens_fluxes[inds_to_keep],observed_fluxes.reshape(1,4),lens_params[inds_to_keep],params_header

    #np.savetxt(savename + 'modelfluxes' + '.txt', lens_fluxes[inds_to_keep], fmt='%.6f')
    #np.savetxt(savename + 'observedfluxes' + '.txt', observed_fluxes.reshape(1, 4), fmt='%.6f')
    #np.savetxt(savename + 'samples.txt', lens_params[inds_to_keep], fmt='%.6f', header=params_header)

def process_chain_i(name=str,which_lens=int,N_pert=1,errors=None):

    fluxes,fluxes_obs,parameters,header = extract_chain(name,which_lens)
    
    add_flux_perturbations(name,errors=errors,N_pert=N_pert,which_lens=which_lens,parameters=parameters,
                           fluxes_obs=np.squeeze(fluxes_obs),fluxes=fluxes,header=header)

#for j in range(2,10):
#    extract_chain('singleplane_test_2',j)
#    add_flux_perturbations('singleplane_test_2',which_lens=j)


