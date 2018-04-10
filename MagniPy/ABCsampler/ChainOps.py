from MagniPy.paths import *
import sys
import numpy as np
from MagniPy.util import *
import ast

def read_chain_info(fname):

    with open(fname,'r') as f:
        lines = f.read().splitlines()

    params_varied = []
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

    return params_varied,truth_dic


def read_run_partition(fname):

    with open(fname, 'r') as f:
        lines = f.readlines()

    Ncores = int(lines[1])
    cores_per_lens = int(lines[4])

    return Ncores, cores_per_lens, int(Ncores * cores_per_lens ** -1)

def add_flux_perturbations(chain_name='',errors=None,N_pert=1):

    Ncores, cores_per_lens, Nlenses = read_run_partition(chainpath + '/processed_chains/' + chain_name + '/simulation_info.txt')

    if errors is None:
        errors = []

    if isinstance(errors,int) or isinstance(errors,float):
        errors = [errors]

    errors = [0]+errors

    for n in range(1, int(Nlenses)):

        chain_file_path = chainpath + 'processed_chains/' + chain_name + 'lens'+str(n)+'/'
        perturbed_path = chain_file_path + '/fluxratios/'

        if ~os.path.exists(perturbed_path):
            create_directory(perturbed_path)

        for error in errors:

            for k in range(1, N_pert + 1):

                perturbed_fname_obs = perturbed_path+'observed_'+ str(int(error * 100)) + 'error_'+ str(k)+'.txt'
                fluxes_obs = np.loadtxt(chain_file_path + 'observedfluxes.txt')

                if error!=0:
                    flux_perturbations_obs = np.random.normal(0, error * fluxes_obs, size=fluxes_obs.shape)
                else:
                    flux_perturbations_obs = np.zeros_like(fluxes_obs)

                perturbed_obs = fluxes_obs + flux_perturbations_obs

                perturbed_ratios_obs = np.delete(perturbed_obs*perturbed_obs[1]**-1,1)
                np.savetxt(perturbed_fname_obs, perturbed_ratios_obs.reshape(1,3), fmt='%.6f')

                perturbed_fname = perturbed_path+'model_'+ str(int(error * 100)) + 'error_'+str(k)+'.txt'
                fluxes = np.loadtxt(chain_file_path + 'modelfluxes.txt')

                if error!=0:
                    flux_perturbations = np.zeros_like(fluxes.shape)
                else:
                    flux_perturbations = np.random.normal(0, error * fluxes, size=fluxes.shape)

                perturbed_fluxes = fluxes + flux_perturbations

                perturbed_ratios = np.delete(perturbed_fluxes*perturbed_fluxes[:,1,np.newaxis]**-1,1,axis=1)
                np.savetxt(perturbed_fname,perturbed_ratios,fmt='%.6f')

                if error==0:
                    break

def extract_chain(chain_name=''):

    chain_info_path = chainpath + chain_name + '/simulation_info.txt'

    Ncores,cores_per_lens,Nlens = read_run_partition(chain_info_path)

    chain_file_path = chainpath + chain_name + '/chain'

    params_header = None

    counter = 1

    if ~os.path.exists(chainpath+'processed_chains/' + chain_name + '/'):
        create_directory(chainpath+'processed_chains/' + chain_name + '/')

    copy_directory(chain_info_path,chainpath+'processed_chains/' + chain_name + '/')

    for i in range(0,Nlens):

        single_lens_fluxes = None
        single_lens_params = None

        if ~os.path.exists(chainpath + 'processed_chains/' + chain_name + '/lens'+str(i+1)+'/'):
            create_directory(chainpath + 'processed_chains/' + chain_name + '/lens'+str(i+1)+'/')

        for j in range(0,cores_per_lens):

            folder_name = chain_file_path+str(counter)+'/'

            inds_to_keep = []

            #fluxes = np.loadtxt(folder_name+'fluxes.txt')

            with open(folder_name+'chain.txt') as f:
                lines=f.readlines()

            for count,line in enumerate(lines):
                line = line.split(' ')

                if int(line[0])!=4:
                    continue

                inds_to_keep.append(count)

                try:
                    fluxes = np.vstack((fluxes,np.array([float(line[5]),float(line[9]),float(line[13]),float(line[17])])))
                except:
                    fluxes = np.array([float(line[5]),float(line[9]),float(line[13]),float(line[17])])

            observed_fluxes = fluxes[0,:]

            fluxes = np.delete(fluxes,0,axis=0)

            params = np.loadtxt(folder_name+'parameters.txt')
            if params_header is None:
                with open(folder_name+'parameters.txt','r') as f:
                    lines = f.readlines()
                params_header = lines[0]

            params = params[inds_to_keep,:]

            try:
                single_lens_fluxes = np.vstack((single_lens_fluxes,fluxes))
            except:
                single_lens_fluxes = fluxes

            try:
                single_lens_params = np.vstack((single_lens_params,params))
            except:
                single_lens_params = params

            counter += 1

        savename = chainpath+'processed_chains/' + chain_name + '/' + 'lens'+str(i+1)+'/'

        np.savetxt(savename+'modelfluxes'+'.txt', single_lens_fluxes,fmt='%.6f')
        np.savetxt(savename+'observedfluxes'+'.txt', observed_fluxes.reshape(1,4),fmt='%.6f')
        np.savetxt(savename+'samples.txt',single_lens_params,fmt='%.6f',header=params_header)

#extract_chain('gamma_test_208_new')
#add_flux_perturbations('gamma_test_208_new')

print read_chain_info(chainpath + '/LOS_test' + '/simulation_info.txt')





