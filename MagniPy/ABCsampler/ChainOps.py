from MagniPy.paths import *
import sys
import numpy as np
from MagniPy.util import *


def read_chain_info(fname):

    with open(fname, 'r') as f:
        lines = f.readlines()

    (Ncores, cores_per_lens) = (lines[1].split(' '))
    Ncores, cores_per_lens = int(Ncores), int(cores_per_lens)

    return Ncores, cores_per_lens, int(Ncores * cores_per_lens ** -1)

def add_flux_perturbations(chain_name='',errors=0.1,N_pert=4):

    Ncores, cores_per_lens, Nlenses = read_chain_info(chainpath + '/processed_chains/'+chain_name + '/simulation_info.txt')

    if isinstance(errors,int) or isinstance(errors,float):
        errors = [errors]

    for n in range(1, int(Nlenses)):

        chain_file_path = chainpath + 'processed_chains/' + chain_name + '/lens'+str(n)+'/'
        perturbed_path = chain_file_path + '/perturbed_fluxes/'

        if ~os.path.exists(perturbed_path):
            create_directory(perturbed_path)

        for error in errors:

            for k in range(1, N_pert + 1):

                perturbed_fname_obs = perturbed_path+'observed_sigma'+ str(int(error * 100)) + '_'+ str(k)+'.txt'
                fluxes_obs = np.loadtxt(chain_file_path + 'observed.txt')
                flux_perturbations_obs = np.random.normal(0, error * fluxes_obs, size=fluxes_obs.shape)
                perturbed_obs = fluxes_obs + flux_perturbations_obs
                np.savetxt(perturbed_fname_obs, perturbed_obs.reshape(1,3), fmt='%.6f')

                perturbed_fname = perturbed_path+'model_sigma'+ str(int(error * 100)) + '_'+str(k)+'.txt'
                fluxes = np.loadtxt(chain_file_path + 'fluxratios.txt')
                flux_perturbations = np.random.normal(0, error * fluxes, size=fluxes.shape)
                perturbed_fluxes = fluxes + flux_perturbations
                np.savetxt(perturbed_fname,perturbed_fluxes,fmt='%.6f')

def extract_chain(chain_name=''):

    chain_info_path = chainpath + chain_name + '/simulation_info.txt'

    Ncores,cores_per_lens,Nlens = read_chain_info(chain_info_path)

    chain_file_path = chainpath + chain_name + '/chain'

    counter = 1

    if ~os.path.exists(chainpath+'processed_chains/' + chain_name + '/'):
        create_directory(chainpath+'processed_chains/' + chain_name + '/')

    copy_directory(chain_info_path,chainpath+'processed_chains/' + chain_name + '/')

    for i in range(0,Nlens):

        single_lens_fluxratios = None
        single_lens_params = None

        if ~os.path.exists(chainpath + 'processed_chains/' + chain_name + '/lens'+str(i+1)+'/'):
            create_directory(chainpath + 'processed_chains/' + chain_name + '/lens'+str(i+1)+'/')

        for j in range(0,cores_per_lens):

            folder_name = chain_file_path+str(counter)+'/'

            fluxes = np.loadtxt(folder_name+'/fluxes.txt')

            fluxratios = fluxes*(fluxes[:,1,np.newaxis])**-1
            print fluxratios
            fluxratios = np.delete(fluxratios,1,axis=0)
            print fluxratios
            a=input('continue')
            params = np.loadtxt(folder_name+'parameters.txt')
            observed = fluxratios[0,:]
            fluxratios = np.delete(fluxratios,0,axis=0)
            counter += 1

            try:
                single_lens_fluxratios = np.vstack((single_lens_fluxratios,fluxratios))
            except:
                single_lens_fluxratios = fluxratios

            try:
                single_lens_params = np.vstack((single_lens_params,params))
            except:
                single_lens_params = params

        savename = chainpath+'processed_chains/' + chain_name + '/' + 'lens'+str(i+1)+'/'

        np.savetxt(savename+'fluxratios'+'.txt', single_lens_fluxratios,fmt='%.6f')
        np.savetxt(savename+'observed'+'.txt', observed.reshape(1,3),fmt='%.6f')

#extract_chain('gamma_test_208_new')




