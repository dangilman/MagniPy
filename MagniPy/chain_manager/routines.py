import numpy as np
from MagniPy.paths import *
from MagniPy.util import create_directory, copy_directory, read_data

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
    for i in range(start_idx+1, end+1):
        folder_name = chain_file_path + str(i)+'/'
        #print(folder_name)
        try:

            fluxes = np.loadtxt(folder_name + '/fluxes.txt')
            obs_data = read_data(folder_name + '/lensdata.txt')
            observed_fluxes = obs_data[0].m
            params = np.loadtxt(folder_name + '/parameters.txt', skiprows=1)
            macro_model = np.loadtxt(folder_name + '/macro.txt')

            macro_model[:,3] = transform_ellip(macro_model[:,3])

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

            lens_all = np.column_stack((macro_model, params))

            init = False
        else:
            lens_fluxes = np.vstack((lens_fluxes,fluxes))
            lens_params = np.vstack((lens_params,params))
            new = np.column_stack((macro_model, params))

            lens_all = np.vstack((lens_all, new))

    observed_fluxes = observed_fluxes[order]

    return lens_fluxes[:,order],observed_fluxes.reshape(1,4),lens_params,lens_all,params_header

def process_raw(name, sim_name='grism_quads'):

    """
    coverts output from cluster into single files for each lens
    """

    fluxes,fluxes_obs,parameters,all,header = extract_chain(name, sim_name)

    all = np.squeeze(all)
    fluxes, fluxes_obs = np.squeeze(fluxes), np.squeeze(fluxes_obs)
    chain_file_path = chainpath_out + 'processed_chains/grism_quads/' + name + '/'

    if ~os.path.exists(chain_file_path):
        create_directory(chain_file_path)

    np.savetxt(chain_file_path + 'modelfluxes' + '.txt', fluxes, fmt='%.6f')
    np.savetxt(chain_file_path + 'observedfluxes' + '.txt', fluxes_obs, fmt='%.6f')
    np.savetxt(chain_file_path + 'samples.txt', parameters, fmt='%.5f', header=header)
    np.savetxt(chain_file_path + 'all_samples.txt', all, fmt='%.5f')

process_raw('lens1422')
