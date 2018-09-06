from MagniPy.Solver.solveroutines import SolveRoutines
from MagniPy.ABCsampler.sampler_routines import *
from copy import deepcopy,copy
from MagniPy.paths import *
from pyHalo.pyhalo import pyHalo

def initialize_macro(solver,data,init):

    _, model = solver.optimize_4imgs_lenstronomy(macromodel=init, datatofit=data, multiplane=False,
                                                 source_shape='GAUSSIAN', source_size_kpc=0.01,
                                                 tol_source=1e-5, tol_mag=0.2, tol_centroid=0.05,
                                                 centroid_0=[0, 0], n_particles=60, n_iterations=700,
                                                 polar_grid=True, optimize_routine='fixed_powerlaw_shear', verbose=False,
                                                 re_optimize=False, particle_swarm=True, restart=4)

    return model

def run_lenstronomy(data, prior, keys, keys_to_vary, macromodel_init, halo_constructor, solver):

    chaindata = []
    parameters = []
    start = True
    N_computed = 0

    while N_computed < keys['Nsamples']:

        samples = prior.sample(scale_by='Nsamples')[0]

        chain_keys_run = copy(keys)

        for i,pname in enumerate(keys_to_vary.keys()):

            chain_keys_run[pname] = samples[i]

        macromodel = deepcopy(macromodel_init[0])

        if 'SIE_gamma' in keys_to_vary:
            macromodel.lens_components[0].update_lenstronomy_args({'gamma': chain_keys_run['SIE_gamma']})

        halo_args = halo_model_args(chain_keys_run)

        d2fit = perturb_data(data,chain_keys_run['position_sigma'],chain_keys_run['flux_sigma'])

        while True:

            halos = halo_constructor.render(chain_keys_run['mass_func_type'], halo_args, nrealizations=1)

            halos[0] = halos[0].filter(data.x, data.y, mindis_front=chain_keys_run['mindis_front'],
                                       mindis_back=chain_keys_run['mindis_back'],
                                       logmasscut_back=chain_keys_run['log_masscut_back'],
                                       logmasscut_front=chain_keys_run['log_masscut_front'])

            if chain_keys_run['multiplane']:
                #print(samples)
                new, _ = solver.optimize_4imgs_lenstronomy(macromodel=macromodel.lens_components[0], realizations=halos,
                                                           datatofit=d2fit, multiplane=chain_keys_run['multiplane'],
                                                           source_size_kpc=chain_keys_run['source_size_kpc'],
                                                           restart=2, n_particles=50, n_iterations=350,
                                                           particle_swarm=True, re_optimize=True, verbose=True, polar_grid=False,
                                                           single_background=chain_keys_run['single_background'])
            else:
                new, _ = solver.optimize_4imgs_lenstronomy(macromodel=macromodel.lens_components[0], realizations=halos,
                                                           datatofit=d2fit, multiplane=chain_keys_run['multiplane'],
                                                           source_size_kpc=chain_keys_run['source_size_kpc'],
                                                           restart=2, particle_swarm=True, re_optimize=True, verbose=False,
                                                           polar_grid=False)

            if chi_square_img(d2fit.x,d2fit.y,new[0].x,new[0].y,0.003) < 2:
                break

        N_computed += 1
        print('completed ' + str(N_computed) + ' of '+str(keys['Nsamples'])+'...')

        samples_array = []

        for pname in keys_to_vary.keys():
            samples_array.append(chain_keys_run[pname])

        if start:
            chaindata = new[0].m
            parameters = np.array(samples_array)
        else:
            chaindata = np.vstack((chaindata,new[0].m))
            parameters = np.vstack((parameters,np.array(samples_array)))

        start = False

    return chaindata, parameters

def runABC(chain_ID='',core_index=int):

    chain_keys,chain_keys_to_vary,output_path,run = initialize(chain_ID,core_index)

    if run is False:
        return

    # Initialize data, macormodel
    datatofit = Data(x=chain_keys['x_to_fit'], y=chain_keys['y_to_fit'], m=chain_keys['flux_to_fit'],
                     t=chain_keys['t_to_fit'], source=chain_keys['source'])

    write_data(output_path + 'lensdata.txt',[datatofit], mode='write')

    if run is False:
        return

    solver = SolveRoutines(zlens=chain_keys['zlens'], zsrc=chain_keys['zsrc'],
                           temp_folder=chain_keys['scratch_file'], clean_up=True)
    _macro = get_default_SIE(z=chain_keys['zlens'])

    macromodel = initialize_macro(solver, datatofit, _macro)

    macromodel[0].lens_components[0].set_varyflags(chain_keys['varyflags'])

    param_names_tovary = chain_keys_to_vary.keys()
    write_info_file(chainpath + chain_keys['output_folder'] + 'simulation_info.txt',
                    chain_keys, chain_keys_to_vary, param_names_tovary)
    copy_directory(chain_ID + '/R_index_config.txt', chainpath + chain_keys['output_folder'])

    prior = ParamSample(params_to_vary=chain_keys_to_vary, Nsamples=1, macromodel=macromodel)

    constructor = pyHalo(chain_keys['zlens'], chain_keys['zsrc'])

    if chain_keys['solve_method'] == 'lensmodel':

        raise Exception('not yet implemented.')

    else:

        fluxes,parameters = run_lenstronomy(datatofit, prior, chain_keys, chain_keys_to_vary, macromodel, constructor, solver)

    header_string = ''
    for name in param_names_tovary:
        header_string += name + ' '

    write_fluxes(output_path+'fluxes.txt',fluxes=fluxes,summed_in_quad=False)

    write_params(parameters,output_path + 'parameters.txt',header_string)

def write_params(params,fname,header):

    with open(fname, 'a') as f:

        f.write(header+'\n')

        if np.shape(params)[0] == 1:
                for p in range(0,len(params)):
                    f.write(str(float(params[p])+' '))
        else:

            for r in  range(0,np.shape(params)[0]):
                row = params[r,:]
                for p in range(0,len(row)):
                    f.write(str(float(row[p]))+' ')
                f.write('\n')


def write_info_file(fpath,keys,keys_to_vary,pnames_vary):

    with open(fpath,'w') as f:

        f.write('Ncores\n'+str(int(keys['Ncores']))+'\n\ncore_per_lens\n'+str(int(keys['cores_per_lens']))+'\n\n')

        f.write('# params_varied\n')

        for key in keys_to_vary.keys():

            f.write(key+'\n')
        f.write('\n\n')
        for pname in keys_to_vary.keys():

            f.write(pname+':\n')

            for key in keys_to_vary[pname].keys():

                f.write(key+' '+str(keys_to_vary[pname][key])+'\n')
            f.write('\n')

        f.write('\n')

        f.write('# truths\n')

        for key in keys['chain_truths'].keys():
            f.write(key+' '+str(keys['chain_truths'][key])+'\n')

        f.write('\n# info\n')

        f.write(keys['chain_description'])

#runABC(prefix+'data/LOS_CDM_1/',1)




