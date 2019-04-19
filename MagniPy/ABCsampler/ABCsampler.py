from MagniPy.Solver.solveroutines import SolveRoutines
from MagniPy.ABCsampler.sampler_routines import *
from copy import deepcopy,copy
from MagniPy.paths import *
from pyHalo.pyhalo import pyHalo
import time
from MagniPy.util import approx_theta_E

def initialize_macro(solver,data,init):

    _, model = solver.optimize_4imgs_lenstronomy(macromodel=init, datatofit=data, multiplane=True,
                                                 source_shape='GAUSSIAN', source_size_kpc=0.05,
                                                 tol_source=1e-5, tol_mag=0.2, tol_centroid=0.05,
                                                 centroid_0=[0, 0], n_particles=60, n_iterations=700,pso_convergence_mean=5000,
                                                 simplex_n_iter=400, polar_grid=False, optimize_routine='fixed_powerlaw_shear',
                                                 verbose=False, re_optimize=False, particle_swarm=True, restart=2)

    return model

def init_macromodels(keys_to_vary, chain_keys_run, solver, data, chain_keys):

    macromodels_init = []

    if 'SIE_gamma' in keys_to_vary:
        gamma_values = [2, 2.04, 2.08, 2.12, 2.16, 2.2]
        #gamma_values = [2]
        for gi in gamma_values:
            _macro = get_default_SIE(z=chain_keys_run['zlens'])
            _macro.lenstronomy_args['gamma'] = gi
            macro_i = initialize_macro(solver, data, _macro)
            macro_i[0].lens_components[0].set_varyflags(chain_keys['varyflags'])
            macromodels_init.append(macro_i)

    else:
        gamma_values = [chain_keys_run['SIE_gamma']]
        _macro = get_default_SIE(z=chain_keys_run['zlens'])
        _macro.update_lenstronomy_args({'gamma': chain_keys_run['SIE_gamma']})
        macro_i = initialize_macro(solver, data, _macro)
        macro_i[0].lens_components[0].set_varyflags(chain_keys['varyflags'])
        macromodels_init.append(initialize_macro(solver, data, _macro))

    return macromodels_init, gamma_values

def choose_macromodel_init(macro_list, gamma_values, chain_keys_run):

    dgamma = np.absolute(np.array(gamma_values) - chain_keys_run['SIE_gamma'])

    index = np.argmin(dgamma)

    return macro_list[index]

def run_lenstronomy(data, prior, keys, keys_to_vary, halo_constructor, solver, output_path, write_header):

    chaindata = []
    parameters = []
    start = True
    N_computed = 0
    init_macro = False
    t0 = time.time()
    readout_steps = 50

    while N_computed < keys['Nsamples']:

        samples = prior.sample(scale_by='Nsamples')[0]

        chain_keys_run = copy(keys)

        for i,pname in enumerate(keys_to_vary.keys()):

            chain_keys_run[pname] = samples[i]

        if not init_macro:
            print('initializing macromodels.... ')
            macro_list, gamma_values = init_macromodels(keys_to_vary, chain_keys_run, solver, data, chain_keys_run)
            init_macro = True

        if 'SIE_gamma' in keys_to_vary:

            base = choose_macromodel_init(macro_list, gamma_values, chain_keys_run)

        else:

            base = macro_list[0]

        macromodel = deepcopy(base[0])
        macromodel.lens_components[0].update_lenstronomy_args({'gamma': chain_keys_run['SIE_gamma']})

        halo_args = halo_model_args(chain_keys_run)

        d2fit = perturb_data(data,chain_keys_run['position_sigma'],chain_keys_run['flux_sigma'])

        while True:
            print(halo_args)
            halos = halo_constructor.render(chain_keys_run['mass_func_type'], halo_args, nrealizations=1)

            try:
                #print('source size: ',chain_keys_run['source_size_kpc'])

                new, _, _ = solver.hierarchical_optimization(macromodel=macromodel.lens_components[0], datatofit=d2fit,
                                   realizations=halos, multiplane=True, n_particles=20, n_iterations=450,
                                   verbose=False, re_optimize=True, restart=1, particle_swarm=True, pso_convergence_mean=20000,
                                   pso_compute_magnification=1000, source_size_kpc=chain_keys_run['source_size_kpc'],
                                    simplex_n_iter=400, polar_grid=False, grid_res=0.002,
                                    LOS_mass_sheet_back=chain_keys_run['LOS_mass_sheet_back'],
                                     LOS_mass_sheet_front=chain_keys_run['LOS_mass_sheet_front'])

                xfit, yfit = new[0].x, new[0].y
                #print(new[0].m)

            except:
                print('error in fitting positions...')
                xfit, yfit = np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])

            if chi_square_img(d2fit.x,d2fit.y,xfit,yfit,0.003) < 1:
                break

        N_computed += 1
        if N_computed%readout_steps == 0:
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

        if N_computed%readout_steps == 0:

            readout(output_path, chaindata, parameters, list(keys_to_vary.keys()), write_header)
            start = True
            write_header = False

        else:
            start = False

    print('time elapsed: ', time.time() - t0)
    return chaindata, parameters

def runABC(chain_ID='',core_index=int):

    chain_keys,chain_keys_to_vary,output_path,run = initialize(chain_ID,core_index)

    if run is False:
        return

    # Initialize data
    datatofit = Data(x=chain_keys['x_to_fit'], y=chain_keys['y_to_fit'], m=chain_keys['flux_to_fit'],
                     t=chain_keys['t_to_fit'], source=chain_keys['source'])

    chain_keys.update({'cone_opening_angle': 5*approx_theta_E(datatofit.x, datatofit.y)})

    write_data(output_path + 'lensdata.txt',[datatofit], mode='write')

    print('lens redshift: ', chain_keys['zlens'])
    print('source redshift: ', chain_keys['zsrc'])
    print('opening angle: ', 5*approx_theta_E(datatofit.x, datatofit.y))

    if run is False:
        return

    solver = SolveRoutines(zlens=chain_keys['zlens'], zsrc=chain_keys['zsrc'],
                           temp_folder=chain_keys['scratch_file'], clean_up=True)

    param_names_tovary = chain_keys_to_vary.keys()
    write_info_file(chainpath + chain_keys['output_folder'] + 'simulation_info.txt',
                    chain_keys, chain_keys_to_vary, param_names_tovary)
    #copy_directory(chain_ID + '/R_index_config.txt', chainpath + chain_keys['output_folder'])

    prior = ParamSample(params_to_vary=chain_keys_to_vary, Nsamples=1)

    constructor = pyHalo(np.round(chain_keys['zlens'],2), np.round(chain_keys['zsrc'],2))

    if chain_keys['solve_method'] == 'lensmodel':

        raise Exception('not yet implemented.')

    else:

        run_lenstronomy(datatofit, prior, chain_keys, chain_keys_to_vary, constructor, solver, output_path,
                        chain_keys['write_header'])

def write_params(params,fname,header, mode='append'):

    if mode == 'append':
        m = 'a'
    else:
        m = 'w'

    with open(fname, m) as f:

        if header is not None:
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

#runABC(prefix+'data/SIDM_run/',1)