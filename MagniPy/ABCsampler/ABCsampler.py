from MagniPy.ABCsampler.sampler_routines import *
from copy import deepcopy,copy
from MagniPy.paths import *
from pyHalo.pyhalo import pyHalo
import time
from MagniPy.util import approx_theta_E

from lenstronomywrapper.LensSystem.macrolensmodel import MacroLensModel
from lenstronomywrapper.LensSystem.LensComponents.powerlawshear import PowerLawShear
from lenstronomywrapper.LensSystem.LensComponents.satellite import SISsatellite
from lenstronomywrapper.ModelingWorkflow.flux_ratio_forwardmodel import forward_model

def run_lenstronomy(data, prior, keys, keys_to_vary,
                    output_path, write_header, readout_best):

    chaindata = []
    parameters = []
    start = True
    N_computed = 0
    t0 = time.time()

    if 'readout_steps' in keys.keys():
        readout_steps = keys['readout_steps']
    else:
        readout_steps = 50

    if 'n_particles' in keys.keys():
        n_particles = keys['n_particles']
    else:
        n_particles = 20

    if 'simplex_n_iter' in keys.keys():
        simplex_n_iter = keys['simplex_n_iter']
    else:
        simplex_n_iter = 200

    if 'n_iterations' in keys.keys():
        n_iterations = keys['n_iterations']
    else:
        n_iterations = 250

    if 'verbose' in keys.keys():
        verbose = keys['verbose']
    else:
        verbose = False

    if verbose:
        print('Running with:')
        print('n_particles: ', n_particles)
        print('n_iterations: ', n_iterations)
        print('simplex_iterations: ', simplex_n_iter)

    current_best = 1e+6
    best_fluxes = [0,0,0,0]

    if write_header:
        save_statistic = True

    else:
        save_statistic = False

    if 'lens_redshift' not in keys_to_vary.keys():
        halo_constructor = pyHalo(np.round(keys['zlens'], 2), np.round(keys['zsrc'], 2))

    while N_computed < keys['Nsamples']:

        d2fit = perturb_data(data, keys['position_sigma'], keys['flux_sigma'])

        print('N computed: ', N_computed)

        while True:

            samples = prior.sample(scale_by='Nsamples')[0]

            chain_keys_run = copy(keys)

            for i,pname in enumerate(keys_to_vary.keys()):

                chain_keys_run[pname] = samples[i]

            if 'lens_redshift' in keys_to_vary.keys():
                halo_constructor = pyHalo(np.round(chain_keys_run['lens_redshift'], 2), np.round(chain_keys_run['zsrc'], 2))

            if 'shear' in keys_to_vary.keys():
                opt_routine = 'fixedshearpowerlaw'
                constrain_params = {'shear': chain_keys_run['shear']}

            else:
                opt_routine = 'fixed_powerlaw_shear'
                constrain_params = None

            kwargs_init = [{'theta_E': 1., 'center_x': 0., 'center_y': 0, 'e1': 0.1, 'e2': 0.1,
                           'gamma': chain_keys_run['SIE_gamma']}, {'gamma1': 0.02, 'gamma2': 0.01}]

            lens_main = PowerLawShear(halo_constructor.zlens, kwargs_init)
            lens_list = [lens_main]

            kwargs_realization = halo_model_args(chain_keys_run, verbose)

            include_satellites = update_satellites(chain_keys_run, keys_to_vary)

            if include_satellites is not None:

                kwargs, zsat, nsat = include_satellites[0], include_satellites[1]
                assert nsat < 3
                if nsat < 2:
                    if 'lens_redshift' in keys_to_vary.keys():
                        raise Exception('must allow for satellite redshift to vary with lens redshift')
                    satellite = SISsatellite(zsat[0], kwargs[0])
                    lens_list.append(satellite)
                if nsat < 3:
                    satellite = SISsatellite(zsat[1], kwargs[1])
                    lens_list.append(satellite)

            macromodel = MacroLensModel(lens_list)

            out = forward_model(d2fit.x, d2fit.y, d2fit.m, macromodel, chain_keys_run['source_fwhm_pc'],
                            halo_constructor, kwargs_realization, chain_keys_run['mass_func_type'],
                                opt_routine, constrain_params, verbose, test_mode=False)

            if np.isfinite(out['summary_stat']):
                break

        N_computed += 1
        if N_computed%readout_steps == 0 and verbose:
            print('completed ' + str(N_computed) + ' of '+str(keys['Nsamples'])+'...')

        samples_array = []

        for pname in keys_to_vary.keys():
            samples_array.append(chain_keys_run[pname])

        if 'shear' not in keys_to_vary.keys() and 'save_shear' in keys.keys():
            samples_array.insert(keys['save_shear']['idx'], np.round(out['external_shear'],4))

        if save_statistic:
            new_statistic = out['summary_stat']
            #print(new_statistic, current_best)
            if new_statistic < current_best:
                current_best = new_statistic
                params_best = samples_array
                best_fluxes = out['magnifications_fit']

        if start:
            macro_array = out['macromodel_parameters']
            chaindata = out['magnifications_fit']
            parameters = np.array(samples_array)
        else:
            macro_array = np.vstack((macro_array, out['macromodel_parameters']))
            chaindata = np.vstack((chaindata, out['magnifications_fit']))
            parameters = np.vstack((parameters, np.array(samples_array)))

        if N_computed%readout_steps == 0:
            readout_macro(output_path, macro_array, write_header)
            readout(output_path, chaindata, parameters, list(keys_to_vary.keys()), write_header)
            if save_statistic and readout_best:

                readout_realizations(out['lens_system_optimized'], output_path, current_best,
                                     params_best, best_fluxes)
            start = True
            write_header = False

        else:
            start = False

    if verbose: print('time elapsed: ', time.time() - t0)
    return chaindata, parameters

def runABC(chain_ID='',core_index=int):

    print('trying to find paramdictionary file... ')
    chain_keys,chain_keys_to_vary,output_path,run, readout_best = initialize(chain_ID,core_index)

    if run is False:
        return

    # Initialize data
    datatofit = Data(x=chain_keys['x_to_fit'], y=chain_keys['y_to_fit'], m=chain_keys['flux_to_fit'],
                     t=chain_keys['t_to_fit'], source=chain_keys['source'])

    rein_main = approx_theta_E(datatofit.x, datatofit.y)

    chain_keys.update({'R_ein_main': rein_main})
    chain_keys.update({'cone_opening_angle': chain_keys['opening_angle_factor'] * rein_main})

    write_data(output_path + 'lensdata.txt',[datatofit], mode='write')

    print('lens redshift: ', chain_keys['zlens'])
    print('source redshift: ', chain_keys['zsrc'])
    print('opening angle: ', chain_keys['opening_angle_factor']*approx_theta_E(datatofit.x, datatofit.y))

    if run is False:
        return

    param_names_tovary = chain_keys_to_vary.keys()
    write_info_file(chainpath + chain_keys['output_folder'] + 'simulation_info.txt',
                    chain_keys, chain_keys_to_vary, param_names_tovary)
    #copy_directory(chain_ID + '/R_index_config.txt', chainpath + chain_keys['output_folder'])

    prior = ParamSample(params_to_vary=chain_keys_to_vary, Nsamples=1)

    run_lenstronomy(datatofit, prior, chain_keys, chain_keys_to_vary, output_path,
                        chain_keys['write_header'], readout_best)

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

#cpl = 2000
#L = 21
#index = (L-1)*cpl + 1
#runABC(prefix+'data/test_run/', 1)


