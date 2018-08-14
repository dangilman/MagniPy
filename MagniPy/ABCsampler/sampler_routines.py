from MagniPy.ABCsampler.param_sample import ParamSample
from MagniPy.paths import *
from MagniPy.util import *
from MagniPy.LensBuild.defaults import *
from copy import deepcopy

def perturb_data(data,delta_pos,delta_flux):

    new_data = deepcopy(data)

    for i,sigma in enumerate(delta_pos):
        new_data.x[i] = np.random.normal(data.x[i],sigma)
        new_data.y[i] = np.random.normal(data.y[i],sigma)
        new_data.m[i] = np.random.normal(data.m[i],max(0.00001,delta_flux[i]*data.m[i]))

    return new_data

def initialize_macro(solver,data,init):

    _, model = solver.optimize_4imgs_lenstronomy(macromodel=init, datatofit=data, multiplane=False,
                                                      grid_rmax=None, source_shape='GAUSSIAN', source_size=0.001,
                                                        tol_source=1e-5, tol_mag=0.1, tol_centroid=0.05,
                                                      centroid_0=[0, 0], n_particles=100, n_iterations=700,
                                                      polar_grid=True, optimize_routine='fixed_powerlaw_shear', verbose=True,
                                                      re_optimize=False, particle_swarm=True, restart=5)

    return model

def set_chain_keys(zlens=None, zsrc=None, source_size=None, multiplane=None, SIE_gamma=2, SIE_shear=None,
                   SIE_shear_start=0.04, mindis_front=0.5,mindis_back=0.4, log_masscut_low=7, sigmas=None, grid_rmax=None, grid_res=None,
                   raytrace_with=None,
                   solve_method=None, lensmodel_varyflags=None, data_to_fit=None, chain_ID=None, Nsamples=None,
                   mass_profile=None, mass_func_type=None,
                   log_mL=None, log_mH=None, fsub=None, A0=None, logmhm=None, zmin=None, zmax=None, params_to_vary={},
                   core_index=int,
                   chain_description='', chain_truths={}, Ncores=int, cores_per_lens=int, main_halo_args={},
                   position_sigma=None, flux_sigma=None):
    chain_keys = {}

    chain_keys['zlens'] = zlens
    chain_keys['zsrc'] = zsrc
    chain_keys['source_size'] = source_size
    if SIE_gamma is not None:
        chain_keys['SIE_gamma'] = SIE_gamma

    chain_keys['multiplane'] = multiplane

    for param in main_halo_args.keys():
        chain_keys.update({param: main_halo_args[param]})

    chain_keys['mindis_front'] = mindis_front
    chain_keys['mindis_back'] = mindis_back
    chain_keys['log_masscut_low'] = log_masscut_low

    chain_keys['grid_rmax'] = grid_rmax
    chain_keys['grid_res'] = grid_res
    chain_keys['raytrace_with'] = raytrace_with
    chain_keys['solve_method'] = solve_method
    if lensmodel_varyflags is None:
        lensmodel_varyflags = ['1', '1', '1', '1', '1', '1', '1', '0', '0', '0']
    chain_keys['varyflags'] = lensmodel_varyflags

    chain_keys['x_to_fit'] = data_to_fit.x.tolist()
    chain_keys['y_to_fit'] = data_to_fit.y.tolist()
    chain_keys['flux_to_fit'] = data_to_fit.m.tolist()
    chain_keys['position_sigma'] = position_sigma
    chain_keys['flux_sigma'] = flux_sigma
    chain_keys['source'] = [data_to_fit.srcx, data_to_fit.srcy]

    if sigmas is None:
        sigmas = default_sigmas

    if data_to_fit.t[1] == 0 and data_to_fit.t[2] == 0:
        sigmas[-1] = [100] * 4

    chain_keys['t_to_fit'] = data_to_fit.t.tolist()
    chain_keys['sigmas'] = sigmas
    chain_keys['chain_ID'] = chain_ID
    chain_keys['output_folder'] = chain_keys['chain_ID'] + '/'

    chain_keys['Nsamples'] = Nsamples
    chain_keys['chain_description'] = chain_description
    chain_keys['chain_truths'] = chain_truths
    chain_keys['Ncores'] = Ncores
    chain_keys['cores_per_lens'] = cores_per_lens

    chain_keys['mass_profile'] = mass_profile
    chain_keys['mass_func_type'] = mass_func_type
    chain_keys['log_mL'] = log_mL
    chain_keys['log_mH'] = log_mH

    if fsub is not None:
        assert A0 is None
        chain_keys['fsub'] = fsub
    elif A0 is not None:
        assert fsub is None
        chain_keys['A0'] = fsub
    if logmhm is not None:
        chain_keys['logmhm'] = logmhm

    chain_keys['zmin'] = zmin
    chain_keys['zmax'] = zmax

    chain_keys_to_vary = {}

    for name in chain_keys.keys():
        if name in params_to_vary.keys():
            chain_keys_to_vary[name] = params_to_vary[name]

    return {'main_keys': chain_keys, 'tovary_keys': chain_keys_to_vary}


def write_param_dictionary(fname='', param_dictionary={}):
    with open(fname, 'w') as f:
        f.write(str(param_dictionary))


def read_paraminput(file):
    with open(file, 'r') as f:
        vals = f.read()

    return eval(vals)

def build_dictionary_list(paramnames=[], values=[], dictionary_to_duplicate=None, N=int):
    params = []

    for i in range(0, N):

        if dictionary_to_duplicate is not None:
            new_dictionary = deepcopy(dictionary_to_duplicate)
        else:
            new_dictionary = {}

        for index, pname in enumerate(paramnames):

            if isinstance(values[index], np.ndarray) or isinstance(values[index], list):
                new_dictionary.update({pname: values[index][i]})
            else:
                new_dictionary.update({pname: values[index]})

        params.append(new_dictionary)

    return params

def halo_model_args(params):

    args = {}

    names = ['log_mL', 'log_mH', 'logmhm']

    for name in names:
        args.update({name: params[name]})

    mass_func_type = params['mass_func_type']

    if mass_func_type == 'plaw_main' or mass_func_type == 'composite_plaw':

        args.update({'mass_func_type':mass_func_type})

        if 'A0_perasec2' in params.keys():
            args.update({'A0_perasec2': params['A0_perasec2']})
        elif 'fsub' in params.keys():
            args.update({'fsub': params['fsub']})

        if 'M_halo' in params.keys():
            args.update({'M_halo': params['M_halo']})
            args.update({'tidal_core': params['tidal_core']})
            args.update({'r_core': params['r_core']})

        elif 'Rs' in params.keys():
            args.update({'Rs': params['Rs']})
            args.update({'r200_kpc': params['r200_kpc']})

        if 'plaw_order2' in params.keys():
            args.update({'plaw_order2': True})

        if 'rmax2d_asec' in params.keys():
            args.update({'rmax2d_asec': params['rmax2d_asec']})

        if 'zmin' in params.keys():
            args.update({'zmin':params['zmin']})
        else:
            args.update({'zmin':0})
        if 'zmax' in params.keys():
            args.update({'zmax':params['zmax']})
        else:
            args.update({'zmax':params['zsrc']})

    return args


def get_inputfile_path(chain_ID, core_index):
    info_file = chain_ID + '/paramdictionary_1.txt'
    temp_keys = read_paraminput(info_file)

    cores_per_lens = temp_keys['main_keys']['cores_per_lens']
    Nlens = temp_keys['main_keys']['Ncores'] * temp_keys['main_keys']['cores_per_lens'] ** -1

    data_id = []

    for d in range(0, int(Nlens)):
        data_id += [d + 1] * cores_per_lens

    f_index = data_id[core_index - 1]

    return chain_ID + 'paramdictionary_' + str(f_index) + '.txt'


def initialize(chain_ID, core_index):

    inputfile_path = get_inputfile_path(chain_ID, core_index)

    all_keys = read_paraminput(inputfile_path)

    chain_keys = all_keys['main_keys']

    chain_keys['scratch_file'] = chain_keys['chain_ID'] + '_' + str(core_index)

    chain_keys_to_vary = all_keys['tovary_keys']

    output_path = chainpath + chain_keys['output_folder']

    if os.path.exists(output_path):
        pass
    else:
        create_directory(output_path)

    output_path = chainpath + chain_keys['output_folder'] + 'chain' + str(core_index) + '/'

    if os.path.exists(output_path + 'fluxes.txt') and os.path.exists(output_path + 'parameters.txt'):
        return False, False, False, False

    if os.path.exists(output_path):
        pass
    else:
        create_directory(output_path)

    if os.path.exists(path_2_lensmodel + 'lensmodel'):
        pass
    else:
        shutil.copy2(lensmodel_location + 'lensmodel', path_2_lensmodel)

    return chain_keys, chain_keys_to_vary, output_path, True