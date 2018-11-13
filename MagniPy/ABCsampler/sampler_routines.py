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

def set_chain_keys(zlens=None, zsrc=None, source_size_kpc=None, multiplane=None, SIE_gamma=2,mindis_front=0.5,
                   mindis_back=0.4, log_masscut_front=None, log_masscut_back=None, sigmas=None, grid_rmax=None, grid_res=None, raytrace_with=None,
                   solve_method=None, lensmodel_varyflags=None, data_to_fit=None, chain_ID=None, Nsamples=None,
                   mass_func_type=None,log_mL=None, log_mH=None, a0_area=None, A0=None, logmhm=None,
                   zmin=None, zmax=None, params_to_vary={},chain_description='',
                   chain_truths={}, Ncores=int,cores_per_lens=int, halo_args_init={},
                   position_sigma=None, flux_sigma=None, single_background=None,Nsamples_perlens=None,
                   LOS_normalization = None):

    chain_keys = {}

    chain_keys['zlens'] = zlens
    chain_keys['zsrc'] = zsrc
    chain_keys['source_size_kpc'] = source_size_kpc
    chain_keys['single_background'] = single_background
    if SIE_gamma is not None:
        chain_keys['SIE_gamma'] = SIE_gamma

    chain_keys['multiplane'] = multiplane

    for param in halo_args_init.keys():
        chain_keys.update({param: halo_args_init[param]})

    chain_keys['mindis_front'] = mindis_front
    chain_keys['mindis_back'] = mindis_back
    chain_keys['log_masscut_front'] = log_masscut_front
    chain_keys['log_masscut_back'] = log_masscut_back

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
    chain_keys['Nsamples_perlens'] = Nsamples_perlens

    chain_keys['mass_func_type'] = mass_func_type
    chain_keys['log_mL'] = log_mL
    chain_keys['log_mH'] = log_mH
    chain_keys['LOS_normalization'] = LOS_normalization

    if a0_area is not None:
        assert A0 is None
        chain_keys['a0_area'] = a0_area

    if logmhm is not None:
        chain_keys['logmhm'] = logmhm

    if zmin is not None:
        chain_keys['zmin'] = zmin
    if zmax is not None:
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

    mass_func_type = params['mass_func_type']

    if mass_func_type == 'composite_powerlaw':

        names = ['log_mlow', 'log_mhigh', 'log_m_break', 'cone_opening_angle', 'power_law_index',
                 'parent_m200', 'c_scale', 'c_power', 'break_index', 'mdef_los', 'mdef_main', 'parent_c',
                 'LOS_normalization']

        for name in names:
            args.update({name: params[name]})

        args.update({'mass_func_type':mass_func_type})

        if 'a0_area' in params.keys():
            args.update({'a0_area': params['a0_area']})
        #elif 'fsub' in params.keys():
        #    args.update({'fsub': params['fsub']})

        if 'zmin' in params.keys():
            args.update({'zmin':params['zmin']})
        else:
            args.update({'zmin':0.01})

        if 'zmax' in params.keys():
            args.update({'zmax':params['zmax']})
        else:
            args.update({'zmax':params['zsrc']-0.01})

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

    print('reading input file: ', inputfile_path)
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
    chain_keys['write_header'] = True
    if os.path.exists(output_path + 'fluxes.txt') and os.path.exists(output_path + 'parameters.txt'):
        values = np.loadtxt(output_path + 'fluxes.txt')
        N_lines = int(np.shape(values)[0])
        if N_lines >= chain_keys['Nsamples']:
            return False, False, False, False
        else:
            chain_keys['Nsamples'] = chain_keys['Nsamples'] - N_lines
            chain_keys['write_header'] = False

    if os.path.exists(output_path):
        pass
    else:
        create_directory(output_path)

    if os.path.exists(path_2_lensmodel + 'lensmodel'):
        pass
    else:
        shutil.copy2(lensmodel_location + 'lensmodel', path_2_lensmodel)

    return chain_keys, chain_keys_to_vary, output_path, True