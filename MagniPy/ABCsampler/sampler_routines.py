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

def set_chain_keys(keys_to_vary, **kwargs):

    lens_data = kwargs['data_to_fit']
    kwargs['x_to_fit'] = lens_data.x.tolist()
    kwargs['y_to_fit'] = lens_data.y.tolist()
    kwargs['flux_to_fit'] = lens_data.m.tolist()
    kwargs['t_to_fit'] = lens_data.t.tolist()
    kwargs['source'] = [lens_data.srcx, lens_data.srcy]
    del kwargs['data_to_fit']

    for name in kwargs['halo_args_init'].keys():
        kwargs.update({name: kwargs['halo_args_init'][name]})

    del kwargs['halo_args_init']

    kwargs['output_folder'] = kwargs['chain_ID'] + '/'

    return {'main_keys': kwargs, 'tovary_keys': keys_to_vary}

def write_param_dictionary(fname='', param_dictionary={}):
    with open(fname, 'w') as f:
        f.write(str(param_dictionary))

def read_paraminput(file):
    try:
        with open(file, 'r') as f:
            vals = f.read()
    except:
        raise Exception('could not locate file '+file)

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

        names = ['log_mlow', 'log_mhigh', 'log_m_break', 'cone_opening_angle', 'opening_angle_factor','power_law_index',
                 'parent_m200', 'c_scale', 'c_power', 'break_index', 'mdef_los', 'mdef_main',
                 'LOS_normalization', 'RocheNorm', 'RocheNu', 'break_scale']

        if 'parent_m200' in params.keys():
            assert 'log_m_parent' not in params.keys()

        for name in names:
            if name in params.keys():
                args.update({name: params[name]})

        args.update({'mass_func_type':mass_func_type})

        if 'log_m_parent' in params.keys():
            args.update({'parent_m200': 10**params['log_m_parent']})

        if 'a0_area' in params.keys():
            args.update({'a0_area': params['a0_area']})
        elif 'sigma_sub' in params.keys():
            args.update({'sigma_sub': params['sigma_sub']})

        if 'zmin' in params.keys():
            args.update({'zmin':params['zmin']})
        else:
            args.update({'zmin':0.01})

        if 'zmax' in params.keys():
            args.update({'zmax':params['zmax']})
        else:
            args.update({'zmax':params['zsrc']-0.01})

        if 'core_ratio' in params.keys():

            args.update({'core_ratio': params['core_ratio']})

        if 'SIDMcross' in params.keys():
            args.update({'SIDMcross': params['SIDMcross']})
            args.update({'vpower': params['vpower']})

    return args

def write_params(params,fname,header, mode):

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

def write_shear_values(fname, m, values):

    with open(fname, mode=m) as f:
        for row in range(0, int(np.shape(values)[0])):
            towrite = str(values[row,0]) + ' ' + str(values[row,1]) + ' ' + str(values[row,2]) + '\n'
            f.write(towrite)

def update_satellites(chain_keys_run, params_varied):

    if chain_keys_run['satellites'] is None:
        return chain_keys_run['satellites']

    sat = deepcopy(chain_keys_run['satellites'])

    nsatellites = chain_keys_run['n_satellites']

    if nsatellites == 1:

        if 'satellite_x' in params_varied:
            sat['kwargs_satellite'][0]['center_x'] = chain_keys_run['satellite_x']
        if 'satellite_y' in params_varied:
            sat['kwargs_satellite'][0]['center_y'] = chain_keys_run['satellite_y']
        if 'satellite_thetaE' in params_varied:
            sat['kwargs_satellite'][0]['theta_E'] = chain_keys_run['satellite_thetaE']


    elif nsatellites == 2:

        if 'satellite_x_1' in params_varied:
            sat['kwargs_satellite'][0]['center_x'] = chain_keys_run['satellite_x_1']
        if 'satellite_y_1' in params_varied:
            sat['kwargs_satellite'][0]['center_y'] = chain_keys_run['satellite_y_1']
        if 'satellite_thetaE_1' in params_varied:
            sat['kwargs_satellite'][0]['theta_E'] = chain_keys_run['satellite_thetaE_1']

        if 'satellite_x_2' in params_varied:
            sat['kwargs_satellite'][1]['center_x'] = chain_keys_run['satellite_x_2']
        if 'satellite_y_2' in params_varied:
            sat['kwargs_satellite'][1]['center_y'] = chain_keys_run['satellite_y_2']
        if 'satellite_thetaE_2' in params_varied:
            sat['kwargs_satellite'][1]['theta_E'] = chain_keys_run['satellite_thetaE_2']


    return sat

def read_macro_array(lens_system):

    lensmodel = lens_system.lens_components[0]

    rein = lensmodel.lenstronomy_args['theta_E']

    cenx, ceny = lensmodel.lenstronomy_args['center_x'], lensmodel.lenstronomy_args['center_y']
    ellipandPA = lensmodel.ellip_PA_polar()
    ellip = ellipandPA[0]
    ellipPA = ellipandPA[1]
    shear = lensmodel.shear
    shearPA = lensmodel.shear_theta
    gamma = lensmodel.lenstronomy_args['gamma']
    if lens_system._has_satellites:

        _, _, kwargs_sat, convention = lens_system.satellite_properties
        coords = []
        for kw in kwargs_sat:
            coords.append([kw['center_x'], kw['center_y']])

        if len(coords) == 1:
            return np.array([rein, cenx, ceny, ellip, ellipPA, shear, shearPA, gamma, coords[0][0], coords[0][1],
                             kwargs_sat[0]['theta_E']])
        elif len(coords) == 2:
            return np.array([rein, cenx, ceny, ellip, ellipPA, shear, shearPA, gamma, coords[0][0], coords[0][1],
                             kwargs_sat[0]['theta_E'], coords[1][0], coords[1][1], kwargs_sat[1]['theta_E']])

    else:
        return np.array([rein, cenx, ceny, ellip, ellipPA, shear, shearPA, gamma])

def readout_realizations(optimized_lens_model, full_lens_model, outputpath, statistic, params, fluxes):

    zlist, lens_list, arg_list, _, _ = optimized_lens_model.lenstronomy_lists()
    zlistfull, lens_listfull, arg_listfull, _, _ = full_lens_model.lenstronomy_lists()

    full_masses = []
    full_c = []
    full_x = []
    full_y = []
    full_z = []

    for halo in full_lens_model.realization.halos:
        full_masses.append(halo.mass)
        full_c.append(halo.mass_def_arg[0])
        full_x.append(halo.x)
        full_y.append(halo.y)
        full_z.append(halo.z)

    full_masses = np.log10(full_masses)
    full_c = np.array(full_c)
    full_x = np.array(full_x)
    full_y = np.array(full_y)
    full_z = np.array(full_z)

    full = np.column_stack((full_masses, full_c))
    full = np.column_stack((full, full_x))
    full = np.column_stack((full, full_y))
    full = np.column_stack((full, full_z))

    masses = []
    c = []
    x, y, z = [], [], []

    for halo in optimized_lens_model.realization.halos:
        masses.append(halo.mass)
        c.append(halo.mass_def_arg[0])
        x.append(halo.x)
        y.append(halo.y)
        z.append(halo.z)

    masses = np.log10(masses)
    c = np.array(c)
    x, y, z = np.array(x), np.array(y), np.array(z)

    partial = np.column_stack((masses, c))
    partial = np.column_stack((partial, x))
    partial = np.column_stack((partial, y))
    partial = np.column_stack((partial, z))

    with open(outputpath+'best_params.txt', 'w') as f:
        f.write(str(statistic)+'\n')
        for fi in fluxes:
            f.write(str(fi)+' ')
        f.write('\n')
        for pi in params:
            f.write(str(pi)+' ')

    with open(outputpath + 'best_realization.txt', 'w') as f:

        f.write('redshifts = '+str(repr(list(zlist)))+'\n\n')
        f.write('lens_model_list = '+str(repr(lens_list)) + '\n\n')
        f.write('lens_model_args = '+str(repr(arg_list)) + '\n\n')

    np.savetxt(outputpath + 'best_mc.txt', X=partial, fmt='%.3f')
    np.savetxt(outputpath + 'best_mc_full.txt', X=full, fmt='%.3f')

    with open(outputpath + 'best_fullrealization.txt', 'w') as f:

        f.write('redshifts = '+str(repr(list(zlistfull)))+'\n\n')
        f.write('lens_model_list = '+str(repr(lens_listfull)) + '\n\n')
        f.write('lens_model_args = '+str(repr(arg_listfull)) + '\n\n')


def summary_stat_flux(f_obs, f_model):

    ratios_obs = f_obs[1:] * f_obs[0] ** -1
    ratios_model = f_model[1:] * f_model[0] ** -1

    diff = np.sqrt(np.sum((ratios_obs - ratios_model)**2))

    return diff

def readout_macro(output_path, params, open_file):

    if open_file:
        mode = 'w'
    else:
        mode = 'a'

    with open(output_path + 'macro.txt', mode) as f:
        rows, cols = int(np.shape(params)[0]), int(np.shape(params)[1])
        for row in range(0, rows):
            for col in range(0, cols):
                f.write(str(np.round(params[row,col],4)) + ' ')
            f.write('\n')

def readout(output_path, fluxes, parameters, param_names_tovary, write_header):

    if write_header:
        header_string = ''
        for name in param_names_tovary:
            header_string += name + ' '
        write_params(parameters, output_path + 'parameters.txt', header_string, mode='write')
        write_fluxes(output_path + 'fluxes.txt', fluxes=fluxes, summed_in_quad=False, mode='write')

    else:

        write_params(parameters,output_path + 'parameters.txt', None, mode='append')
        write_fluxes(output_path + 'fluxes.txt', fluxes=fluxes, summed_in_quad=False, mode='append')

def readout_withshear(output_path, fluxes, parameters, param_names_tovary, shear_values_1, shear_values_2,
                    shear_values_3, shear_values_4, write_header):

    if write_header:
        header_string = ''
        for name in param_names_tovary:
            header_string += name + ' '
        write_params(parameters, output_path + 'parameters.txt', header_string, mode='write')
        write_fluxes(output_path + 'fluxes.txt', fluxes=fluxes, summed_in_quad=False, mode='write')

        for i in range(0, len(shear_values_1)):
            write_shear_values(output_path + 'shearvals_img1_' + str(i + 1) + '.txt', 'w', shear_values_1[i])
            write_shear_values(output_path + 'shearvals_img2_' + str(i + 1) + '.txt', 'w', shear_values_2[i])
            write_shear_values(output_path + 'shearvals_img3_' + str(i + 1) + '.txt', 'w', shear_values_3[i])
            write_shear_values(output_path + 'shearvals_img4_' + str(i + 1) + '.txt', 'w', shear_values_4[i])

    else:

        write_params(parameters,output_path + 'parameters.txt', None, mode='append')
        write_fluxes(output_path + 'fluxes.txt', fluxes=fluxes, summed_in_quad=False, mode='append')
        for i in range(0, len(shear_values_1)):
            write_shear_values(output_path + 'shearvals_img1_' + str(i + 1) + '.txt', 'a', shear_values_1[i])
            write_shear_values(output_path + 'shearvals_img2_' + str(i + 1) + '.txt', 'a', shear_values_2[i])
            write_shear_values(output_path + 'shearvals_img3_' + str(i + 1) + '.txt', 'a', shear_values_3[i])
            write_shear_values(output_path + 'shearvals_img4_' + str(i + 1) + '.txt', 'a', shear_values_4[i])


def get_inputfile_path(chain_ID, core_index):
    info_file = chain_ID + '/paramdictionary_1.txt'
    print('searching directory '+info_file)
    temp_keys = read_paraminput(info_file)

    cores_per_lens = temp_keys['main_keys']['cores_per_lens']
    Nlens = temp_keys['main_keys']['Ncores'] * temp_keys['main_keys']['cores_per_lens'] ** -1

    data_id = []

    Nlens = int(np.round(Nlens))

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
    readout_best = True

    if os.path.exists(output_path + 'fluxes.txt') and os.path.exists(output_path + 'parameters.txt'):
        values = np.loadtxt(output_path + 'fluxes.txt')
        N_lines = int(np.shape(values)[0])

        if N_lines > chain_keys['Nsamples']*0.5:
            readout_best = False

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

    return chain_keys, chain_keys_to_vary, output_path, True, readout_best