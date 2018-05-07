from MagniPy.Analysis.PresetOperations.fluxratio_distributions import *
from param_sample import ParamSample
from MagniPy.Analysis.PresetOperations.halo_constructor import Realization
from copy import deepcopy,copy
from MagniPy.paths import *
import shutil
from time import time

def set_chain_keys(zlens=None, zsrc=None, source_size=None, multiplane=None, SIE_gamma=2, SIE_shear=None,
                   SIE_shear_start=0.04, mindis=0.5, log_masscut_low=7, sigmas=None, grid_rmax=None, grid_res=None, raytrace_with=None,
                   solve_method=None, lensmodel_varyflags=None, data_to_fit=None, chain_ID=None, Nsamples=None, mass_profile=None, mass_func_type=None,
                   log_mL=None, log_mH=None, fsub=None, A0=None, logmhm=None, zmin=None, zmax=None, params_to_vary={},core_index=int,
                   chain_description='',chain_truths={},Ncores=int,cores_per_lens=int):

    chain_keys = {}
    chain_keys['sampler'] = {}
    chain_keys['lens'] = {}
    chain_keys['halos'] = {}
    chain_keys['modeling'] = {}
    chain_keys['data'] = {}

    chain_keys['lens']['zlens'] = zlens
    chain_keys['lens']['zsrc'] = zsrc
    chain_keys['lens']['source_size'] = source_size
    if SIE_gamma is not None:
        chain_keys['lens']['SIE_gamma'] = SIE_gamma
    if SIE_shear_start is None:
        SIE_shear_start = SIE_shear
    chain_keys['lens']['SIE_shear'] = SIE_shear_start
    chain_keys['lens']['SIE_shear_start'] = SIE_shear_start
    chain_keys['lens']['multiplane'] = multiplane

    chain_keys['modeling']['mindis'] = mindis
    chain_keys['modeling']['log_masscut_low'] = log_masscut_low

    chain_keys['modeling']['grid_rmax'] = grid_rmax
    chain_keys['modeling']['grid_res'] = grid_res
    chain_keys['modeling']['raytrace_with'] = raytrace_with
    chain_keys['modeling']['solve_method'] = solve_method
    if lensmodel_varyflags is None:
        lensmodel_varyflags = ['1', '1', '1', '1', '1', '1', '1', '0', '0', '0']
    chain_keys['modeling']['varyflags'] = lensmodel_varyflags

    chain_keys['data']['x_to_fit'] = data_to_fit.x.tolist()
    chain_keys['data']['y_to_fit'] = data_to_fit.y.tolist()
    chain_keys['data']['flux_to_fit'] = data_to_fit.m.tolist()

    if sigmas is None:
        sigmas = default_sigmas

    if data_to_fit.t[1]==0 and data_to_fit.t[2] == 0:
        sigmas[-1] = [100]*4

    chain_keys['data']['t_to_fit'] = data_to_fit.t.tolist()
    chain_keys['modeling']['sigmas'] = sigmas
    chain_keys['sampler']['chain_ID'] = chain_ID
    chain_keys['sampler']['output_folder'] = chain_keys['sampler']['chain_ID'] + '/'

    chain_keys['sampler']['Nsamples'] = Nsamples
    chain_keys['sampler']['chain_description'] = chain_description
    chain_keys['sampler']['chain_truths'] = chain_truths
    chain_keys['sampler']['Ncores'] = Ncores
    chain_keys['sampler']['cores_per_lens'] = cores_per_lens

    chain_keys['halos']['mass_profile'] = mass_profile
    chain_keys['halos']['mass_func_type'] = mass_func_type
    chain_keys['halos']['log_mL'] = log_mL
    chain_keys['halos']['log_mH'] = log_mH

    if fsub is not None:
        chain_keys['halos']['fsub'] = fsub
    elif A0 is not None:
        chain_keys['halos']['A0'] = fsub

    if logmhm is not None:
        chain_keys['halos']['logmhm'] = logmhm

    chain_keys['halos']['zmin'] = zmin
    chain_keys['halos']['zmax'] = zmax

    chain_keys_to_vary = {}

    for groupname,items in chain_keys.iteritems():

        for item_name in items:

            if item_name in params_to_vary.keys():

                chain_keys_to_vary[item_name] = params_to_vary[item_name]

    return {'main_keys':chain_keys,'tovary_keys':chain_keys_to_vary}

def write_param_dictionary(fname='',param_dictionary={}):

    with open(fname,'w') as f:
        f.write(str(param_dictionary))

def read_paraminput(file):
    with open(file,'r') as f:
        vals = f.read()

    return eval(vals)

def build_dictionary_list(paramnames=[],values=[],dictionary_to_duplicate=None,N=int):

    params = []

    for i in range(0,N):

        if dictionary_to_duplicate is not None:
            new_dictionary = deepcopy(dictionary_to_duplicate)
        else:
            new_dictionary = {}

        for index,pname in enumerate(paramnames):


            if isinstance(values[index],np.ndarray) or isinstance(values[index],list):
                new_dictionary.update({pname:values[index][i]})
            else:
                new_dictionary.update({pname:values[index]})

        params.append(new_dictionary)

    return params

def halo_model_args(mass_func_type='',params={}):

    args = {}

    names = ['log_mL', 'log_mH', 'logmhm']

    for name in names:
        args.update({name: params[name]})

    if mass_func_type=='plaw_main' or mass_func_type=='composite_plaw':
        if 'A0_perasec2' in params.keys():
            args.update({'A0_perasec2':params['A0_perasec2']})
        elif 'fsub' in params.keys():
            args.update({'fsub':params['fsub']})
    if mass_func_type=='composite_plaw':
        args.update({'zmin':params['zmin']})
        args.update({'zmax':params['zmax']})

    return args

def get_inputfile_path(chain_ID,core_index):

    info_file = chain_ID + '/paramdictionary_1.txt'
    temp_keys = read_paraminput(info_file)

    cores_per_lens = temp_keys['main_keys']['sampler']['cores_per_lens']
    Nlens = temp_keys['main_keys']['sampler']['Ncores'] * temp_keys['main_keys']['sampler']['cores_per_lens'] ** -1

    data_id = []

    for d in range(0, int(Nlens)):
        data_id += [d + 1] * cores_per_lens

    f_index = data_id[core_index-1]

    return chain_ID + 'paramdictionary_' + str(f_index) + '.txt'

def runABC(chain_ID='',core_index=int,Nsplit=1000):

    inputfile_path = get_inputfile_path(chain_ID,core_index)

    all_keys = read_paraminput(inputfile_path)

    chain_keys = all_keys['main_keys']

    chain_keys['sampler']['scratch_file'] = chain_keys['sampler']['chain_ID'] + '_' + str(core_index)

    chain_keys_to_vary = all_keys['tovary_keys']

    output_path = chainpath + chain_keys['sampler']['output_folder']

    if os.path.exists(output_path):
        pass
    else:
        create_directory(output_path)

    output_path = chainpath + chain_keys['sampler']['output_folder']+'chain'+str(core_index)+'/'

    if os.path.exists(output_path+'fluxes.txt') and os.path.exists(output_path+'parameters.txt') and \
            os.path.exists(output_path+'lensdata.txt') and os.path.exists(output_path+'astrometric_errors.txt'):

        return


    if os.path.exists(output_path):
        pass
    else:
        create_directory(output_path)

    if os.path.exists(path_2_lensmodel+'lensmodel'):
        pass
    else:
        shutil.copy2(lensmodel_location+'lensmodel',path_2_lensmodel)

    # Initialize data, macormodel

    datatofit = Data(x=chain_keys['data']['x_to_fit'], y=chain_keys['data']['y_to_fit'], m=chain_keys['data']['flux_to_fit'],
                     t=chain_keys['data']['t_to_fit'], source=None)

    macromodel_default_start = default_startkwargs

    print 'intializing macromodels... '

    macromodel_default_start['shear'] = chain_keys['lens']['SIE_shear_start']

    macromodel_start = get_default_SIE(z=chain_keys['lens']['zlens'])

    solver = SolveRoutines(zlens=chain_keys['lens']['zlens'], zsrc=chain_keys['lens']['zsrc'], temp_folder=chain_keys['sampler']['scratch_file'])

    opt_data, mod = solver.two_step_optimize(macromodel_start, datatofit=datatofit, realizations=None,
                                             multiplane=chain_keys['lens']['multiplane'],
                                             method='lensmodel', ray_trace=False, sigmas=chain_keys['modeling']['sigmas'],
                                             identifier=chain_keys['sampler']['chain_ID'],
                                             grid_rmax=chain_keys['modeling']['grid_rmax'], res=chain_keys['modeling']['grid_res'],
                                             source_size=chain_keys['lens']['source_size'])

    macromodel = mod[0].lens_components[0]
    macromodel.set_varyflags(chain_keys['modeling']['varyflags'])

    # Get parameters to vary
    prior = ParamSample(params_to_vary=chain_keys_to_vary,Nsamples=chain_keys['sampler']['Nsamples'],macromodel=macromodel)
    samples = prior.sample(scale_by='Nsamples')

    param_names_tovary = prior.param_names
    header_string = ''
    for name in param_names_tovary:
        header_string += name + ' '

    write_info_file(chainpath + chain_keys['sampler']['output_folder'] + 'simulation_info.txt',
                    chain_keys, chain_keys_to_vary, param_names_tovary)

    chainkeys = {}

    for group,items in chain_keys.iteritems():
        for pname,value in items.iteritems():
            if pname not in param_names_tovary:
                chainkeys.update({pname:value})

    run_commands = build_dictionary_list(paramnames=param_names_tovary,values=np.squeeze(np.split(samples,np.shape(samples)[1],axis=1)),
                                         N=int(np.shape(samples)[0]),dictionary_to_duplicate=chainkeys)

    for i in range(0,len(run_commands)):
        run_commands[i].update({'halo_model_args':halo_model_args(run_commands[i]['mass_func_type'],run_commands[i])})

    # Draw realizations
    realizations = []
    print 'building realizations... '

    constructor = Realization(zlens=run_commands[0]['zlens'],zsrc=run_commands[0]['zsrc'],LOS_mass_sheet=False)

    t0 = time()
    for index,params in enumerate(run_commands):

        realizations += constructor.halo_constructor(massprofile=params['mass_profile'],
                                         model_name=params['mass_func_type'],
                                         model_args=params['halo_model_args'], Nrealizations=1,
                                        filter_halo_positions=True,
                                         x_filter=datatofit.x, y_filter=datatofit.x, mindis=params['mindis'],
                                         log_masscut_low=params['log_masscut_low'])

    print 'done.'
    print 'time to draw realizations (min): ',np.round((time() - t0)*60**-1,1)

    macromodels = []

    if 'SIE_gamma' or 'SIE_shear' in param_names_tovary:

        for commands in run_commands:

            newmac = deepcopy(macromodel)

            if 'SIE_gamma' in commands:
                newmac.args['gamma'] = commands['SIE_gamma']
            if 'SIE_shear' in commands:
                newmac.args['shear'] = commands['SIE_shear']

            macromodels.append(newmac)


    else:

        macromodels = [macromodel]*len(run_commands)

    print 'done.'

    Nsplit = len(run_commands)

    if len(run_commands)<1000 or chainkeys['solve_method']=='lenstronomy':
        Nsplit = len(run_commands)
    else: Nsplit = 1000

    assert len(run_commands)%Nsplit == 0

    print 'solving realizations... '
    i = 0
    N_computed = 0

    if os.path.exists(output_path + 'lensdata.txt'):
        pass
    else:
        write_data(output_path + 'lensdata.txt', [datatofit])

    chaindata = []

    for i in range(0,int(len(run_commands)*Nsplit**-1)):

        solver = SolveRoutines(zlens=chainkeys['zlens'], zsrc=chainkeys['zsrc'],
                               temp_folder=run_commands[i]['scratch_file'])

        macro_mods = macromodels[i*Nsplit:(i+1)*Nsplit]
        reals = realizations[i * Nsplit:(i + 1) * Nsplit]

        if chainkeys['solve_method'] == 'lensmodel':
            new, _ = solver.fit(macromodel=macro_mods[i],
                                realizations=reals, datatofit=datatofit,
                                multiplane=chainkeys['multiplane'], method=chainkeys['solve_method'], ray_trace=True,
                                sigmas=chainkeys['sigmas'],
                                identifier=run_commands[i]['chain_ID'], grid_rmax=run_commands[i]['grid_rmax'],
                                res=run_commands[i]['grid_res'],
                                source_size=run_commands[i]['source_size'], print_mag=True,
                                raytrace_with=run_commands[i]['raytrace_with'])
        else:
            new, _ = solver.fit(macromodel=macro_mods[i],realizations=reals, datatofit=datatofit,
                                              multiplane=chainkeys['multiplane'], method=chainkeys['solve_method'],
                                              ray_trace=True,
                                              sigmas=chainkeys['sigmas'],
                                              identifier=run_commands[i]['chain_ID'],
                                              grid_rmax=run_commands[i]['grid_rmax'],
                                              res=run_commands[i]['grid_res'],
                                              source_size=run_commands[i]['source_size'], print_mag=True,
                                              raytrace_with=run_commands[i]['raytrace_with'])
        chaindata += new
        N_computed += len(new)

    fluxes,astrometric_errors = [],[]

    for dset in chaindata:

        if dset.nimg != datatofit.nimg:
            fluxes.append(np.array([1000, 1000, 1000, 1000]))
            astrometric_errors.append(1000)

        else:
            fluxes.append(dset.m)
            astrometric_errors.append(np.sqrt(np.sum((dset.x - datatofit.x) ** 2 + (dset.y - datatofit.y) ** 2)))

    f_handle = file(output_path + 'astrometric_errors.txt', 'a')
    np.savetxt(f_handle, X=np.array(astrometric_errors), fmt='%.6f')
    f_handle = file(output_path+'fluxes.txt', 'a')
    np.savetxt(f_handle, X=np.array(fluxes), fmt='%.6f')
    f_handle = file(output_path+'parameters.txt', 'a')
    np.savetxt(f_handle, X=np.array(samples), fmt='%.6f',header=header_string)

def write_info_file(fpath,keys,keys_to_vary,pnames_vary):

    with open(fpath,'w') as f:

        f.write('Ncores\n'+str(int(keys['sampler']['Ncores']))+'\n\ncore_per_lens\n'+str(int(keys['sampler']['cores_per_lens']))+'\n\n')

        f.write('# params_varied\n')

        for key,param in keys_to_vary.iteritems():

            f.write(key+'\n')
        f.write('\n\n')
        for key,param in keys_to_vary.iteritems():

            f.write(key+':\n')
            for vary_param,value in param.iteritems():

                f.write(vary_param+' '+str(value)+'\n')
            f.write('\n')

        f.write('\n')

        f.write('# truths\n')

        for key,item in keys['sampler']['chain_truths'].iteritems():
            f.write(key+' '+str(item)+'\n')

        f.write('\n# info\n')

        f.write(keys['sampler']['chain_description'])

#runABC(os.getenv('HOME')+'/data/singleplane_test_lensmod/',1)
