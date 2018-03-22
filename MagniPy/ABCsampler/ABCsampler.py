from MagniPy.Analysis.PresetOperations.fluxratio_distributions import *
from param_sample import ParamSample
from MagniPy.Analysis.PresetOperations.halo_constructor import halo_constructor
from copy import deepcopy
import shutil

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

def runABC(inputfile_path='',Nsplit=1000):

    #chain_keys = read_paraminput(inputfile_path)

    output_path = chainpath + chain_keys['sampler']['output_folder']

    if os.path.exists(output_path):
        pass
    else:
        create_directory(output_path)

    output_path = chainpath + chain_keys['sampler']['output_folder']+'chain'+str(chain_keys['sampler']['core_index'])+'/'

    if os.path.exists(output_path):
        pass
    else:
        create_directory(output_path)

    if os.path.exists(path_2_lensmodel+'lensmodel'):
        pass
    else:
        shutil.copy2(lensmodel_location+'lensmodel',path_2_lensmodel)

    # Initialize data, macormodel
    datatofit = [chain_keys['data']['x_to_fit'], chain_keys['data']['y_to_fit'], chain_keys['data']['flux_to_fit'], chain_keys['data']['t_to_fit']]
    macromodel_default_start = default_startkwargs

    # Get parameters to vary
    prior = ParamSample(params_to_vary=chain_keys_to_vary,Nsamples=chain_keys['sampler']['Nsamples'])
    samples_full = prior.sample(scale_by='Nsamples')
    param_names_tovary = prior.param_names
    if len(chain_keys_to_vary_sparse.keys())!=0:
        prior_sparse = ParamSample(params_to_vary=chain_keys_to_vary_sparse,Nsamples=int(np.shape(samples_full)[0]))
        sparse_samples = prior_sparse.sample(scale_by='Nsamples')
        param_names_tovary += prior_sparse.param_names
        samples = np.hstack((samples_full,sparse_samples))
    else:
        samples = samples_full

    chainkeys = {}

    for group,items in chain_keys.iteritems():
        for pname,value in items.iteritems():
            if pname not in param_names_tovary:
                chainkeys.update({pname:value})

    run_commands = build_dictionary_list(paramnames=param_names_tovary,values=np.squeeze(np.split(samples,np.shape(samples)[1],axis=1)),
                                         N=int(np.shape(samples)[0]),dictionary_to_duplicate=chainkeys)

    for i in range(0,len(run_commands)):
        run_commands[i].update({'halo_model_args':halo_model_args(run_commands[0]['mass_func_type'],run_commands[0])})

    # Draw realizations
    realizations = []
    for params in run_commands:
        realizations += halo_constructor(massprofile=params['mass_profile'],
                                         model_name=params['mass_func_type'],
                                         model_args=params['halo_model_args'], Nrealizations=1, zlens=params['zlens'],
                                         zsrc=params['zsrc'], filter_halo_positions=True,
                                         x_filter=datatofit[0], y_filter=datatofit[1], mindis=params['mindis'],
                                         log_masscut_low=params['log_masscut_low'])

    macromodel_default_start['shear'] = chainkeys['SIE_shear_start']

    macromodel_start = Deflector(subclass=SIE(), tovary=True,
                                 varyflags=chainkeys['varyflags'],
                                 redshift=chain_keys['lens']['zlens'],
                                 **macromodel_default_start)

    macromodel,lensdata = initialize_macromodel(macromodel_start, data2fit=datatofit, method=chainkeys['solve_method'],
                                       sigmas=chainkeys['sigmas'],
                                       grid_rmax=chainkeys['grid_rmax'], res=chainkeys['grid_res'],
                                       zlens=chainkeys['zlens'], zsrc=chainkeys['zsrc'],
                                       source_size=chainkeys['source_size'], outfilename=chainkeys['scratch_file'],
                                       multiplane=chainkeys['multiplane'], raytrace_with=chainkeys['raytrace_with'])

    macromodels = []

    if 'SIE_gamma' or 'SIE_shear' in param_names_tovary:

        for commands in run_commands:

            newmac = deepcopy(macromodel)

            if 'SIE_gamma' in commands:
                newmac.args['gamma'] = commands['SIE_gamma']
            elif 'SIE_shear' in commands:
                newmac.args['shear'] = commands['SIE_shear']

            macromodels.append(newmac)

    else:
        macromodels = [macromodel]*len(run_commands)

    if len(run_commands) < 1000:
        Nsplit = len(run_commands)

    if len(run_commands)%Nsplit != 0:
        Nsplit = 1000
    if len(run_commands)%Nsplit !=0:
        Nsplit = 600
    if len(run_commands)%Nsplit !=0:
        Nsplit = 500

    assert len(run_commands)%Nsplit==0,'run_commands length '+str(len(run_commands))+\
                                       ' and Nsplit length '+str(Nsplit)+' not compatible.'

    N_run = len(run_commands)*Nsplit**-1

    chain_data = []
    for i in range(0, int(N_run)):
        chain_data += reoptimize_with_halos(start_macromodels=macromodels,
                                            realizations=realizations[i * Nsplit:(i + 1) * Nsplit],
                                            data2fit=datatofit, outfilename=run_commands[i]['chain_ID'],
                                            zlens=run_commands[i]['zlens'],
                                            zsrc=run_commands[i]['zsrc'], identifier=run_commands[i]['chain_ID']+'_',
                                            grid_rmax=run_commands[i]['grid_rmax'],res=run_commands[i]['grid_res'],
                                            multiplane_flag=run_commands[i]['multiplane'],sigmas=run_commands[i]['sigmas'],
                                            source_size=run_commands[i]['source_size'],
                                            raytrace_with=run_commands[i]['raytrace_with'],test_only=False,
                                            write_to_file=False,outfilepath=run_commands[i]['scratch_file'],
                                            method=run_commands[i]['solve_method'])


    write_data(output_path+'chain.txt',chain_data)

    write_data(output_path+'lensdata.txt',[lensdata])

    header_string = ''
    for name in param_names_tovary:
        header_string+= name + ' '

    np.savetxt(output_path+'parameters.txt',samples,header = header_string, fmt='%.6f')

    with open(output_path+'info.txt','w') as f:
        for pname in param_names_tovary:
            f.write(pname+' ')
        f.write('\n')
        for key,value in chain_keys_to_vary.iteritems():
            f.write(key+' '+str(value)+'\n')






chain_keys = {}
core_index = 1

chain_keys['sampler'] = {}
chain_keys['lens'] = {}
chain_keys['halos'] = {}
chain_keys['modeling'] = {}
chain_keys['data'] = {}

chain_keys['lens']['zlens'] = 0.5
chain_keys['lens']['zsrc'] = 1.5
chain_keys['lens']['source_size'] = 0.0012*2.3**-1
chain_keys['lens']['SIE_gamma'] = 2
chain_keys['lens']['SIE_shear'] = 0.04
chain_keys['lens']['SIE_shear_start'] = chain_keys['lens']['SIE_shear']
chain_keys['lens']['multiplane']=False

chain_keys['modeling']['mindis'] = 0.5
chain_keys['modeling']['log_masscut_low'] = 7
chain_keys['modeling']['sigmas'] = default_sigmas
chain_keys['modeling']['grid_rmax'] = 0.06
chain_keys['modeling']['grid_res'] = 0.001
chain_keys['modeling']['raytrace_with'] = 'lenstronomy'
chain_keys['modeling']['solve_method'] = 'lensmodel'
chain_keys['modeling']['varyflags'] = ['1','1','1','1','1','1','1','0','0','0']

chain_keys['data']['x_to_fit'] = [0.912 ,-0.9246,-0.5254,0.5011]
chain_keys['data']['y_to_fit'] = [0.6344, -0.6008 , 0.7895,-0.7962]
chain_keys['data']['flux_to_fit'] = [ 0.97634,1,0.69093,0.66614]
chain_keys['data']['t_to_fit'] = [ 0,0.9,14.4,15.2]

chain_keys['sampler']['chain_ID'] = 'tester_chain'
chain_keys['sampler']['scratch_file'] = chain_keys['sampler']['chain_ID'] + '_'+str(core_index)
chain_keys['sampler']['output_folder'] = chain_keys['sampler']['chain_ID']+'/'
chain_keys['sampler']['core_index'] = core_index
chain_keys['sampler']['Nsamples'] = 10

chain_keys['halos']['mass_profile'] = 'NFW'
chain_keys['halos']['mass_func_type'] = 'plaw_main'
chain_keys['halos']['log_mL'] = 6
chain_keys['halos']['log_mH'] = 10
chain_keys['halos']['fsub'] = 1
chain_keys['halos']['logmhm'] = 0
chain_keys['halos']['zmin'] = 0
chain_keys['halos']['zmax'] = 1

params_to_vary = {}
params_to_vary['fsub'] = {'prior_type':'Uniform','low':0,'high':0.01,'steps':21,'sample_full':True}
params_to_vary['logmhm'] = {'prior_type':'Uniform','low':6,'high':7,'steps':21,'sample_full':True}
params_to_vary['SIE_gamma'] = {'prior_type':'Gaussian','mean':2,'sigma':0.05,'positive_definite':True,'sample_full':True}
#params_to_vary['SIE_shear'] = {'prior_type':'Gaussian','mean':chain_keys['lens']['SIE_shear_start'],'sigma':0.01,'positive_definite':True,'sample_full':False}
chain_keys_to_vary = {}
chain_keys_to_vary_sparse = {}

for groupname,items in chain_keys.iteritems():

    for item_name in items:

        if item_name in params_to_vary.keys():

            if params_to_vary[item_name]['sample_full']:
                chain_keys_to_vary[item_name] = params_to_vary[item_name]
            else:
                chain_keys_to_vary_sparse[item_name] = params_to_vary[item_name]

runABC('')



