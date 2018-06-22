from MagniPy.util import *
from halo_constructor import Constructor
from MagniPy.Solver.solveroutines import SolveRoutines
from MagniPy.LensBuild.defaults import *
from MagniPy.paths import *
from MagniPy.lensdata import Data
import random
from create_data import create_data

def cross_section_compute(massprofile='', halo_model='', model_args={},
                                    data2fit=None, Ntotal=int, outfilename='', zlens=None, zsrc=None,
                                    identifier=None, grid_rmax=None, res=None, sigmas=None,
                                    source_size=None, raytrace_with='lenstronomy',filter_halo_positions=False, outfilepath=None,
                                    ray_trace=True, method='lenstronomy',
                                    start_shear=0.05,mindis=0.5,log_masscut_low=7,mass_bin_size = 1):

    configs = ['cross']
    data = []

    if data2fit is None:
        for i in range(0,Ntotal):
            config = random.choice(configs)

            data.append(create_data(identifier='dset',config=config,zlens=zlens,zsrc=zsrc,substructure_model_args={'fsub':0,'M_halo':10**13},massprofile=massprofile,
                             halo_model='plaw_main',multiplane=False,ray_trace=True,astrometric_perturbation=0,return_system=False,
                                    shear_prior=[start_shear,1e-9]))

    else:
        data = [Data(x=data2fit[0],y=data2fit[1],m=data2fit[2],t=data2fit[3],source=None)]*Ntotal

    filter_kwargs_list = []

    if filter_halo_positions:

        for i in range(0,Ntotal):

            filter_kwargs_list.append({'x_filter':data[i].x,'y_filter':data[i].y,'mindis':mindis,'log_masscut_low':log_masscut_low})
    else:

        for i in range(0,Ntotal):

            filter_kwargs_list.append({})

    start_macromodel = get_default_SIE(zlens)
    start_macromodel.redshift = zlens

    halo_generator = Constructor(zlens=zlens, zsrc=zsrc)

    solver = SolveRoutines(zlens=zlens, zsrc=zsrc, temp_folder=identifier)

    if halo_model == 'plaw_main':
        multiplane = False
    elif halo_model == 'plaw_LOS':
        multiplane = True
    elif halo_model == 'delta_LOS':
        multiplane = True
    elif halo_model == 'composite_plaw':
        multiplane = True

    # initialize macromodel
    start_macromodel.shear = start_shear

    residuals = []
    n = 0
    bin_centers = []

    while n<Ntotal:

        halos_init = halo_generator.render(massprofile=massprofile, model_name=halo_model, model_args=model_args, Nrealizations=1,
                                           filter_halo_positions=filter_halo_positions, **filter_kwargs_list[n])
        print 'solving... '
        data_control, system = solver.two_step_optimize(macromodel=start_macromodel, datatofit=data[n],
                                                      realizations=halos_init,
                                                      multiplane=multiplane, method=method, ray_trace=True,
                                                      sigmas=sigmas,
                                                      identifier=identifier, grid_rmax=grid_rmax, res=res,
                                                      source_shape='GAUSSIAN',
                                                      source_size=source_size, raytrace_with=raytrace_with,
                                                      print_mag=False)

        astro_error = np.sqrt(np.sum((data[n].x - data_control[0].x) ** 2 + (data[n].y - data_control[0].y) ** 2))

        if astro_error > 1e-5:
            continue

        masses = []
        other_indicies = []

        only_halos = [object for object in halos_init[0] if object.other_args['name'] not in ['CONVERGENCE','SHEAR','SPEMD']]

        for halo in only_halos:
            masses.append(halo.other_args['mass'])

        masses = np.log10(masses)

        mass_bin_center = np.random.uniform(model_args['log_mL']+mass_bin_size,model_args['log_mH']-mass_bin_size)

        include = np.where(np.absolute(masses - mass_bin_center) > mass_bin_size)

        halos = [object for i,object in enumerate(only_halos) if i in include[0]]

        try:
            halos += halos_init[other_indicies[0]]
        except:
            pass

        newdata, system = solver.two_step_optimize(macromodel=start_macromodel, datatofit=data[n],
                                                        realizations=[halos],
                                                        multiplane=multiplane, method=method, ray_trace=True,
                                                        sigmas=sigmas,
                                                        identifier=identifier, grid_rmax=grid_rmax, res=res,
                                                        source_shape='GAUSSIAN',
                                                        source_size=source_size, raytrace_with=raytrace_with,
                                                        print_mag=False)

        astro_error = np.sqrt(np.sum((data[n].x - newdata[0].x) ** 2 + (data[n].y - newdata[0].y) ** 2))

        if astro_error > 1e-5:
            continue

        residuals.append(newdata[0].flux_anomaly(data_control[0],sum_in_quad=True))
        bin_centers.append(mass_bin_center)
        n+=1

    return bin_centers,residuals

#print cross_section_compute('TNFW',halo_model='plaw_main',
#                            model_args={'fsub':0.005,'log_mL':7,'log_mH':10,'M_halo':10**13,'tidal_core':True,'r_core':'Rs'},
#                           zlens=0.5,zsrc=1.5,Ntotal=1,grid_rmax=0.12,res=0.001,source_size=0.0005,filter_halo_positions=True)






