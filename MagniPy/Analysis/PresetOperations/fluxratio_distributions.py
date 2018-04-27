import numpy as np
from MagniPy.util import *
from halo_constructor import Realization
from MagniPy.Solver.solveroutines import SolveRoutines
from MagniPy.LensBuild.defaults import *
from MagniPy.paths import *
from MagniPy.lensdata import Data

def initialize_macromodel(init_macromodel=None,data2fit=None,method=None,sigmas=None,grid_rmax=None,res=None,zlens=None,zsrc=None,
                          source_size=None,outfilename=None,multiplane=None,raytrace_with=None):

    assert zlens is not None
    if method is None:
        method = default_solve_method

    if init_macromodel is None:
        init_macromodel = get_default_SIE(zlens)

    if sigmas is None:
        sigmas = default_sigmas
    if grid_rmax is None:
        grid_rmax = default_gridrmax(source_size)
    if res is None:
        res = default_res(source_size)

    if init_macromodel is None:
        start_macromodel = get_default_SIE(zlens)
        start_macromodel.redshift = zlens
    if sigmas is None:
        sigmas = default_sigmas
    if grid_rmax is None:
        grid_rmax = default_gridrmax(source_size)
    if res is None:
        res = default_res(source_size)

    solver = SolveRoutines(zlens=zlens, zsrc=zsrc, temp_folder=outfilename)

    _, macromodel = solver.two_step_optimize(macromodel=init_macromodel,datatofit=data2fit,multiplane=multiplane,sigmas=sigmas,identifier=None,
                             ray_trace=True,method=method,raytrace_with=raytrace_with,grid_rmax=grid_rmax,res=res,source_size=source_size)

    return macromodel[0].lens_components[0]


def reoptimize_with_halos(data2fit=classmethod, realizations=None, outfilename='', zlens=None, multiplane_flag=None, zsrc=None,
                          start_macromodels=None, identifier=None, grid_rmax=None, res=None, sigmas=None,
                          source_size=None, raytrace_with=None, test_only=False, write_to_file=False,
                          filter_halo_positions=None, outfilepath=None, method=None, **filter_kwargs):


    solver = SolveRoutines(zlens=zlens, zsrc=zsrc, clean_up=True, temp_folder=outfilename)

    model_data, _ = solver.fit(macromodel=start_macromodels, datatofit=data2fit, realizations=realizations,
                                             multiplane=multiplane_flag, method=method, ray_trace=True, sigmas=sigmas,
                                             identifier=identifier, grid_rmax=grid_rmax, res=res, source_shape='GAUSSIAN',
                                             source_size=source_size, raytrace_with=raytrace_with, print_mag=True)


    if write_to_file:
        write_data(outfilepath+outfilename+'.txt', model_data)
    else:
        return model_data

def compute_fluxratio_distributions(massprofile='', halo_model='', model_args={},
                                    data2fit=[], Ntotal=int, outfilename='', zlens=None, zsrc=None,
                                    start_macromodel=None, identifier=None, grid_rmax=None, res=None, sigmas=None,
                                    source_size=None, raytrace_with='lenstronomy', test_only=False, write_to_file=False,
                                    filter_halo_positions=None, outfilepath=None,ray_trace=True, method='lenstronomy',
                                    start_shear=0.05,mindis=0.5,log_masscut_low=7):

    data2fit = [[-0.12992,-0.91168,0.87564,0.03536],[1.01623,0.56775,0.57489,-0.89155],[1.,0.723173,0.643823,0.264251],[0,0,0,0]]

    if isinstance(data2fit,list):
        data2fit = Data(x=data2fit[0],y=data2fit[1],m=data2fit[2],t=data2fit[3],source=None)

    if filter_halo_positions:
        x_filter = data2fit.x
        y_filter = data2fit.y
        filter_kwargs = {'x_filter':x_filter,'y_filter':y_filter,'mindis':mindis,'log_masscut_low':log_masscut_low}
    else:
        filter_kwargs = {}

    if write_to_file:
        print outfilepath
        assert os.path.exists(outfilepath)

    if start_macromodel is None:
        start_macromodel = get_default_SIE(zlens)
        start_macromodel.redshift = zlens
    if sigmas is None:
        sigmas = default_sigmas

    halo_generator = Realization(zlens=zlens,zsrc=zsrc)

    if start_macromodel is None:
        start_macromodel = get_default_SIE(zlens)
        start_macromodel.redshift = zlens
    if sigmas is None:
        sigmas = default_sigmas
    if grid_rmax is None:
        grid_rmax = default_gridrmax(source_size)
    if res is None:
        res = default_res(source_size)

    solver = SolveRoutines(zlens=zlens, zsrc=zsrc, temp_folder=outfilename)

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
    print 'initializing macromodel... '
    _, macro_init = solver.fit(macromodel=start_macromodel,datatofit=data2fit,realizations=None,
                                                 multiplane=multiplane,method=method,ray_trace=ray_trace,sigmas=sigmas,
                                                 identifier=identifier,grid_rmax=grid_rmax,res=res,source_shape='GAUSSIAN',
                                                source_size=source_size,raytrace_with=raytrace_with,print_mag=False)

    fit_fluxes = None
    n = 0
    print 'solving realizations... '
    while n<Ntotal:

        halos = halo_generator.halo_constructor(massprofile=massprofile,model_name=halo_model,model_args=model_args,Nrealizations=1,
                                                filter_halo_positions=filter_halo_positions,**filter_kwargs)


        model_data, _ = solver.fit(macromodel=macro_init[0].lens_components[0],datatofit=data2fit,realizations=halos,
                                                 multiplane=multiplane,method=method,ray_trace=True,sigmas=sigmas,
                                                 identifier=identifier,grid_rmax=grid_rmax,res=res,source_shape='GAUSSIAN',
                                                source_size=source_size,raytrace_with=raytrace_with,print_mag=False)

        dset = model_data[0]

        astro_error = np.sqrt(np.sum((dset.x - data2fit.x) ** 2 + (dset.y - data2fit.y) ** 2))

        if astro_error<1e-5:

            try:
                fit_fluxes = np.vstack((fit_fluxes,model_data[0].flux_anomaly(data2fit,index=0)))
            except:
                fit_fluxes = model_data[0].flux_anomaly(data2fit,index=0)
            n += 1
            print n

    if write_to_file:

        #write_data(outfilepath+outfilename+'.txt', data, mode='append')
        write_fluxes(filename=outfilepath+outfilename+'.txt',fluxes=fit_fluxes,mode='append')
    else:
        return model_data


