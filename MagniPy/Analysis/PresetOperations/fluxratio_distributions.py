import numpy as np
from MagniPy.util import *
from halo_constructor import Realization
from MagniPy.Solver.solveroutines import SolveRoutines
from MagniPy.LensBuild.defaults import *
from MagniPy.paths import *
from MagniPy.lensdata import Data
from create_data import create_data
import random

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
                                    data2fit=None, Ntotal=int, outfilename='', zlens=None, zsrc=None,
                                    start_macromodel=None, identifier=None, grid_rmax=None, res=None, sigmas=None,
                                    source_size=None, raytrace_with='lenstronomy', test_only=False, write_to_file=False,
                                    filter_halo_positions=False, outfilepath=None,ray_trace=True, method='lenstronomy',
                                    start_shear=0.05,mindis=0.5,log_masscut_low=7):

    configs = ['cross']
    data = []

    if data2fit is None:
        for i in range(0,Ntotal):
            config = random.choice(configs)

            data.append(create_data(identifier='dset',config=config,zlens=zlens,zsrc=zsrc,substructure_model_args={'fsub':0,'M_halo':10**13},massprofile=massprofile,
                             halo_model='plaw_main',multiplane=False,ray_trace=True,astrometric_perturbation=0,return_gamma=False,
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

    if write_to_file:
        assert outfilepath is not None
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

    fit_fluxes = None
    n = 0
    print 'solving realizations... '

    while n<Ntotal:

        halos = halo_generator.halo_constructor(massprofile=massprofile,model_name=halo_model,model_args=model_args,Nrealizations=1,
                                                filter_halo_positions=filter_halo_positions,**filter_kwargs_list[n])

        model_data, system = solver.two_step_optimize(macromodel=start_macromodel,datatofit=data[n],realizations=halos,
                                                 multiplane=multiplane,method=method,ray_trace=True,sigmas=sigmas,
                                                 identifier=identifier,grid_rmax=grid_rmax,res=res,source_shape='GAUSSIAN',
                                                source_size=source_size,raytrace_with=raytrace_with,print_mag=False)

        if model_data[0].nimg!=data[n].nimg:
            continue

        astro_error = np.sqrt(np.sum((data[n].x - model_data[0].x) ** 2 + (data[n].y - model_data[0].y) ** 2))

        if astro_error<1e-5:

            if fit_fluxes is None:
                fit_fluxes = model_data[0].flux_anomaly(data[n], index=0, sum_in_quad=True)
            else:
                fit_fluxes = np.vstack((fit_fluxes, model_data[0].flux_anomaly(data[n], index=0, sum_in_quad=True)))

            n += 1
            print n

    if write_to_file:

        write_fluxes(filename=outfilepath+outfilename+'.txt',fluxes=fit_fluxes,mode='append')

    else:

        return model_data,fit_fluxes,system


