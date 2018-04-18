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

    solver = SolveRoutines(zmain=zlens, zsrc=zsrc, temp_folder=outfilename)

    _, macromodel = solver.two_step_optimize(macromodel=init_macromodel,datatofit=data2fit,multiplane=multiplane,sigmas=sigmas,identifier=None,
                             ray_trace=True,method=method,raytrace_with=raytrace_with,grid_rmax=grid_rmax,res=res,source_size=source_size)

    return macromodel[0].lens_components[0]


def reoptimize_with_halos(data2fit=classmethod, realizations=None, outfilename='', zlens=None, multiplane_flag=None, zsrc=None,
                          start_macromodels=None, identifier=None, grid_rmax=None, res=None, sigmas=None,
                          source_size=None, raytrace_with=None, test_only=False, write_to_file=False,
                          filter_halo_positions=None, outfilepath=None, method=None, **filter_kwargs):


    solver = SolveRoutines(zmain=zlens, zsrc=zsrc, clean_up=True, temp_folder=outfilename)

    model_data, _ = solver.fit(macromodel=start_macromodels, datatofit=data2fit, realizations=realizations,
                                             multiplane=multiplane_flag, method=method, ray_trace=True, sigmas=sigmas,
                                             identifier=identifier, grid_rmax=grid_rmax, res=res, source_shape='GAUSSIAN',
                                             source_size=source_size, raytrace_with=raytrace_with, print_mag=True)


    if write_to_file:
        write_data(outfilepath+outfilename+'.txt', model_data)
    else:
        return model_data

def compute_fluxratio_distributions(massprofile='', halo_model='', model_args={},
                                    data2fit=[], Nrealizations=int, outfilename='', zlens=None, zsrc=None,
                                    start_macromodel=None, identifier=None, grid_rmax=None, res=None, sigmas=None,
                                    source_size=None, raytrace_with='lenstronomy', test_only=False, write_to_file=False,
                                    filter_halo_positions=None, outfilepath=None, method='lenstronomy', **filter_kwargs):

    if isinstance(data2fit,list):
        data2fit = Data(x=data2fit[0],y=data2fit[1],m=data2fit[2],t=data2fit[3],source=None)

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

    solver = SolveRoutines(zmain=zlens, zsrc=zsrc, temp_folder=outfilename)

    if halo_model == 'plaw_main':

        multiplane = False
    elif halo_model == 'plaw_LOS':

        multiplane = True
    elif halo_model == 'delta_LOS':

        multiplane = True
    elif halo_model == 'composite_plaw':

        multiplane = True

    halos = halo_generator.halo_constructor(massprofile=massprofile,model_name=halo_model,model_args=model_args,Nrealizations=Nrealizations,
                                            filter_halo_positions=filter_halo_positions,**filter_kwargs)

    model_data, _ = solver.two_step_optimize(macromodel=start_macromodel,datatofit=data2fit,realizations=halos,
                                             multiplane=multiplane,method=method,ray_trace=True,sigmas=sigmas,
                                             identifier=identifier,grid_rmax=grid_rmax,res=res,source_shape='GAUSSIAN',
                                            source_size=source_size,raytrace_with=raytrace_with,print_mag=True)


    if write_to_file:
        write_data(outfilepath+outfilename+'.txt', model_data, mode='append')
    else:
        return model_data

