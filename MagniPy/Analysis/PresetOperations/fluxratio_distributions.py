import numpy as np
from MagniPy.util import *
from MagniPy.LensBuild.renderhalos import HaloGen
from MagniPy.Solver.solveroutines import SolveRoutines
from MagniPy.LensBuild.defaults import *
from MagniPy.paths import *
from MagniPy.lensdata import Data

def initialize_macromodel(init_macromodel=None,data2fit=None,method=None,sigmas=None,grid_rmax=None,res=None,zlens=None,zsrc=None,
                          source_size=None,outfilename=None,multiplane=None,raytrace_with=None):

    datatofit = Data(x=data2fit[0], y=data2fit[1], m=data2fit[2], t=data2fit[3], source=None)
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

    _, macromodel = solver.two_step_optimize(macromodel=init_macromodel,datatofit=datatofit,multiplane=multiplane,sigmas=sigmas,identifier=None,
                             ray_trace=True,method=method,raytrace_with=raytrace_with,grid_rmax=grid_rmax,res=res,source_size=source_size)

    return macromodel[0].lens_components[0],datatofit



def reoptimize_with_halos(data2fit=[], realizations=None, outfilename='', zlens=None, multiplane_flag=None, zsrc=None,
                          start_macromodels=None, identifier=None, grid_rmax=None, res=None, sigmas=None,
                          source_size=None, raytrace_with=None, test_only=False, write_to_file=False,
                          filter_halo_positions=None, outfilepath=None, method=None, **filter_kwargs):

    datatofit = Data(x=data2fit[0],y=data2fit[1],m=data2fit[2],t=data2fit[3],source=None)

    solver = SolveRoutines(zmain=zlens, zsrc=zsrc, clean_up=True, temp_folder=outfilename)

    model_data, _ = solver.fit(macromodel=start_macromodels, datatofit=datatofit, realizations=realizations,
                                             multiplane=multiplane_flag, method=method, ray_trace=True, sigmas=sigmas,
                                             identifier=identifier, grid_rmax=grid_rmax, res=res, source_shape='GAUSSIAN',
                                             source_size=source_size, raytrace_with=raytrace_with, print_mag=True)


    if write_to_file:
        write_data(outfilepath+outfilename+'.txt', model_data)
    else:
        return model_data

def compute_fluxratio_distributions(subhalo_model_profile='', subhalo_model_type='', subhalo_model_args={},
                                    data2fit=[], Nrealizations=int, outfilename='', zlens=None, z_src=None,
                                    start_macromodel=None, identifier=None, grid_rmax=None, res=None, sigmas=None,
                                    source_size=None, raytrace_with=None, test_only=False, write_to_file=False,
                                    filter_halo_positions=None, outfilepath=None, method=None, **filter_kwargs):

    datatofit = Data(x=data2fit[0],y=data2fit[1],m=data2fit[2],t=data2fit[3],source=None)

    if method is None:
        method = default_solve_method

    if write_to_file:

        if outfilepath is None:
            outfilepath = fluxratio_data_path

        assert os.path.exists(outfilepath)

    if test_only:
        print outfilename
        print subhalo_model_profile
        print subhalo_model_type
        print subhalo_model_args
        return

    if start_macromodel is None:
        start_macromodel = get_default_SIE(zlens)
        start_macromodel.redshift = zlens
    if sigmas is None:
        sigmas = default_sigmas
    if grid_rmax is None:
        grid_rmax = default_gridrmax(source_size)
    if res is None:
        res = default_res(source_size)

    halo_generator = HaloGen(zd=zlens,zsrc=z_src)

    if start_macromodel is None:
        start_macromodel = get_default_SIE(zlens)
        start_macromodel.redshift = zlens
    if sigmas is None:
        sigmas = default_sigmas
    if grid_rmax is None:
        grid_rmax = default_gridrmax(source_size)
    if res is None:
        res = default_res(source_size)

    solver = SolveRoutines(zmain=zlens, zsrc=z_src, temp_folder=outfilename)

    if subhalo_model_type == 'plaw_main':
        spatial_name = 'uniform_cored_nfw'
        multiplane = False
    elif subhalo_model_type == 'plaw_LOS':
        spatial_name = 'uniform2d'
        multiplane = True
    elif subhalo_model_type == 'delta_LOS':
        spatial_name = 'uniform2d'
        multiplane = True
    elif subhalo_model_type == 'composite_plaw':
        spatial_name = 'uniform_cored_nfw'
        multiplane = True

    halos = halo_generator.draw_model(model_name=subhalo_model_type, spatial_name=spatial_name,
                                      massprofile=subhalo_model_profile, model_kwargs=subhalo_model_args, Nrealizations=Nrealizations,
                                      filter_halo_positions=filter_halo_positions,**filter_kwargs)

    model_data, _ = solver.two_step_optimize(macromodel=start_macromodel,datatofit=datatofit,realizations=halos,
                                             multiplane=multiplane,method=method,ray_trace=True,sigmas=sigmas,
                                             identifier=identifier,grid_rmax=grid_rmax,res=res,source_shape='GAUSSIAN',
                                            source_size=source_size,raytrace_with=raytrace_with,print_mag=True)


    if write_to_file:
        write_data(outfilepath+outfilename+'.txt', model_data)
    else:
        return model_data
