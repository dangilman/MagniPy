import numpy as np
from MagniPy.util import *
from halo_constructor import Constructor
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

    halo_generator = Constructor(zlens=zlens, zsrc=zsrc)

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

    fit_fluxes = []
    shears, shear_pa, xcen, ycen = [], [], [], []
    n = 0

    if method=='lenstronomy':

        while len(fit_fluxes)<Ntotal:

            print 'rendering halos... '
            halos = halo_generator.render(massprofile=massprofile, model_name=halo_model, model_args=model_args, Nrealizations=1,
                                          filter_halo_positions=filter_halo_positions, **filter_kwargs_list[n])

            print 'optimizing... '
            model_data, system = solver.two_step_optimize(macromodel=start_macromodel,datatofit=data[n],realizations=halos,
                                                     multiplane=multiplane,method=method,ray_trace=True,sigmas=sigmas,
                                                     identifier=identifier,grid_rmax=grid_rmax,res=res,source_shape='GAUSSIAN',
                                                    source_size=source_size,raytrace_with=raytrace_with,print_mag=False)

            for sys,dset in zip(system,model_data):

                if dset.nimg != data[0].nimg:
                    continue

                astro_error = chi_square_img(data[0].x,data[0].y,dset.x,dset.y,0.003,reorder=True)
                print astro_error
                if astro_error > 9:
                    continue

                fit_fluxes.append(dset.flux_anomaly(data[0], sum_in_quad=True, index=0))
                shears.append(sys.lens_components[0].shear)
                xcen.append(sys.lens_components[0].lenstronomy_args['center_x'])
                ycen.append(sys.lens_components[0].lenstronomy_args['center_y'])
                shear_pa.append(sys.lens_components[0].shear_theta)

    elif method=='lensmodel':

        print 'building realizations... '
        halos = halo_generator.render(massprofile=massprofile, model_name=halo_model, model_args=model_args,
                                      Nrealizations=Ntotal,
                                      filter_halo_positions=filter_halo_positions, **filter_kwargs_list[0])

        print 'solving realizations... '

        model_data, system = solver.two_step_optimize(macromodel=start_macromodel, datatofit=data[0],
                                                      realizations=halos,
                                                      multiplane=multiplane, method=method, ray_trace=True,
                                                      sigmas=sigmas,
                                                      identifier=identifier, grid_rmax=grid_rmax, res=res,
                                                      source_shape='GAUSSIAN',
                                                      source_size=source_size, raytrace_with=raytrace_with,
                                                      print_mag=False)

        for dset in model_data:

            if dset.nimg != data[0].nimg:
                continue

            astro_error = chi_square_img(data[0].x, data[0].y, dset.x, dset.y, 0.003, reorder=True)

            if astro_error > 9:
                continue

            fit_fluxes.append(dset[0].flux_anomaly(data[0], sum_in_quad=True,index=0))
            shears.append(dset[0].lens_components[0].shear)
            xcen.append(dset[0].lens_components[0].lenstronomy_args['center_x'])
            ycen.append(dset[0].lens_components[0].lenstronomy_args['center_y'])
            shear_pa.append(dset[0].lens_components[0].shear_theta)



    if write_to_file:

        write_fluxes(filename=outfilepath+outfilename+'.txt',fluxes=np.array(fit_fluxes),mode='append')
        with open(fluxratio_data_path + identifier+ '_shears.txt', 'a') as f:
            np.savetxt(f, X=np.array(shears))
        with open(fluxratio_data_path + identifier+ '_shear_pa_LOS.txt', 'a') as f:
            np.savetxt(f, X=np.array(shear_pa))
        with open(fluxratio_data_path + identifier+ '_xcenter_LOS.txt', 'a') as f:
            np.savetxt(f, X=np.array(xcen))
        with open(fluxratio_data_path + identifier+ '_ycenter_LOS.txt', 'a') as f:
            np.savetxt(f, X=np.array(ycen))

    else:

        return model_data,fit_fluxes,system


