from MagniPy.magnipy import Magnipy
from MagniPy.LensBuild.defaults import raytrace_with_default,default_sigmas,default_res,\
    default_source_shape,default_source_size_kpc,default_solve_method,default_file_identifier,get_default_SIE_random
import copy
import numpy as np
from MagniPy.Solver.LenstronomyWrap.lenstronomy_wrap import LenstronomyWrap
from MagniPy.Solver.hierarchical_optimization import *

class SolveRoutines(Magnipy):

    """
    This class uses the routines set up in MagniPy to solve the lens equation in various ways with lenstronomy or lensmodel
    """

    def hierarchical_optimization(self, datatofit=None, macromodel=None, realizations=None, multiplane=True,
                                  source_shape='GAUSSIAN',
                                  source_size_kpc=None, grid_res=None, tol_source=1e-5, tol_mag=0.2, tol_centroid=0.05,
                                  centroid_0=[0, 0],
                                  n_particles=50, n_iterations=250, polar_grid=False,
                                  optimize_routine='fixed_powerlaw_shear', verbose=False, re_optimize=False,
                                  particle_swarm=True, restart=1,
                                  constrain_params=None, pso_convergence_mean=5000,
                                  pso_compute_magnification=200, tol_simplex_params=1e-3, tol_simplex_func=0.01,
                                  simplex_n_iter=300, background_globalmin_masses=None,
                                  background_aperture_masses=None, background_filters=None,
                                  min_mass=6, m_break=0, particle_swarm_reopt=True, optimize_iteration = None,
                                  reoptimize_scale=None,LOS_mass_sheet_front = 7.7, LOS_mass_sheet_back = 8, satellites=None):

        if source_shape is None:
            source_shape = default_source_shape
        if source_size_kpc is None:
            source_size_kpc = default_source_size_kpc
        if grid_res is None:
            grid_res = default_res(source_size_kpc)

        assert multiplane is True

        if centroid_0[0] != 0 or centroid_0[1] != 0:
            realizations[0] = realizations[0].shift_centroid(centroid_0)

        assert isinstance(source_size_kpc, float) or isinstance(source_size_kpc, int)
        m_ref = max(m_break, min_mass)

        if centroid_0[0] != 0 or centroid_0[1] != 0:
            realizations[0] = realizations[0].shift_centroid(centroid_0)

        foreground_realization, background_realization = split_realization(datatofit, realizations[0])

        print('optimizing foreground... ')
        foreground_rays, foreground_macromodel, foreground_halos, keywords_lensmodel = optimize_foreground(macromodel,
                              [foreground_realization], datatofit, tol_source, tol_mag, tol_centroid, centroid_0,  n_particles,
                              n_iterations, source_shape, source_size_kpc, polar_grid, optimize_routine, re_optimize, verbose, particle_swarm,
                          restart, constrain_params, pso_convergence_mean, pso_compute_magnification, tol_simplex_params,
                            tol_simplex_func, simplex_n_iter, m_ref, self, LOS_mass_sheet_front, LOS_mass_sheet_back,
                                  centroid = centroid_0, satellites=satellites)

        print('optimizing background... ')
        optimized_data, model, outputs, keywords_lensmodel = optimize_background(foreground_macromodel, foreground_halos[0], background_realization, foreground_rays,
                      datatofit, tol_source, tol_mag, tol_centroid, centroid_0, n_particles,  n_iterations,
                      source_shape, source_size_kpc, polar_grid, optimize_routine, re_optimize, verbose,
                        particle_swarm, restart, constrain_params, pso_convergence_mean, pso_compute_magnification,
                        tol_simplex_params, tol_simplex_func, simplex_n_iter, m_ref, self,
                        background_globalmin_masses = background_globalmin_masses,
                         background_aperture_masses = background_aperture_masses, background_filters = background_filters,
                        reoptimize_scale = reoptimize_scale, particle_swarm_reopt = particle_swarm_reopt,
                         LOS_mass_sheet_front = LOS_mass_sheet_front, LOS_mass_sheet_back = LOS_mass_sheet_back,
                          optimize_iteration=optimize_iteration, centroid = centroid_0, satellites=satellites)

        fluxes = self._ray_trace_finite(optimized_data[0].x, optimized_data[0].y, optimized_data[0].srcx, optimized_data[0].srcy, True,
                               keywords_lensmodel['lensModel'], keywords_lensmodel['kwargs_lens'], grid_res, source_shape,
                               source_size_kpc, polar_grid)

        optimized_data[0].set_mag(fluxes)
        optimized_data[0].sort_by_pos(datatofit.x,datatofit.y)

        return optimized_data, model, outputs

    def optimize_4imgs_lenstronomy(self, lens_systems=None, datatofit=None, macromodel=None, realizations=None, multiplane=None, source_shape='GAUSSIAN',
                                   source_size_kpc=None, grid_res = None, tol_source=1e-5, tol_mag = 0.2, tol_centroid = 0.05, centroid_0=[0, 0],
                                   n_particles = 50, n_iterations = 250, polar_grid = False,
                                   optimize_routine = 'fixed_powerlaw_shear', verbose=False, re_optimize=False,
                                   particle_swarm = True, restart=1,
                                   constrain_params=None, pso_convergence_mean=5000,
                                   pso_compute_magnification=200, tol_simplex_params=5e-3, tol_simplex_func = 0.05,
                                   simplex_n_iter=300, LOS_mass_sheet_front = 7.7, LOS_mass_sheet_back = 8,
                                   chi2_mode='source', tol_image = 0.005, satellites=None):


        if source_shape is None:
            source_shape = default_source_shape

        if source_size_kpc is None:
            source_size_kpc = default_source_size_kpc

        if grid_res is None:
            grid_res = default_res(source_size_kpc)

        if lens_systems is None:

            lens_systems = []

            if isinstance(macromodel,list):

                assert len(macromodel) == len(realizations), 'if macromodel is a list, must have same number of elements as realizations'

                for macro,real in zip(macromodel,realizations):
                    lens_systems.append(self.build_system(main=macro, realization=real, multiplane=multiplane,LOS_mass_sheet_front=LOS_mass_sheet_front,
                                                          LOS_mass_sheet_back=LOS_mass_sheet_back, satellites=satellites))
            else:
                if realizations is not None:
                    for real in realizations:
                        lens_systems.append(self.build_system(main=macromodel, realization=real,
                                                              multiplane=multiplane,LOS_mass_sheet_front=LOS_mass_sheet_front,
                                                          LOS_mass_sheet_back=LOS_mass_sheet_back, satellites=satellites))
                else:
                    lens_systems.append(self.build_system(main=copy.deepcopy(macromodel),multiplane=multiplane, LOS_mass_sheet_front=LOS_mass_sheet_front,
                                                          LOS_mass_sheet_back=LOS_mass_sheet_back, satellites=satellites))


        optimized_data, model, _, _ = self._optimize_4imgs_lenstronomy(lens_systems, data2fit=datatofit, tol_source=tol_source,
                                                                 tol_mag=tol_mag, tol_centroid=tol_centroid, centroid_0=centroid_0,
                                                                 n_particles=n_particles, n_iterations=n_iterations,
                                                                 res=grid_res, source_shape=source_shape,
                                                                 source_size_kpc=source_size_kpc, polar_grid=polar_grid,
                                                                 optimizer_routine=optimize_routine, verbose=verbose, re_optimize=re_optimize,
                                                                 particle_swarm=particle_swarm, restart=restart,
                                                                 constrain_params=constrain_params, pso_convergence_mean=pso_convergence_mean,
                                                                 pso_compute_magnification=pso_compute_magnification,
                                                                 tol_simplex_params=tol_simplex_params, tol_simplex_func = tol_simplex_func,
                                                                 simplex_n_iter=simplex_n_iter, chi2_mode = chi2_mode, tol_image = tol_image)

        return optimized_data,model

    def solve_lens_equation(self, full_system=None, macromodel=None, realizations=None, multiplane=None, method=None,
                            ray_trace=True, identifier=None, srcx=None, srcy=None,  res=None,
                            source_shape='GAUSSIAN', source_size_kpc=None, sort_by_pos=None, filter_subhalos=False,
                            filter_by_pos=False, filter_kwargs={}, raytrace_with=None, polar_grid=False, shr_coords=1,
                            brightimg=True, LOS_mass_sheet_back = 6, LOS_mass_sheet_front = 6, centroid_0 = [0, 0],
                            satellites = None):

        if raytrace_with is None:
            raytrace_with = raytrace_with_default

        if source_shape is None:
            source_shape = default_source_shape

        if source_size_kpc is None:
            source_size_kpc = default_source_size_kpc

        if res is None:
            res = default_res(source_size_kpc)

        if method is None:
            method = default_solve_method

        if identifier is None:
            identifier = default_file_identifier

        lens_systems = []

        if full_system is None:
            assert macromodel is not None
            if realizations is not None:
                for real in realizations:
                    lens_systems.append(self.build_system(main=macromodel, realization=real, multiplane=multiplane, LOS_mass_sheet_back = LOS_mass_sheet_back,
                                                    LOS_mass_sheet_front = LOS_mass_sheet_front, satellites=satellites))
            else:
                lens_systems.append(self.build_system(main=copy.deepcopy(macromodel), realization=None, multiplane=multiplane,LOS_mass_sheet_back = LOS_mass_sheet_back,
                                                    LOS_mass_sheet_front = LOS_mass_sheet_front, satellites=satellites))

        else:

            lens_systems.append(copy.deepcopy(full_system))

        assert method is not None
        assert method in ['lensmodel', 'lenstronomy']

        data = self._solve_4imgs(lens_systems=lens_systems, method=method, identifier=identifier, srcx=srcx + centroid_0[0],
                                 srcy=srcy + centroid_0[1],
                                 res=res, source_shape=source_shape, ray_trace=ray_trace,
                                 raytrace_with=raytrace_with, source_size_kpc=source_size_kpc, polar_grid=polar_grid,
                                 shr_coords=shr_coords,brightimg=brightimg)

        if sort_by_pos is not None:
            data[0].sort_by_pos(sort_by_pos.x,sort_by_pos.y)
        return data


    def two_step_optimize(self, macromodel=None, datatofit=None, realizations=None, multiplane=False, method=None, ray_trace=None, sigmas=None,
                          identifier=None, srcx=None, srcy=None, res=None,
                          source_shape='GAUSSIAN', source_size=None, print_mag=False, raytrace_with=None,
                          filter_by_position=False, polar_grid=False, filter_kwargs={},solver_type='PROFILE_SHEAR',
                          N_iter_max=100,shr_coords=1):

        # optimizes the macromodel first, then uses it to optimize with additional halos in the lens model

        assert datatofit is not None

        if sigmas is None:
            sigmas = default_sigmas

        if raytrace_with is None:
            raytrace_with = raytrace_with_default

        if source_size is None:
            source_size = default_source_size_kpc

        if source_shape is None:
            source_shape = default_source_shape

        if res is None:
            res = default_res(source_size)

        if method is None:
            method = default_solve_method

        if identifier is None:
            identifier = default_file_identifier

        if method == 'lensmodel':
            _,macromodel_init = self.macromodel_initialize(macromodel=copy.deepcopy(macromodel), datatofit=datatofit,
                                                           multiplane=multiplane, method='lensmodel', sigmas=sigmas,
                                                           identifier=identifier, res=res,
                                                           source_shape=source_shape, source_size=source_size, print_mag=print_mag,
                                                           shr_coords=shr_coords)
            optimized_data, newsystem = self.fit(macromodel=macromodel_init, datatofit=datatofit,
                                                 realizations=realizations, multiplane=multiplane,
                                                 ray_trace=ray_trace, sigmas=sigmas, identifier=identifier, srcx=srcx,
                                                 srcy=srcy,  res=res,method='lensmodel',
                                                 source_shape=source_shape, source_size=source_size,
                                                 print_mag=print_mag, raytrace_with=raytrace_with,
                                                 filter_by_position=filter_by_position, polar_grid=polar_grid,
                                                 filter_kwargs=filter_kwargs,shr_coords=shr_coords)
        else:

            optimized_data, newsystem = self.fit(macromodel=macromodel, datatofit=datatofit,
                                                 realizations=realizations, multiplane=multiplane, method=method,
                                                 ray_trace=ray_trace, sigmas=sigmas, identifier=identifier, srcx=srcx,
                                                 srcy=srcy, res=res,
                                                 source_shape=source_shape, source_size=source_size,
                                                 print_mag=print_mag, raytrace_with=raytrace_with,
                                                 filter_by_position=filter_by_position, polar_grid=polar_grid,
                                                 filter_kwargs=filter_kwargs, solver_type=solver_type,N_iter_max=N_iter_max)




        return optimized_data,newsystem

    def macromodel_initialize(self, macromodel, datatofit, multiplane, method=None, sigmas=None,
                              identifier=None,  res=None,
                              source_shape='GAUSSIAN', source_size=None, print_mag=False,
                              solver_type=None,shr_coords=1):

        # fits just a single macromodel profile to the data

        assert method is not None
        assert method in ['lensmodel','lenstronomy']

        if sigmas is None:
            sigmas = default_sigmas

        if method is None:
            method = default_solve_method

        lens_systems = []
        assert macromodel is not None

        lens_systems.append(self.build_system(main=copy.deepcopy(macromodel), realization=None, multiplane=multiplane))

        optimized_data, model = self._optimize_4imgs_lensmodel(lens_systems=lens_systems, data2fit=datatofit, method=method,
                                                               sigmas=sigmas, identifier=identifier,
                                                               res=res, source_shape=source_shape, ray_trace=False,
                                                               source_size_kpc=source_size, print_mag=print_mag, opt_routine='randomize',
                                                               solver_type=solver_type, shr_coords=shr_coords)

        newmacromodel = model[0].lens_components[0]

        return optimized_data,newmacromodel

    def fit(self, macromodel=None, datatofit=None, realizations=None, multiplane = None, method=None, ray_trace=True, sigmas=None,
                      identifier=None, srcx=None, srcy=None, res=None,
                      source_shape='GAUSSIAN', source_size=None, print_mag=False, raytrace_with=None, filter_by_position=False,
                      polar_grid=False,filter_kwargs={},which_chi = 'src',solver_type='PROFILE_SHEAR',N_iter_max=100,shr_coords=1):

        # uses source plane chi^2


        if which_chi == 'src':
            basic_or_full = 'basic'
        else:
            basic_or_full = 'full'
        if raytrace_with is None:
            raytrace_with = raytrace_with_default

        if source_shape is None:
            source_shape = default_source_shape

        if source_size is None:
            source_size = default_source_size_kpc

        if res is None:
            res = default_res(source_size)

        if method is None:
            method = default_solve_method

        if sigmas is None:
            sigmas = default_sigmas

        if method is None:
            method = default_solve_method

        if identifier is None:
            identifier = default_file_identifier

        lens_systems= []

        ################################################################################

        # If macromodel is a list same length as realizations, build the systems and fit each one

        if isinstance(macromodel,list):

            assert len(macromodel) == len(realizations), 'if macromodel is a list, must have same number of elements as realizations'

            for macro,real in zip(macromodel,realizations):
                lens_systems.append(self.build_system(main=macro, realization=real, multiplane=multiplane))
        else:
            if realizations is not None:
                for real in realizations:
                    lens_systems.append(self.build_system(main=copy.deepcopy(macromodel), realization=real,
                                                          multiplane=multiplane))
            else:
                lens_systems.append(self.build_system(main=copy.deepcopy(macromodel),multiplane=multiplane))

        if method == 'lenstronomy':

            optimized_data, model = self.optimize_4imgs_lenstronomy(lens_systems=lens_systems, datatofit=datatofit,
                                                                    multiplane = multiplane, res=res, source_shape=source_shape,
                                                                    source_size_kpc=source_size, raytrace_with=raytrace_with, polar_grid=polar_grid, initialize = False,
                                                                    solver_type=solver_type, n_particles=40, n_iterations=200, tol_source=1e-3, tol_centroid=0.05,
                                                                    tol_mag=0.2, centroid_0=[0,0])

        else:
            optimized_data, model = self._optimize_4imgs_lensmodel(lens_systems=lens_systems, data2fit=datatofit,
                                                                   method=method,
                                                                   sigmas=sigmas, identifier=identifier,
                                                                   res=res, source_shape=source_shape,
                                                                   ray_trace=ray_trace,
                                                                   source_size_kpc=source_size, print_mag=print_mag,
                                                                   opt_routine=basic_or_full,
                                                                   raytrace_with=raytrace_with, polar_grid=polar_grid,
                                                                   solver_type=solver_type, shr_coords=shr_coords)

        return optimized_data,model

    def _trace(self, lens_system, data=None, multiplane=None,res=None, source_shape=None, source_size=None,
               raytrace_with=None, srcx=None, srcy=None, build=False, realizations=None):

        if raytrace_with is None:
            raytrace_with = raytrace_with_default

        if source_shape is None:
            source_shape = default_source_shape

        if source_size is None:
            source_size = default_source_size_kpc

        if res is None:
            res = default_res(source_size)

        if build:
            lens_system = self.build_system(main=lens_system, realization=realizations, multiplane=multiplane)

        if raytrace_with == 'lensmodel':

            xsrc,ysrc = data.srcx,data.srcy

            if srcx is not None:
                xsrc = srcx
            if srcy is not None:
                ysrc = srcy

            fluxes = self.do_raytrace_lensmodel(lens_system=lens_system,xpos=data.x,ypos=data.y,xsrc=xsrc,ysrc=ysrc,
                                    multiplane=multiplane,res=res,source_shape=source_shape,source_size=source_size,
                                       cosmology=self.cosmo,zsrc=self.cosmo.zsrc,raytrace_with=raytrace_with)

        elif raytrace_with == 'lenstronomy':

            lenstronomy_wrap = LenstronomyWrap(multiplane=multiplane,cosmo=self.cosmo,z_source=self.cosmo.zsrc)
            lenstronomy_wrap.assemble(lens_system)

            fluxes = self.do_raytrace_lenstronomy(lenstronomy_wrap,data.x,data.y,multiplane=multiplane,res=res,
                                         source_shape=source_shape,source_size=source_size,cosmology=self.cosmo.cosmo,zsrc=self.cosmo.zsrc)

        return fluxes*max(fluxes)**-1







