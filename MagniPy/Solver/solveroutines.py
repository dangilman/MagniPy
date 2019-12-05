from MagniPy.magnipy import Magnipy
from MagniPy.LensBuild.defaults import raytrace_with_default,default_sigmas,default_res,\
    default_source_shape,default_source_size_kpc,default_solve_method,default_file_identifier,get_default_SIE_random
import copy
import numpy as np
from MagniPy.Solver.LenstronomyWrap.lenstronomy_wrap import LenstronomyWrap
from MagniPy.Solver.hierarchical_optimization import *
from MagniPy.util import chi_square_img

class SolveRoutines(Magnipy):

    """
    This class uses the routines set up in MagniPy to solve the lens equation in various ways with lenstronomy or lensmodel
    """

    def hierarchical_optimization(self, datatofit=None, macromodel=None, realization=None, multiplane=True,
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
                                  min_mass=6, m_break=0, LOS_mass_sheet_front = 7.7, LOS_mass_sheet_back = 8, satellites=None,
                                  adaptive_grid=False, grid_rmax_scale=1, check_foreground_fit=False,
                                  foreground_aperture_masses=None,foreground_globalmin_masses=None,foreground_filters=None,
                                  reoptimize_scale_filters=None, particle_swarm_reopt_filters=None,
                                  reoptimize_scale_background=None,particle_swarm_reopt_background=None,optimize_iteration_background=None):

        if source_shape is None:
            source_shape = default_source_shape
        if source_size_kpc is None:
            source_size_kpc = default_source_size_kpc
        if grid_res is None:
            grid_res = default_res(source_size_kpc)

        assert multiplane is True

        if centroid_0[0] != 0 or centroid_0[1] != 0:
            realization = realization.shift_centroid(centroid_0)

        assert isinstance(source_size_kpc, float) or isinstance(source_size_kpc, int)

        foreground_realization, background_realization = split_realization(datatofit, realization)

        if verbose: print('optimizing foreground... ')

        if foreground_aperture_masses is not None:
            assert foreground_globalmin_masses is not None
            assert foreground_filters is not None
            assert reoptimize_scale_filters is not None
            assert particle_swarm_reopt_filters is not None
            assert len(foreground_aperture_masses) == len(foreground_globalmin_masses)
            assert len(reoptimize_scale_filters) == len(foreground_globalmin_masses)
            assert len(particle_swarm_reopt_filters) == len(reoptimize_scale_filters)

        else:
            foreground_aperture_masses, foreground_globalmin_masses, foreground_filters, \
            reoptimize_scale_filters, particle_swarm_reopt_filters = foreground_mass_filters(foreground_realization, LOS_mass_sheet_front)


        foreground_rays, foreground_macromodel, foreground_halos, keywords_lensmodel, data_foreground = optimize_foreground(macromodel,
                              [foreground_realization], datatofit, tol_source, tol_mag, tol_centroid, centroid_0,  n_particles,
                              n_iterations, source_shape, source_size_kpc, polar_grid, optimize_routine, re_optimize, verbose, particle_swarm,
                          restart, constrain_params, pso_convergence_mean, pso_compute_magnification, tol_simplex_params,
                            tol_simplex_func, simplex_n_iter, self, LOS_mass_sheet_front, LOS_mass_sheet_back,
                                  centroid_0, satellites, check_foreground_fit,foreground_aperture_masses, foreground_globalmin_masses,
                                                                                                                            foreground_filters, reoptimize_scale_filters, particle_swarm_reopt_filters)

        if data_foreground is None:
            return None, None, None

        if check_foreground_fit:
            if chi_square_img(datatofit.x, datatofit.y, data_foreground[0].x, data_foreground[0].y, 0.003) >= 1:
                return None, None, None

        #if np.sum(realization.redshifts > self.zmain) == 0:
        #    return data_foreground, foreground_macromodel, None, keywords_lensmodel

        if verbose: print('optimizing background... ')

        if background_aperture_masses is not None:
            assert background_globalmin_masses is not None
            assert background_filters is not None
            assert reoptimize_scale_background is not None
            assert particle_swarm_reopt_background is not None
            assert optimize_iteration_background is not None
            assert len(background_aperture_masses) == len(background_globalmin_masses)
            assert len(reoptimize_scale_background) == len(background_aperture_masses)
            assert len(particle_swarm_reopt_background) == len(optimize_iteration_background)

        else:
            background_aperture_masses, background_globalmin_masses, background_filters, \
            reoptimize_scale_background, particle_swarm_reopt_background, optimize_iteration_background = background_mass_filters(background_realization,
                                                                                             LOS_mass_sheet_back)

        optimized_data, model, outputs, keywords_lensmodel = optimize_background(foreground_macromodel, foreground_halos[0], background_realization, foreground_rays,
                      keywords_lensmodel['source_x'], keywords_lensmodel['source_y'], datatofit, tol_source, tol_mag, tol_centroid, centroid_0, n_particles,  n_iterations,
                      source_shape, source_size_kpc, polar_grid, optimize_routine, re_optimize, verbose,
                        particle_swarm, restart, constrain_params, pso_convergence_mean, pso_compute_magnification,
                        tol_simplex_params, tol_simplex_func, simplex_n_iter, self,
                        background_globalmin_masses = background_globalmin_masses,
                         background_aperture_masses = background_aperture_masses, background_filters = background_filters,
                        reoptimize_scale = reoptimize_scale_background, particle_swarm_reopt = particle_swarm_reopt_background,
                         LOS_mass_sheet_front = LOS_mass_sheet_front, LOS_mass_sheet_back = LOS_mass_sheet_back,
                          optimize_iteration=optimize_iteration_background, centroid = centroid_0, satellites=satellites)

        fluxes = self._ray_trace_finite(optimized_data[0].x, optimized_data[0].y, optimized_data[0].srcx, optimized_data[0].srcy, True,
                               keywords_lensmodel['lensModel'], keywords_lensmodel['kwargs_lens'], grid_res, source_shape,
                               source_size_kpc, polar_grid, adaptive_grid, grid_rmax_scale)

        optimized_data[0].set_mag(fluxes)
        optimized_data[0].sort_by_pos(datatofit.x,datatofit.y)

        if satellites is not None:

            if 'position_convention' in satellites.keys() and \
                    satellites['position_convention'] == 'lensed':

                physical_kwargs = keywords_lensmodel['lensModel']._full_lensmodel.\
                    lens_model._convention(keywords_lensmodel['kwargs_lens'])
                physical_x, physical_y = physical_kwargs[2]['center_x'], physical_kwargs[2]['center_y']

                model[0]._satellite_physical_location = np.array([physical_x, physical_y])
            else:
                model[0]._satellite_physical_location = np.array([keywords_lensmodel['kwargs_lens'][2]['center_x'],
                                                          keywords_lensmodel['kwargs_lens'][2]['center_y']])

        return optimized_data, model, outputs, keywords_lensmodel

    def optimize_4imgs_lenstronomy(self, lens_systems=None, datatofit=None, macromodel=None, realization=None, multiplane=None, source_shape='GAUSSIAN',
                                   source_size_kpc=None, grid_res = None, tol_source=1e-5, tol_mag = 0.2, tol_centroid = 0.05, centroid_0=[0, 0],
                                   n_particles = 50, n_iterations = 250, polar_grid = False,
                                   optimize_routine = 'fixed_powerlaw_shear', verbose=False, re_optimize=False,
                                   particle_swarm = True, restart=1,
                                   constrain_params=None, pso_convergence_mean=5000,
                                   pso_compute_magnification=200, tol_simplex_params=5e-3, tol_simplex_func = 0.05,
                                   simplex_n_iter=300, LOS_mass_sheet_front = 7.7, LOS_mass_sheet_back = 8,
                                   chi2_mode='source', tol_image = 0.005, satellites=None, use_finite_source=True,
                                   adaptive_grid=False, grid_rmax_scale=1):


        if source_shape is None:
            source_shape = default_source_shape

        if source_size_kpc is None:
            source_size_kpc = default_source_size_kpc

        if grid_res is None:
            grid_res = default_res(source_size_kpc)

        if lens_systems is None:

            lens_systems = []

            if realization is not None:
                lens_systems.append(self.build_system(main=macromodel, realization=realization,
                                                          multiplane=multiplane,LOS_mass_sheet_front=LOS_mass_sheet_front,
                                                      LOS_mass_sheet_back=LOS_mass_sheet_back, satellites=satellites))
            else:

                lens_systems.append(self.build_system(main=copy.deepcopy(macromodel),multiplane=multiplane, LOS_mass_sheet_front=LOS_mass_sheet_front,
                                                      LOS_mass_sheet_back=LOS_mass_sheet_back, satellites=satellites))


        optimized_data, model, _, info = self._optimize_4imgs_lenstronomy(lens_systems, data2fit=datatofit, tol_source=tol_source,
                                                                 tol_mag=tol_mag, tol_centroid=tol_centroid, centroid_0=centroid_0,
                                                                 n_particles=n_particles, n_iterations=n_iterations,
                                                                 res=grid_res, source_shape=source_shape,
                                                                 source_size_kpc=source_size_kpc, polar_grid=polar_grid,
                                                                 optimizer_routine=optimize_routine, verbose=verbose, re_optimize=re_optimize,
                                                                 particle_swarm=particle_swarm, restart=restart,
                                                                 constrain_params=constrain_params, pso_convergence_mean=pso_convergence_mean,
                                                                 pso_compute_magnification=pso_compute_magnification,
                                                                 tol_simplex_params=tol_simplex_params, tol_simplex_func = tol_simplex_func,
                                                                 simplex_n_iter=simplex_n_iter, chi2_mode = chi2_mode, tol_image = tol_image,
                                                                       finite_source_magnification=use_finite_source,
                                                                       adaptive_grid=adaptive_grid, grid_rmax_scale=grid_rmax_scale)


        return optimized_data, model, info

    def solve_lens_equation(self, full_system=None, macromodel=None, realizations=None, multiplane=None, method=None,
                            ray_trace=True, identifier=None, srcx=None, srcy=None,  res=None,
                            source_shape='GAUSSIAN', source_size_kpc=None, sort_by_pos=None, filter_subhalos=False,
                            filter_by_pos=False, filter_kwargs={}, raytrace_with=None, polar_grid=False, shr_coords=1,
                            brightimg=True, LOS_mass_sheet_back = 6, LOS_mass_sheet_front = 6, centroid_0 = [0, 0],
                            satellites = None, adaptive_grid=False):

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
                                 shr_coords=shr_coords,brightimg=brightimg, adaptive_grid=adaptive_grid)

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







