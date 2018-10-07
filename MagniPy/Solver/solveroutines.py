from MagniPy.magnipy import Magnipy
from MagniPy.LensBuild.defaults import raytrace_with_default,default_sigmas,default_res,\
    default_source_shape,default_source_size_kpc,default_solve_method,default_file_identifier,get_default_SIE_random
import copy
import numpy as np
from MagniPy.Solver.LenstronomyWrap.lenstronomy_wrap import LenstronomyWrap

class SolveRoutines(Magnipy):

    """
    This class uses the routines set up in MagniPy to solve the lens equation in various ways with lenstronomy or lensmodel
    """

    def hierarchical_optimization(self, datatofit=None, macromodel=None, realizations=None, multiplane=None, source_shape='GAUSSIAN',
                                  source_size_kpc=None, grid_res = None, tol_source=1e-5, tol_mag = 0.2, tol_centroid = 0.05, centroid_0=[0, 0],
                                  n_particles = 50, n_iterations = 250, polar_grid = False,
                                  optimize_routine = 'fixed_powerlaw_shear', verbose=False, re_optimize=False,
                                  particle_swarm = True, solver_type = 'PROFILE_SHEAR', restart=1,
                                  constrain_params=None, pso_convergence_mean=5000,
                                  pso_compute_magnification=200, tol_simplex_params=1e-3, tol_simplex_func = 0.01,
                                  simplex_n_iter=300, background_globalmin_masses = None,
                                  background_aperture_masses = None, background_filters = None,
                                  mindis_front = 0.5, logmasscut_front = 7.5, min_mass = 6, m_break = 0,
                                  particle_swarm_reopt = True, save_all_paths=True):

        if background_globalmin_masses is None or background_aperture_masses is None:
            m_ref = max(m_break, min_mass)
            if m_ref < 7:
                background_aperture_masses = [10, 8, 7.5, 0]
                background_globalmin_masses = [10, 9, 8, 8]
                background_filters = [0.5, 0.5, 0.2, 0.03]
                reoptimize_scale = [1, 1, 0.3, 0.2]
                particle_swarm_reopt = [True, True, False, False]

            elif m_ref < 8:
                background_aperture_masses = [9, 8, 0]
                background_globalmin_masses = [9, 9, 8]
                background_filters = [0.5, 0.25, 0.025]
                reoptimize_scale = [1, 1, 0.2]
                particle_swarm_reopt = [True, True, False]

            else:
                background_aperture_masses = [9, 0]
                background_globalmin_masses = [9, 8]
                background_filters = [0.5, 0.2]
                reoptimize_scale = [1, 0.2]
        else:
            reoptimize_scale = np.ones_like(background_filters)

        if not isinstance(particle_swarm_reopt, list):
            particle_swarm_reopt = [particle_swarm_reopt] * len(background_filters)
        else:
            assert len(particle_swarm_reopt) == len(background_filters)

        raytrace_with = raytrace_with_default

        if source_shape is None:
            source_shape = default_source_shape

        if source_size_kpc is None:
            source_size_kpc = default_source_size_kpc

        if grid_res is None:
            grid_res = default_res(source_size_kpc)

        assert len(background_filters) == len(background_aperture_masses)
        assert len(background_globalmin_masses) == len(background_filters)
        N_iterations = len(background_filters)
        backx, backy, background_Tzs, background_zs, reoptimized_realizations = [], [], [], [], []

        for h in range(0, N_iterations):

            if h == N_iterations - 1:
                finite_source_magnification = True
            else:
                finite_source_magnification = False

            print('iterating '+str(h+1)+' of '+str(N_iterations)+'... ')
            if h == 0:

                optimizer_kwargs = {'save_background_path':True, 're_optimize_scale': reoptimize_scale[h]}

                realization_filtered = realizations[0].filter(datatofit.x, datatofit.y, mindis_front = mindis_front,
                                                              mindis_back = background_filters[h], logmasscut_front = logmasscut_front,
                                                              logmasscut_back = background_aperture_masses[h],
                                                              logabsolute_mass_cut = background_globalmin_masses[h])

            else:
                macromodel = model[0].lens_components[0]
                re_optimize = True
                particle_swarm = particle_swarm_reopt[h]
                optimizer_kwargs = {'re_optimize_scale': reoptimize_scale[h]}
                if h == N_iterations - 1 and save_all_paths is False:
                    optimizer_kwargs.update({'save_background_path': False})
                else:
                    optimizer_kwargs.update({'save_background_path': True})

                real = realizations[0].filter(datatofit.x, datatofit.y, mindis_front=mindis_front,
                                                              mindis_back=background_filters[h], logmasscut_front=logmasscut_front,
                                                              logmasscut_back=background_globalmin_masses[h],
                                                              ray_x=x_background, ray_y=y_background,
                                                              logabsolute_mass_cut = background_aperture_masses[h],
                                                              background_redshifts = background_redshifts, Tzlist_background = Tzlist)

                realization_filtered = real.join(realization_filtered)

            #realization_filtered = realizations[0].realization_from_indicies(np.squeeze(filter_indicies))
            print('N background halos: ', len(realization_filtered.masses[np.where(realization_filtered.redshifts > self.zmain)]))
            #print(macromodel.lenstronomy_args)
            lens_system = self.build_system(main=macromodel, realization=realization_filtered, multiplane=multiplane)

            optimized_data, model, optimizer_kwargs = self._optimize_4imgs_lenstronomy([lens_system], data2fit=datatofit, tol_source=tol_source,
                                                                     tol_mag=tol_mag, tol_centroid=tol_centroid, centroid_0=centroid_0,
                                                                     n_particles=n_particles, n_iterations=n_iterations,
                                                                     res=grid_res, source_shape=source_shape,
                                                                     source_size_kpc=source_size_kpc,
                                                                     raytrace_with=raytrace_with, polar_grid=polar_grid, solver_type=solver_type,
                                                                     optimizer_routine=optimize_routine, verbose=verbose, re_optimize=re_optimize,
                                                                     particle_swarm=particle_swarm, restart=restart,
                                                                     constrain_params=constrain_params,
                                                                     pso_convergence_mean=pso_convergence_mean,
                                                                     pso_compute_magnification=pso_compute_magnification,
                                                                     tol_simplex_params=tol_simplex_params, tol_simplex_func = tol_simplex_func,
                                                                     simplex_n_iter=simplex_n_iter, optimizer_kwargs = optimizer_kwargs,
                                                                     finite_source_magnification = finite_source_magnification)

            if save_all_paths is True:
                x_background, y_background, background_redshifts, Tzlist = optimizer_kwargs['x_background'], \
                                  optimizer_kwargs['y_background'], optimizer_kwargs['background_redshifts'], \
                                                                                 optimizer_kwargs['Tz_list_background']
                backx.append(x_background)
                backy.append(y_background)
                background_Tzs.append(Tzlist)
                background_zs.append(background_redshifts)
                reoptimized_realizations.append(realization_filtered)

        return optimized_data,model, (backx, backy, background_Tzs, background_zs, reoptimized_realizations)

    def optimize_4imgs_lenstronomy(self, lens_systems=None, datatofit=None, macromodel=None, realizations=None, multiplane=None, source_shape='GAUSSIAN',
                                   source_size_kpc=None, grid_res = None, tol_source=1e-5, tol_mag = 0.2, tol_centroid = 0.05, centroid_0=[0, 0],
                                   n_particles = 50, n_iterations = 250, polar_grid = False,
                                   optimize_routine = 'fixed_powerlaw_shear', verbose=False, re_optimize=False,
                                   particle_swarm = True, solver_type = 'PROFILE_SHEAR', restart=1,
                                   constrain_params=None, shifting_background=False, pso_convergence_mean=5000,
                                   pso_compute_magnification=200, tol_simplex_params=1e-3, tol_simplex_func = 0.01,
                                   simplex_n_iter=300):


        raytrace_with = raytrace_with_default

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
                    lens_systems.append(self.build_system(main=macro, realization=real, multiplane=multiplane))
            else:
                if realizations is not None:
                    for real in realizations:
                        lens_systems.append(self.build_system(main=macromodel, realization=real,
                                                              multiplane=multiplane))
                else:
                    lens_systems.append(self.build_system(main=copy.deepcopy(macromodel),multiplane=multiplane))


        optimized_data, model, _ = self._optimize_4imgs_lenstronomy(lens_systems, data2fit=datatofit, tol_source=tol_source,
                                                                 tol_mag=tol_mag, tol_centroid=tol_centroid, centroid_0=centroid_0,
                                                                 n_particles=n_particles, n_iterations=n_iterations,
                                                                 res=grid_res, source_shape=source_shape,
                                                                 source_size_kpc=source_size_kpc,
                                                                 raytrace_with=raytrace_with, polar_grid=polar_grid, solver_type=solver_type,
                                                                 optimizer_routine=optimize_routine, verbose=verbose, re_optimize=re_optimize,
                                                                 particle_swarm=particle_swarm, restart=restart,
                                                                 constrain_params=constrain_params,
                                                                 shifting_background=shifting_background, pso_convergence_mean=pso_convergence_mean,
                                                                 pso_compute_magnification=pso_compute_magnification,
                                                                 tol_simplex_params=tol_simplex_params, tol_simplex_func = tol_simplex_func,
                                                                 simplex_n_iter=simplex_n_iter)

        return optimized_data,model

    def solve_lens_equation(self, full_system=None, macromodel=None, realizations=None, multiplane=None, method=None,
                            ray_trace=True, identifier=None, srcx=None, srcy=None,  res=None,
                            source_shape='GAUSSIAN', source_size_kpc=None, sort_by_pos=None, filter_subhalos=False,
                            filter_by_pos=False, filter_kwargs={}, raytrace_with=None, polar_grid=True, shr_coords=1,
                            brightimg=True):

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
                    lens_systems.append(self.build_system(main=macromodel, realization=real, multiplane=multiplane))
            else:
                lens_systems.append(self.build_system(main=copy.deepcopy(macromodel), realization=None, multiplane=multiplane))

        else:

            lens_systems.append(copy.deepcopy(full_system))

        assert method is not None
        assert method in ['lensmodel', 'lenstronomy']

        data = self._solve_4imgs(lens_systems=lens_systems, method=method, identifier=identifier, srcx=srcx, srcy=srcy,
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
                      polar_grid=True,filter_kwargs={},which_chi = 'src',solver_type='PROFILE_SHEAR',N_iter_max=100,shr_coords=1):

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







