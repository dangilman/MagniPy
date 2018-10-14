import inspect
import numpy as np

def split_realization(datatofit, realization):

    foreground = realization.filter(datatofit.x, datatofit.y, mindis_front=10,
                                                  mindis_back=0, logmasscut_front=0,
                                                  logabsolute_mass_cut_front=0,
                                                  logmasscut_back=20,
                                                  logabsolute_mass_cut_back=20)

    background = realization.filter(datatofit.x, datatofit.y, mindis_front=0,
                                    mindis_back=10, logmasscut_front=20,
                                    logabsolute_mass_cut_front=20,
                                    logmasscut_back=0,
                                    logabsolute_mass_cut_back=0)

    return foreground, background

def optimize_foreground(macromodel, realizations, datatofit,tol_source,tol_mag, tol_centroid, centroid_0, n_particles, n_iterations,
                 source_shape, source_size_kpc, polar_grid, optimizer_routine, re_optimize, verbose, particle_swarm, restart, constrain_params, pso_convergence_mean, pso_compute_magnification, tol_simplex_params,
                tol_simplex_func, simplex_n_iter, m_ref, solver_class, LOS_mass_sheet):

    foreground_aperture_masses, foreground_globalmin_masses, foreground_filters, \
    reoptimize_scale, particle_swarm_reopt = foreground_mass_filters(m_ref)

    for h in range(0, len(foreground_filters)):

        optimizer_kwargs = {'save_background_path': True, 're_optimize_scale': reoptimize_scale[h]}

        if h == 0:

            realization_filtered = realizations[0].filter(datatofit.x, datatofit.y, mindis_front=foreground_filters[h],
                                                          mindis_back=1, logmasscut_front=foreground_globalmin_masses[h],
                                                          logabsolute_mass_cut_front = foreground_aperture_masses[h],
                                                          logmasscut_back=12,
                                                          logabsolute_mass_cut_back=12)

        else:

            macromodel = model[0].lens_components[0]
            re_optimize = True
            particle_swarm = particle_swarm_reopt[h]
            optimizer_kwargs.update({'re_optimize_scale': reoptimize_scale[h]})

            real = realizations[0].filter(datatofit.x, datatofit.y, mindis_front=foreground_filters[h],
                          logmasscut_front=foreground_globalmin_masses[h], logmasscut_back=12, ray_x=out_kwargs['path_x'],
                                          ray_y=out_kwargs['path_y'], logabsolute_mass_cut_back=12,
                                          path_redshifts=out_kwargs['path_redshifts'],
                                          path_Tzlist=out_kwargs['path_Tzlist'],
                                          logabsolute_mass_cut_front = foreground_aperture_masses[h])

            realization_filtered = real.join(realization_filtered)

        N_foreground_halos = len(realization_filtered.masses[np.where(realization_filtered.redshifts <= solver_class.zmain)])

        if verbose:
            print('N foreground halos: ', N_foreground_halos)

        do_optimization = True
        if h > 0:
            if N_foreground_halos == 0:
                do_optimization = False
            if N_foreground_halos == N_foreground_halos_last:
                do_optimization = False

        if do_optimization:

            lens_system = solver_class.build_system(main=macromodel, realization=realization_filtered, multiplane=True, LOS_mass_sheet=LOS_mass_sheet)

            optimized_data, model, out_kwargs, keywords_lensmodel = solver_class._optimize_4imgs_lenstronomy([lens_system],
                                                                                             data2fit=datatofit,
                                                                                           tol_source=tol_source,
                                                                                           tol_mag=tol_mag,
                                                                                           tol_centroid=tol_centroid,
                                                                                           centroid_0=centroid_0,
                                                                                           n_particles=n_particles,
                                                                                           n_iterations=n_iterations,
                                                                                           source_shape=source_shape,
                                                                                           source_size_kpc=source_size_kpc,
                                                                                           return_ray_path=True,
                                                                                           polar_grid=polar_grid,
                                                                                           optimizer_routine=optimizer_routine,
                                                                                           verbose=verbose,
                                                                                           re_optimize=re_optimize,
                                                                                           particle_swarm=particle_swarm,
                                                                                           restart=restart,
                                                                                           constrain_params=constrain_params,
                                                                                           pso_convergence_mean=pso_convergence_mean,
                                                                                           pso_compute_magnification=pso_compute_magnification,
                                                                                           tol_simplex_params=tol_simplex_params,
                                                                                           tol_simplex_func=tol_simplex_func,
                                                                                           simplex_n_iter=simplex_n_iter,
                                                                                           optimizer_kwargs=optimizer_kwargs,
                                                                                           finite_source_magnification=False)

            foreground_rays = out_kwargs['precomputed_rays']
            foreground_macromodel = model[0].lens_components[0]
            N_foreground_halos_last = N_foreground_halos

    return foreground_rays, foreground_macromodel, [realization_filtered], keywords_lensmodel

def optimize_background(macromodel, realization_foreground, realization_background, foreground_rays, datatofit, tol_source, tol_mag, tol_centroid, centroid_0, n_particles,
                        n_iterations, source_shape, source_size_kpc, polar_grid, optimizer_routine, re_optimize, verbose,
                        particle_swarm, restart, constrain_params, pso_convergence_mean, pso_compute_magnification,
                        tol_simplex_params, tol_simplex_func, simplex_n_iter, m_ref, solver_class,
                        background_globalmin_masses = None, background_aperture_masses = None, background_filters = None,
                        reoptimize_scale = None, particle_swarm_reopt = None, LOS_mass_sheet = True):

    if background_globalmin_masses is None or background_aperture_masses is None:

        background_aperture_masses, background_globalmin_masses, background_filters, \
        reoptimize_scale, particle_swarm_reopt, optimize_iteration = background_mass_filters(m_ref)
    else:
        assert len(background_filters) == len(background_aperture_masses)
        assert len(background_globalmin_masses) == len(background_filters)

    N_iterations = len(background_filters)
    backx, backy, background_Tzs, background_zs, reoptimized_realizations = [], [], [], [], []

    for h in range(0, N_iterations):
        if verbose:
            print('iterating ' + str(h + 1) + ' of ' + str(N_iterations) + '... ')

        if h == 0:

            optimizer_args = {'save_background_path': True,
                                're_optimize_scale': reoptimize_scale[h],
                                'precomputed_rays': foreground_rays}

            filtered_background = realization_background.filter(datatofit.x, datatofit.y, mindis_front=0,
                                          mindis_back=background_filters[h], logmasscut_front=12,
                                          logabsolute_mass_cut_front = 12,
                                          logmasscut_back=background_globalmin_masses[h],
                                          logabsolute_mass_cut_back=background_aperture_masses[h])

            realization_filtered = realization_foreground.join(filtered_background)

        else:

            macromodel = model[0].lens_components[0]
            re_optimize = True
            particle_swarm = particle_swarm_reopt[h]
            optimizer_args.update({'re_optimize_scale': reoptimize_scale[h]})
            #optimizer_args.update({'save_background_path': True})
            #optimizer_args.update({'precomputed_rays': foreground_rays})

            filtered_background = realization_background.filter(datatofit.x, datatofit.y, mindis_front=0,
                                          logmasscut_front=12, mindis_back = background_filters[h],
                                          logmasscut_back=background_globalmin_masses[h], ray_x=path_x,
                                          ray_y=path_y, logabsolute_mass_cut_back=background_aperture_masses[h],
                                          path_redshifts=path_redshifts,
                                          path_Tzlist=path_Tzlist,
                                          logabsolute_mass_cut_front=12)

            realization_filtered = realization_filtered.join(filtered_background)

        # realization_filtered = realizations[0].realization_from_indicies(np.squeeze(filter_indicies))
        N_background_halos = len(realization_filtered.masses[np.where(realization_filtered.redshifts > solver_class.zmain)])
        if verbose:
            print('N foreground halos: ', len(realization_filtered.masses[np.where(realization_filtered.redshifts <= solver_class.zmain)]))
            print('N background halos: ', N_background_halos)
            print('N total: ', len(realization_filtered.masses))

        do_optimization = True

        if h > 0:
            if N_background_halos == 0:
                do_optimization = False
            if N_background_halos == N_background_halos_last:
                do_optimization = False
        #print(optimize_iteration[h])
        if optimize_iteration[h] is False:
            do_optimization = False

        if do_optimization:
            # print(macromodel.lenstronomy_args)
            lens_system = solver_class.build_system(main=macromodel, realization=realization_filtered, multiplane=True,
                                                    LOS_mass_sheet = LOS_mass_sheet)

            optimized_data, model, out_kwargs, keywords_lensmodel = solver_class._optimize_4imgs_lenstronomy([lens_system],
                                                                 data2fit=datatofit,
                                                                                       tol_source=tol_source,
                                                                                       tol_mag=tol_mag,
                                                                                       tol_centroid=tol_centroid,
                                                                                       centroid_0=centroid_0,
                                                                                       n_particles=n_particles,
                                                                                       n_iterations=n_iterations,
                                                                                       source_shape=source_shape,
                                                                                       source_size_kpc=source_size_kpc,
                                                                                       return_ray_path=True,
                                                                                       polar_grid=polar_grid,
                                                                                       optimizer_routine=optimizer_routine,
                                                                                       verbose=verbose,
                                                                                       re_optimize=re_optimize,
                                                                                       particle_swarm=particle_swarm,
                                                                                       restart=restart,
                                                                                       constrain_params=constrain_params,
                                                                                       pso_convergence_mean=pso_convergence_mean,
                                                                                       pso_compute_magnification=pso_compute_magnification,
                                                                                       tol_simplex_params=tol_simplex_params,
                                                                                       tol_simplex_func=tol_simplex_func,
                                                                                       simplex_n_iter=simplex_n_iter,
                                                                                       optimizer_kwargs=optimizer_args,
                                                                                       finite_source_magnification=False)

            path_x, path_y, path_redshifts, path_Tzlist = out_kwargs['path_x'], out_kwargs['path_y'], \
                                                          out_kwargs['path_redshifts'], out_kwargs[
                                                              'path_Tzlist']
            backx.append(path_x)
            backy.append(path_y)
            background_Tzs.append(path_Tzlist)
            background_zs.append(path_redshifts)
            reoptimized_realizations.append(realization_filtered)
            N_background_halos_last = N_background_halos

    return optimized_data, model, \
           (backx, backy, background_Tzs, background_zs, reoptimized_realizations), keywords_lensmodel

def foreground_mass_filters(m_ref):

    if m_ref < 7:
        foreground_aperture_masses = [8, 7, 0]
        foreground_globalmin_masses = [8, 8, 8]
        foreground_filters = [0.5, 0.1, 0.05]
        reoptimize_scale = [0.5, 0.5, 0.5]
        particle_swarm_reopt = [True, True, False]
    else:
        foreground_aperture_masses = [0]
        foreground_globalmin_masses = [8]
        foreground_filters = [0.5]
        reoptimize_scale = [1]
        particle_swarm_reopt = [True]

    return foreground_aperture_masses, foreground_globalmin_masses, foreground_filters, \
           reoptimize_scale, particle_swarm_reopt

def background_mass_filters(m_ref):

    rung_0_mass = 7.7
    rung_0_window = 10

    background_aperture_masses = [rung_0_mass]
    background_globalmin_masses = [rung_0_mass]
    background_filters = [rung_0_window]
    reoptimize_scale = [0.5]
    particle_swarm_reopt = [True]
    optimize_iteration = [True]

    rung_1_mass = 7.5
    rung_2_mass = 6
    rung_3_mass = 0
    rung_1_window = 0.25
    rung_2_window = 0.02
    rung_3_window = 0.01

    if m_ref < 6:
        background_aperture_masses += [rung_1_mass, rung_2_mass, rung_3_mass]
        background_globalmin_masses += [rung_0_mass, rung_0_mass, rung_0_mass]
        background_filters += [rung_1_window, rung_2_window, rung_3_window]
        reoptimize_scale += [0.5, 0.05, 0.05]
        particle_swarm_reopt += [True, False, False]
        optimize_iteration += [True, False, False]

    elif m_ref < 7.5:
        background_aperture_masses += [rung_1_mass, rung_2_mass]
        background_globalmin_masses += [rung_0_mass, rung_0_mass]
        background_filters += [rung_1_window, rung_2_window]
        reoptimize_scale += [0.5, 0.05]
        particle_swarm_reopt += [True, False]
        optimize_iteration += [True, False]

    elif m_ref < 8:
        background_aperture_masses += [rung_1_mass, rung_2_mass]
        background_globalmin_masses += [rung_0_mass, rung_0_mass]
        background_filters += [rung_1_window, rung_2_window]
        reoptimize_scale += [1, 0.2]
        particle_swarm_reopt += [True, False]
        optimize_iteration += [True, True]

    else:
        background_aperture_masses += [0]
        background_globalmin_masses += [rung_0_mass]
        background_filters += [0.3]
        reoptimize_scale += [0.2]
        particle_swarm_reopt += [False]
        optimize_iteration += [True]

    return background_aperture_masses, background_globalmin_masses, background_filters, \
    reoptimize_scale, particle_swarm_reopt, optimize_iteration


def build_kwargs(lens_system,data2fit,tol_source,tol_mag, tol_centroid, centroid_0, n_particles, n_iterations, res,
                 source_shape, source_size_kpc, return_ray_path, polar_grid, optimizer_routine, verbose, re_optimize,
                 particle_swarm, restart, constrain_params, pso_convergence_mean, pso_compute_magnification, tol_simplex_params,
                tol_simplex_func, simplex_n_iter, optimizer_kwargs, finite_source_magnification):

    arg_list = {}

    argspec = inspect.getargvalues(inspect.currentframe())

    for pname in argspec.args:
        arg_list.update({pname: argspec.locals[pname]})

    return arg_list
