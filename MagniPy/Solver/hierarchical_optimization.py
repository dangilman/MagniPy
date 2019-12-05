from MagniPy.util import chi_square_img
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
                tol_simplex_func, simplex_n_iter, solver_class, LOS_mass_sheet_front, LOS_mass_sheet_back, centroid, satellites,
                        check_foreground_fit, foreground_aperture_masses, foreground_globalmin_masses, foreground_filters, \
    reoptimize_scale, particle_swarm_reopt):

    #foreground_aperture_masses, foreground_globalmin_masses, foreground_filters, \
    #reoptimize_scale, particle_swarm_reopt = foreground_mass_filters(m_ref, LOS_mass_sheet_front)

    source_x, source_y = 0, 0

    for h in range(0, len(foreground_filters)):

        optimizer_kwargs = {'save_background_path': True, 're_optimize_scale': reoptimize_scale[h]}

        if h == 0:

            realization_filtered = realizations[0].filter(datatofit.x, datatofit.y, mindis_front=foreground_filters[h],
                                                          mindis_back=1, source_x = source_x, source_y=source_y,
                                                          logmasscut_front=foreground_globalmin_masses[h],
                                                          logabsolute_mass_cut_front = foreground_aperture_masses[h],
                                                          logmasscut_back=12,
                                                          logabsolute_mass_cut_back=12, centroid = centroid, zmax=solver_class.zmain)
            if verbose:
                print('initial optimization')

        else:

            macromodel = model[0].lens_components[0]
            re_optimize = True
            particle_swarm = particle_swarm_reopt[h]
            optimizer_kwargs.update({'re_optimize_scale': reoptimize_scale[h]})

            real = realizations[0].filter(datatofit.x, datatofit.y, mindis_front=foreground_filters[h],
                                          source_x = source_x, source_y=source_y,
                          logmasscut_front=foreground_globalmin_masses[h], logmasscut_back=12, ray_x=out_kwargs['path_x'],
                                          ray_y=out_kwargs['path_y'], logabsolute_mass_cut_back=12,
                                          path_redshifts=out_kwargs['path_redshifts'],
                                          path_Tzlist=out_kwargs['path_Tzlist'],
                                          logabsolute_mass_cut_front = foreground_aperture_masses[h], zmax=solver_class.zmain)

            if verbose:
                print('optimization '+str(h+1))

            realization_filtered = real.join(realization_filtered)

        N_foreground_halos = len(realization_filtered.masses[np.where(realization_filtered.redshifts <= solver_class.zmain)])

        if verbose:
            print('nhalos: ', len(realization_filtered.halos))
            print('aperture size: ', foreground_filters[h])
            print('minimum mass in aperture: ', foreground_aperture_masses[h])
            print('minimum global mass: ', foreground_globalmin_masses[h])
            print('N foreground halos: ', N_foreground_halos)

        do_optimization = True
        if h > 0:
            if N_foreground_halos == 0:
                do_optimization = False
            if N_foreground_halos == N_foreground_halos_last:
                do_optimization = False

        if do_optimization:

            lens_system = solver_class.build_system(main=macromodel, realization=realization_filtered, multiplane=True,
                                                    LOS_mass_sheet_front=LOS_mass_sheet_front,
                                                    LOS_mass_sheet_back=LOS_mass_sheet_back, satellites=satellites)

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
                                                                                           finite_source_magnification=False,
                                                                                           chi2_mode='source', adaptive_grid=False)

            source_x, source_y = keywords_lensmodel['source_x'], keywords_lensmodel['source_y']
            foreground_rays = out_kwargs['precomputed_rays']
            foreground_macromodel = model[0].lens_components[0]
            N_foreground_halos_last = N_foreground_halos

        else:
            model[0].realization = realization_filtered
            foreground_macromodel = model[0].lens_components[0]

        if check_foreground_fit:

            if chi_square_img(datatofit.x, datatofit.y, optimized_data[0].x, optimized_data[0].y, 0.003) >= 1:
                return None, None, None, None, None

    return foreground_rays, foreground_macromodel, [realization_filtered], keywords_lensmodel, optimized_data

def optimize_background(macromodel, realization_foreground, realization_background, foreground_rays, source_x, source_y, datatofit, tol_source, tol_mag, tol_centroid, centroid_0, n_particles,
                        n_iterations, source_shape, source_size_kpc, polar_grid, optimizer_routine, re_optimize, verbose,
                        particle_swarm, restart, constrain_params, pso_convergence_mean, pso_compute_magnification,
                        tol_simplex_params, tol_simplex_func, simplex_n_iter, solver_class,
                        background_globalmin_masses = None, background_aperture_masses = None, background_filters = None,
                        reoptimize_scale = None, optimize_iteration = None, particle_swarm_reopt = None,
                        LOS_mass_sheet_front = 7.7, LOS_mass_sheet_back = 8, centroid = None, satellites = None,
                        ):

    assert len(background_filters) == len(background_aperture_masses)
    assert len(background_globalmin_masses) == len(background_filters)

    N_iterations = len(background_filters)
    backx, backy, background_Tzs, background_zs, reoptimized_realizations = [], [], [], [], []

    for h in range(0, N_iterations):
        if verbose:
            print('iterating ' + str(h + 1) + ' of ' + str(N_iterations) + '... ')

        if h == 0:

            if verbose:
                print('initial background optimization')

            optimizer_args = {'save_background_path': True,
                                're_optimize_scale': reoptimize_scale[h],
                                'precomputed_rays': foreground_rays}

            filtered_background = realization_background.filter(datatofit.x, datatofit.y, mindis_front=0,
                                          mindis_back=background_filters[h], logmasscut_front=12,
                                          logabsolute_mass_cut_front = 12, source_x=source_x, source_y=source_y,
                                          logmasscut_back=background_globalmin_masses[h],
                                          logabsolute_mass_cut_back=background_aperture_masses[h],
                                                                zmin=solver_class.zmain)

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
                                          source_x=source_x, source_y=source_y, logmasscut_back=background_globalmin_masses[h], ray_x=path_x,
                                          ray_y=path_y, logabsolute_mass_cut_back=background_aperture_masses[h],
                                          path_redshifts=path_redshifts,
                                          path_Tzlist=path_Tzlist,
                                          logabsolute_mass_cut_front=12, zmin=solver_class.zmain)

            realization_filtered = realization_filtered.join(filtered_background)

        # realization_filtered = realizations[0].realization_from_indicies(np.squeeze(filter_indicies))
        N_background_halos = len(realization_filtered.masses[np.where(realization_filtered.redshifts > solver_class.zmain)])
        if verbose:
            print('N foreground halos: ', len(realization_filtered.masses[np.where(realization_filtered.redshifts <= solver_class.zmain)]))
            print('N background halos: ', N_background_halos)
            print('N total: ', len(realization_filtered.masses))\

            print('aperture size: ', background_filters[h])
            print('minimum mass in aperture: ', background_aperture_masses[h])
            print('minimum global mass: ', background_globalmin_masses[h])

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

            lens_system = solver_class.build_system(main=macromodel, realization=realization_filtered,
                                                    multiplane=True, LOS_mass_sheet_front = LOS_mass_sheet_front,
                                                    LOS_mass_sheet_back = LOS_mass_sheet_back, satellites=satellites)

            optimized_data, model, out_kwargs, keywords_lensmodel = solver_class._optimize_4imgs_lenstronomy([lens_system],
                                                                 data2fit=datatofit,tol_source=tol_source,
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
                                                                                       finite_source_magnification=False,
                                                                                       chi2_mode='source', adaptive_grid=False)

            path_x, path_y, path_redshifts, path_Tzlist = out_kwargs['path_x'], out_kwargs['path_y'], \
                                                          out_kwargs['path_redshifts'], out_kwargs[
                                                              'path_Tzlist']
            source_x, source_y = keywords_lensmodel['source_x'], keywords_lensmodel['source_y']
            backx.append(path_x)
            backy.append(path_y)
            background_Tzs.append(path_Tzlist)
            background_zs.append(path_redshifts)
            reoptimized_realizations.append(realization_filtered)
            N_background_halos_last = N_background_halos

    else:
        model[0].realization = realization_filtered
        reoptimized_realizations.append(realization_filtered)


    return optimized_data, model, \
           (backx, backy, background_Tzs, background_zs, reoptimized_realizations), keywords_lensmodel

def foreground_mass_filters(realization, LOS_mass_sheet):

    nhalos = len(realization.halos)

    if nhalos <= 500:

        foreground_aperture_masses = [LOS_mass_sheet, 0]
        foreground_globalmin_masses = [LOS_mass_sheet, LOS_mass_sheet]
        foreground_filters = [10, 0.4]
        reoptimize_scale = [1, 0.5]
        particle_swarm_reopt = [True, False]

    else:

        foreground_aperture_masses = [LOS_mass_sheet, 7, 0]
        foreground_globalmin_masses = [LOS_mass_sheet, LOS_mass_sheet, LOS_mass_sheet]
        foreground_filters = [10, 0.3, 0.1]
        reoptimize_scale = [1, 0.5, 0.5]
        particle_swarm_reopt = [True, True, False]

    return foreground_aperture_masses, foreground_globalmin_masses, foreground_filters, \
           reoptimize_scale, particle_swarm_reopt

def background_mass_filters(realization, LOS_mass_sheet):

    rung_0_mass = LOS_mass_sheet
    rung_0_window = 10

    background_aperture_masses = [rung_0_mass]
    background_globalmin_masses = [rung_0_mass]
    background_filters = [rung_0_window]
    reoptimize_scale = [0.4]
    particle_swarm_reopt = [True]
    optimize_iteration = [True]

    rung_1_mass = 7.5
    rung_2_mass = 7
    rung_3_mass = 0
    rung_1_window = 0.4
    rung_2_window = 0.3
    rung_3_window = 0.075

    nhalos_large = np.sum(realization.masses > 10**7)

    if nhalos_large > 150:
        background_aperture_masses += [rung_1_mass, rung_2_mass, rung_3_mass]
        background_globalmin_masses += [rung_0_mass, rung_0_mass, rung_0_mass]
        background_filters += [rung_1_window, rung_2_window, rung_3_window]
        reoptimize_scale += [0.5, 0.4, 0.15]
        particle_swarm_reopt += [False, False, False]
        optimize_iteration += [True, True, False]

    else :
        background_aperture_masses += [rung_2_mass, rung_3_mass]
        background_globalmin_masses += [rung_0_mass, rung_0_mass]
        background_filters += [rung_1_window, rung_2_window]
        reoptimize_scale += [0.4, 0.5]
        particle_swarm_reopt += [False, False]
        optimize_iteration += [True, False]


    return background_aperture_masses, background_globalmin_masses, background_filters, \
    reoptimize_scale, particle_swarm_reopt, optimize_iteration

