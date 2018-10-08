def background_mass_filters(m_ref):

    background_aperture_masses = [10]
    background_globalmin_masses = [10]
    background_filters = [1]
    reoptimize_scale = [1]
    particle_swarm_reopt = [True]

    if m_ref < 7:
        background_aperture_masses += [8.5, 7.5, 0]
        background_globalmin_masses += [8.5, 8, 8]
        background_filters += [0.5, 0.15, 0.01]
        reoptimize_scale += [1, 0.3, 0.2]
        particle_swarm_reopt += [True, False, False]

    elif m_ref < 7.5:
        background_aperture_masses += [7.5, 0]
        background_globalmin_masses += [8, 8]
        background_filters += [0.4, 0.03]
        reoptimize_scale += [0.2, 0.2]
        particle_swarm_reopt += [True, False, False]

    elif m_ref < 8:
        background_aperture_masses += [8, 0]
        background_globalmin_masses += [8, 8]
        background_filters += [0.5, 0.025]
        reoptimize_scale += [1, 0.2]
        particle_swarm_reopt += [True, False]

    else:
        background_aperture_masses += [0]
        background_globalmin_masses += [8]
        background_filters += [0.3]
        reoptimize_scale += [0.2]
        particle_swarm_reopt += [False]

    return background_aperture_masses, background_globalmin_masses, background_filters, \
    reoptimize_scale, particle_swarm_reopt


