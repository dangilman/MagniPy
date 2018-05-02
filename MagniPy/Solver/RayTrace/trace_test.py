if True:
    from MagniPy.Solver.solveroutines import *
    from MagniPy.LensBuild.lens_assemble import Deflector
    from MagniPy.MassModels.SIE import SIE
    from MagniPy.MassModels.NFW import NFW
    import matplotlib.pyplot as plt
    from MagniPy.Solver.LenstronomyWrap.kwargs_translate import gravlens_to_lenstronomy


    solver = SolveRoutines(zmain=0.5,zsrc=1.5,clean_up=False,temp_folder='raytrace_test')
    srcx,srcy = 0.055,0.01
    identifier='LOSeffect'

    subhalox,subhaloy = 0.92,0.4
    subhalomass = 10**9.5

    startkwargs = {'R_ein':0.6,'ellip':0.5,'ellip_theta':-50,'x':0,'y':0}
    start = Deflector(subclass=SIE(),tovary=True,varyflags=['1','1','1','1','1','0','0','0','0','0'],redshift=0.5,**startkwargs)

    truthkwargs = {'R_ein':1,'ellip':0.2,'ellip_theta':-80,'x':0,'y':0}
    truth = Deflector(subclass=SIE(),tovary=True,varyflags=['1','1','1','1','1','0','0','0','0','0'],redshift=0.5,**truthkwargs)

    halokwargs1 = {'mass':subhalomass,'x':subhalox,'y':subhaloy,'mhm':0}
    halokwargs2 = {'R_ein':0.01,'x':subhalox,'y':subhaloy,'ellip':0.01,'ellip_theta':0}

    halos = []
    zvals = [0.05]
    for z in zvals:
        halos.append([Deflector(subclass=NFW(z=z,zsrc=1.5),redshift=z,**halokwargs1)])
        #halos.append([Deflector(subclass=SIE(), redshift=z, **halokwargs2)])

    realizations = halos
    multiplane = True

    data = solver.solve_lens_equation(macromodel=truth, multiplane=multiplane, realizations=None,
                                       method='lensmodel', ray_trace=False,
                                       source_size=0.0012 * 2.3 ** -1, srcx=srcx, srcy=srcy, identifier=identifier,
                                       grid_rmax=0.05)

    test_params = True

    if test_params:
        datafit1, lensmodel_lens = solver.two_step_optimize(macromodel=start, realizations=realizations, datatofit=data[0],
                                                            multiplane=multiplane,
                                                            method='lensmodel', ray_trace=True,
                                                            raytrace_with='lenstronomy',
                                                            identifier=identifier,
                                                            grid_rmax=.08, res=.0005, source_shape='GAUSSIAN',
                                                            source_size=0.0012 * 2.4 ** -1, solver_type='PROFILE')
        datafit_ptsource, lensmodel_lens = solver.two_step_optimize(macromodel=start, realizations=realizations,
                                                            datatofit=data[0],
                                                            multiplane=multiplane,
                                                            method='lensmodel', ray_trace=True,
                                                            identifier=identifier,raytrace_with='lensmodel',
                                                            grid_rmax=.08, res=.0005, source_shape='GAUSSIAN',
                                                            source_size=0.0012 * 2.4 ** -1, solver_type='PROFILE')

        print datafit1[0].m
        print datafit_ptsource[0].m


        #print solver.do_raytrace(lensmodel_lens[0], datafit1[0], multiplane=multiplane, raytrace_with='lensmodel')

    fit = False
    if fit:

        datafit1, lensmodel_lens = solver.two_step_optimize(macromodel=start, realizations=halos, datatofit=data[0],
                                              multiplane=multiplane,
                                              method='lensmodel', ray_trace=True, raytrace_with='lenstronomy',
                                              identifier=identifier,
                                              grid_rmax=.08, res=.0005, source_shape='GAUSSIAN',
                                              source_size=0.0012 * 2.4 ** -1, solver_type='PROFILE_SHEAR')

        datafit2, lenstronomy_lens = solver.two_step_optimize(macromodel=start, realizations=halos, datatofit=data[0],
                                              multiplane=multiplane,
                                              method='lenstronomy', ray_trace=True, raytrace_with='lenstronomy',
                                              identifier=identifier,
                                              grid_rmax=.08, res=.0005, source_shape='GAUSSIAN',
                                              source_size=0.0012 * 2.4 ** -1, solver_type='PROFILE')

        datafit3, _ = solver.two_step_optimize(macromodel=start, realizations=halos, datatofit=data[0],
                                                       multiplane=multiplane,
                                                       method='lensmodel', ray_trace=False, identifier=identifier,
                                                       grid_rmax=.08, res=.0005, source_shape='GAUSSIAN',
                                                       source_size=0.0012 * 2.4 ** -1, solver_type='PROFILE_SHEAR')

        print datafit1[0].compute_flux_ratios()
        print datafit2[0].compute_flux_ratios()
        print datafit3[0].compute_flux_ratios()
        #print data[0].compute_flux_ratios()

    solveLEQ = False
    if solveLEQ:
        data1 = solver.solve_lens_equation(macromodel=truth,multiplane=multiplane,realizations=halos,
                                          method='lensmodel',ray_trace=True,raytrace_with='lensmodel',
                                          source_size=0.0012*2.3**-1,srcx=srcx,srcy=srcy,identifier=identifier,grid_rmax=0.05)


        data2 = solver.solve_lens_equation(macromodel=truth, multiplane=multiplane,realizations=halos,
                                          method='lensmodel', ray_trace=True, raytrace_with='lenstronomy',
                                          source_size=0.0012 * 2.3 ** -1, srcx=srcx, srcy=srcy, identifier=identifier,
                                          grid_rmax=0.05,sort_by_pos=data[0])

        print data1[0].m
        print data2[0].m
        print data[0].m



