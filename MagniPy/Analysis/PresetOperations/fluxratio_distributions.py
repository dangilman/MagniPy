from MagniPy.util import *
from MagniPy.Solver.solveroutines import SolveRoutines
from MagniPy.LensBuild.defaults import *
from MagniPy.paths import *
from MagniPy.lensdata import Data
from pyHalo.pyhalo import pyHalo
from copy import copy
from time import time

def compute_fluxratio_distributions(halo_model='', model_args={},
                                    data2fit=None, Ntotal=int, outfilename='', zlens=None, zsrc=None,
                                    start_macromodel=None, identifier=None, res=None, sigmas=None,
                                    source_size_kpc=None, write_to_file=False, filter_halo_positions=False, outfilepath=None, method='lenstronomy',
                                    mindis_front=0.5, mindis_back=0.3, logmcut_back=None, logmcut_front=None,
                                    n_restart=1, pso_conv_mean = 100,
                                    srcx = 0, srcy = 0, use_source=True, hierarchical = True, grid_res = 0.002,
                                    LOS_mass_sheet = None, multiplane = True, **kwargs):
    tstart = time()

    data = Data(x=data2fit[0],y=data2fit[1],m=data2fit[2],t=data2fit[3],source=[srcx, srcy])

    if write_to_file:
        assert outfilepath is not None
        assert os.path.exists(outfilepath)

    if start_macromodel is None:
        start_macromodel = get_default_SIE(zlens)
        start_macromodel.redshift = zlens
    if sigmas is None:
        sigmas = default_sigmas

    if start_macromodel is None:
        start_macromodel = get_default_SIE(zlens)
        start_macromodel.redshift = zlens
    if sigmas is None:
        sigmas = default_sigmas

    if res is None:
        res = default_res(source_size_kpc)

    solver = SolveRoutines(zlens=zlens, zsrc=zsrc, temp_folder=identifier)

    if multiplane is None:
        if halo_model == 'main_lens':
            multiplane = False
        elif halo_model == 'line_of_sight':
            multiplane = True
        elif halo_model == 'delta_LOS':
            multiplane = True
        elif halo_model == 'composite_powerlaw':
            multiplane = True
    else:
        print('over-riding default multiplane status, using: ', multiplane)

    # initialize macromodel
    fit_fluxes = []
    shears, shear_pa, xcen, ycen = [], [], [], []
    n = 0
    init_macromodel = None

    pyhalo = pyHalo(zlens,zsrc)

    if method=='lenstronomy':

        while len(fit_fluxes)<Ntotal:

            Nreal = Ntotal - len(fit_fluxes)

            #print(str(len(fit_fluxes)) +' of '+str(Ntotal))

            realizations = pyhalo.render(halo_model,model_args)

            if hierarchical:
                filter_halo_positions = False
            if filter_halo_positions:
                if use_source:
                    use_real = list(real.filter(data.x, data.y, mindis_front = mindis_front, mindis_back = mindis_back,
                             logmasscut_front = logmcut_front, logmasscut_back = logmcut_back, source_x = data.srcx,
                                       source_y = data.srcy) for real in realizations)
                else:
                    use_real = list(real.filter(data.x, data.y, mindis_front=mindis_front, mindis_back=mindis_back,
                                                logmasscut_front=logmcut_front, logmasscut_back=logmcut_back) for real in realizations)
            else:
                use_real = realizations

            if hierarchical and multiplane is True:

                model_data, system, _ = solver.hierarchical_optimization(datatofit=data, macromodel=start_macromodel,
                                                                       realizations=use_real,
                                                                       multiplane=multiplane, n_particles=20,
                                                                       simplex_n_iter=400, n_iterations=300,
                                                                       source_size_kpc=source_size_kpc,
                                                                       optimize_routine='fixed_powerlaw_shear',
                                                                       verbose=True, pso_convergence_mean=pso_conv_mean,
                                                                       re_optimize=False, particle_swarm=True,
                                                                         pso_compute_magnification=1000,
                                                                       restart=n_restart, grid_res=grid_res,
                                                                         LOS_mass_sheet = LOS_mass_sheet, **kwargs)

            else:

                model_data, system = solver.optimize_4imgs_lenstronomy(datatofit=data, macromodel=start_macromodel,
                                                                       realizations=use_real,
                                                                       multiplane=multiplane, n_particles=20,
                                                                       simplex_n_iter=400, n_iterations=300,
                                                                       pso_compute_magnification=1000,
                                                                       source_size_kpc=source_size_kpc,
                                                                       optimize_routine='fixed_powerlaw_shear',
                                                                       verbose=True, pso_convergence_mean=pso_conv_mean,
                                                                       re_optimize=False, particle_swarm=True,
                                                                       restart=n_restart, grid_res=grid_res,
                                                                       LOS_mass_sheet=LOS_mass_sheet)

            for sys,dset in zip(system,model_data):

                if dset.nimg != data.nimg:
                    continue

                astro_error = chi_square_img(data.x,data.y,dset.x,dset.y,0.003,reorder=False)

                if astro_error > 2:
                    continue

                fit_fluxes.append(dset.flux_anomaly(data, sum_in_quad=False, index=0))
                shears.append(sys.lens_components[0].shear)
                xcen.append(sys.lens_components[0].lenstronomy_args['center_x'])
                ycen.append(sys.lens_components[0].lenstronomy_args['center_y'])
                shear_pa.append(sys.lens_components[0].shear_theta)

        write_fluxes(fluxratio_data_path+identifier + 'fluxes_'+outfilename+'.txt', fit_fluxes, summed_in_quad=False)
        tend = time()
        runtime = (tend - tstart)*60**-1
        with open(fluxratio_data_path+identifier +'runtime_'+outfilename+'.txt', 'a') as f:
            f.write(str(np.round(runtime, 2))+'\n')
