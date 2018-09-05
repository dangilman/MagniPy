from MagniPy.util import *
from MagniPy.Solver.solveroutines import SolveRoutines
from MagniPy.LensBuild.defaults import *
from MagniPy.paths import *
from MagniPy.lensdata import Data
from pyHalo.pyhalo import pyHalo
from copy import copy

def compute_fluxratio_distributions(halo_model='', model_args={},
                                    data2fit=None, Ntotal=int, outfilename='', zlens=None, zsrc=None,
                                    start_macromodel=None, identifier=None, res=None, sigmas=None,
                                    source_size_kpc=None, raytrace_with='lenstronomy', test_only=False, write_to_file=False,
                                    filter_halo_positions=False, outfilepath=None, ray_trace=True, method='lenstronomy',
                                    start_shear=0.05, mindis_front=0.5, mindis_back=0.3, log_masscut_low=7,
                                    single_background=False):

    data = Data(x=data2fit[0],y=data2fit[1],m=data2fit[2],t=data2fit[3],source=None)

    if write_to_file:
        assert outfilepath is not None
        assert os.path.exists(outfilepath)

    if start_macromodel is None:
        start_macromodel = get_default_SIE(zlens)
        start_macromodel.redshift = zlens
    if sigmas is None:
        sigmas = default_sigmas

    halo_generator = pyHalo(zlens, zsrc)

    if start_macromodel is None:
        start_macromodel = get_default_SIE(zlens)
        start_macromodel.redshift = zlens
    if sigmas is None:
        sigmas = default_sigmas

    if res is None:
        res = default_res(source_size_kpc)

    solver = SolveRoutines(zlens=zlens, zsrc=zsrc, temp_folder=identifier)

    if halo_model == 'main_lens':
        multiplane = False
    elif halo_model == 'line_of_sight':
        multiplane = True
    elif halo_model == 'delta_LOS':
        multiplane = True
    elif halo_model == 'composite_powerlaw':
        multiplane = True

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

            if filter_halo_positions:
                use_real = list(real.filter(data.x, data.y, mindis_front = mindis_front, mindis_back = mindis_back,
                             logmasscut_front = log_masscut_low, logmasscut_back = 8, back_scale_z = 0) for real in realizations)
            else:
                use_real = realizations

            if init_macromodel is None:
                _, init = solver.optimize_4imgs_lenstronomy(datatofit=data,macromodel=start_macromodel,realizations=None,
                                   multiplane=multiplane,n_particles = 50, n_iterations = 300,
                                   optimize_routine = 'fixed_powerlaw_shear',verbose=False,
                                         re_optimize=False, particle_swarm=True,restart=3)

            model_data, system = solver.optimize_4imgs_lenstronomy(datatofit=data,macromodel=start_macromodel,realizations=use_real,
                                   multiplane=multiplane,n_particles = 50, n_iterations = 300,source_size_kpc=source_size_kpc,
                                   optimize_routine = 'fixed_powerlaw_shear',verbose=True,
                                         re_optimize=False, particle_swarm=True, restart=2,
                                           single_background=single_background, init_system = init[0])

            for sys,dset in zip(system,model_data):

                if dset.nimg != data.nimg:
                    continue

                astro_error = chi_square_img(data.x,data.y,dset.x,dset.y,0.003,reorder=False)

                if astro_error > 4:
                    continue

                fit_fluxes.append(dset.flux_anomaly(data, sum_in_quad=False, index=0))
                shears.append(sys.lens_components[0].shear)
                xcen.append(sys.lens_components[0].lenstronomy_args['center_x'])
                ycen.append(sys.lens_components[0].lenstronomy_args['center_y'])
                shear_pa.append(sys.lens_components[0].shear_theta)

