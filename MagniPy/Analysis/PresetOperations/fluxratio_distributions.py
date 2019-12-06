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
                                    start_macromodel=None, identifier=None, satellites=None,
                                    source_size_kpc=None, write_to_file=False, outfilepath=None,
                                    n_restart=1, pso_conv_mean = 100,
                                    source_x = 0, source_y = 0, grid_res = 0.002,
                                    LOS_mass_sheet_front = 7.7,
                                    LOS_mass_sheet_back = 8,
                                    multiplane=True, **kwargs):
    tstart = time()

    data = Data(x=data2fit[0],y=data2fit[1],m=data2fit[2],t=data2fit[3],source=[source_x, source_y])

    if write_to_file:
        assert outfilepath is not None
        assert os.path.exists(outfilepath)

    if start_macromodel is None:
        start_macromodel = get_default_SIE(zlens)
        start_macromodel.redshift = zlens

    if start_macromodel is None:
        start_macromodel = get_default_SIE(zlens)
        start_macromodel.redshift = zlens

    solver = SolveRoutines(zlens=zlens, zsrc=zsrc, temp_folder=identifier)

    # initialize macromodel
    fit_fluxes = []
    shears, shear_pa, xcen, ycen = [], [], [], []

    pyhalo = pyHalo(zlens,zsrc)

    while len(fit_fluxes)<Ntotal:

        #print(str(len(fit_fluxes)) +' of '+str(Ntotal))
        print('rendering... ')
        realization = pyhalo.render(halo_model,model_args)[0]
        print('done.')
        realization = realization.shift_background_to_source(source_x, source_y)

        model_data, system, outputs, _ = solver.hierarchical_optimization(datatofit=data, macromodel=start_macromodel,
                                                                   realization=realization,
                                                                   multiplane=multiplane, n_particles=20,
                                                                   simplex_n_iter=400, n_iterations=300,
                                                                   source_size_kpc=source_size_kpc,
                                                                   optimize_routine='fixed_powerlaw_shear',
                                                                   verbose=True, pso_convergence_mean=pso_conv_mean,
                                                                   re_optimize=False, particle_swarm=True, tol_mag=None,
                                                                     pso_compute_magnification=700,
                                                                   restart=n_restart, grid_res=grid_res,
                                                                     LOS_mass_sheet_front = LOS_mass_sheet_front,
                                                                     LOS_mass_sheet_back = LOS_mass_sheet_back, satellites=satellites,
                                                                     **kwargs)

        for sys,dset in zip(system,model_data):

            if dset.nimg != data.nimg:
                continue

            astro_error = chi_square_img(data.x,data.y,dset.x,dset.y,0.003,reorder=False)

            if astro_error > 1:
                continue

            fit_fluxes.append(dset.flux_anomaly(data, sum_in_quad=False, index=0))

    write_fluxes(fluxratio_data_path+identifier + 'fluxes_'+outfilename+'.txt', fit_fluxes, summed_in_quad=False)
    tend = time()
    runtime = (tend - tstart)*60**-1
    with open(fluxratio_data_path+identifier +'runtime_'+outfilename+'.txt', 'a') as f:
        f.write(str(np.round(runtime, 2))+'\n')
