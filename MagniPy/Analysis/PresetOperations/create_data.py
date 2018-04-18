import numpy as np
from halo_constructor import Realization
from MagniPy.LensBuild.lens_assemble import Deflector
from MagniPy.MassModels.SIE import SIE
from MagniPy.Solver.solveroutines import SolveRoutines
from MagniPy.util import identify

def create_data(identifier='create_data',config=None,b_prior=[1,0.2],ellip_prior=[.2,.05],shear_prior=[0.05,0.01],ePA_prior=[-90,90],
                sPA_prior=[-90,90],gamma_prior=None,zlens=None,zsrc=None,substructure_model_args={},source_size=0.0012*2.355**-1,massprofile='TNFW',
                raytrace_with='lenstronomy',method='lenstronomy',halo_model='',multiplane=False,solver_class=None,ray_trace=True,subhalo_realizations=None,astrometric_perturbation=0.003,return_gamma=True):

    run = True

    realization = Realization(zlens=zlens,zsrc=zsrc)

    if config=='cross':
        target = 0
    elif config=='fold':
        target = 1
    elif config == 'cusp':
        target = 2
    else:
        raise Exception('config must be one of cross, cusp, fold')

    if solver_class is None:
        solver = SolveRoutines(zmain=zlens,zsrc=zsrc,temp_folder=identifier)
    else:
        solver = solver_class

    while run:

        if config=='cusp':
            r = np.random.uniform(0.03**2,0.07**2,1)
        else:
            r = np.random.uniform(0, .07 ** 2, 1)

        theta = np.random.uniform(0, np.pi * 2, 1)
        src_x, src_y = float(r ** .5 * np.cos(theta)), float(r ** .5 * np.sin(theta))

        R_ein = np.absolute(np.random.normal(b_prior[0], b_prior[1]))
        ellip = np.absolute(np.random.normal(ellip_prior[0], ellip_prior[1]))
        shear = np.absolute(np.random.normal(shear_prior[0], shear_prior[1]))
        epa = np.random.normal(ePA_prior[0], ePA_prior[1])
        spa = np.random.normal(sPA_prior[0], sPA_prior[1])
        if gamma_prior is None:
            gamma = 2
        elif gamma_prior[1]==0:
            gamma = gamma_prior[0]
        else:
            gamma = np.random.normal(gamma_prior[0], gamma_prior[1])
            raytrace_with = 'lenstronomy'

        truth = {'R_ein': R_ein, 'ellip': ellip, 'ellip_theta': epa, 'x': 0, 'y': 0, 'shear': shear, 'shear_theta': spa,
                 'gamma': gamma}

        main = Deflector(subclass=SIE(), redshift=zlens, tovary=True,
                         varyflags=['1', '1', '1', '1', '1', '0', '0', '0', '0', '0'], **truth)

        if halo_model in ['plaw_LOS','composite_plaw']:
            assert multiplane

        if subhalo_realizations is None:
            subhalo_realizations = realization.halo_constructor(massprofile=massprofile, model_name=halo_model,model_args=substructure_model_args,
                                               Nrealizations=1, zlens=zlens, zsrc=zsrc)



        dset_v0 = solver.solve_lens_equation(macromodel=main, method=method, realizations=subhalo_realizations,
                                           identifier=identifier,
                                           srcx=src_x, srcy=src_y, grid_rmax=.08,
                                           res=0.001, source_shape='GAUSSIAN', ray_trace=False,
                                           raytrace_with=raytrace_with, source_size=source_size,
                                           multiplane=multiplane)

        if dset_v0[0].nimg != 4 or identify(dset_v0[0].x, dset_v0[0].y, R_ein) != target:

            continue

        else:

            dset = solver.solve_lens_equation(macromodel=main, method=method, realizations=subhalo_realizations,
                                               identifier=identifier,
                                               srcx=src_x, srcy=src_y, grid_rmax=.08,
                                               res=0.001, source_shape='GAUSSIAN', ray_trace=ray_trace,
                                               raytrace_with=raytrace_with, source_size=source_size,
                                               multiplane=multiplane)

            dset[0].x += np.random.normal(0,astrometric_perturbation,size=4)

            dset[0].y += np.random.normal(0,astrometric_perturbation,size=4)
    
            if return_gamma:
                return dset[0],gamma
            else:
                return dset[0]
