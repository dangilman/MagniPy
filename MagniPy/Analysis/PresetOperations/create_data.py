import numpy as np
from halo_constructor import Constructor
from MagniPy.LensBuild.lens_assemble import Deflector
from MagniPy.MassModels.SIE import SIE
from MagniPy.Solver.solveroutines import SolveRoutines
from MagniPy.util import identify
from MagniPy.MassModels.SersicNFW import SersicNFW

def create_data(config=None,b_prior=[1,0.2],ellip_prior=[.2,.05],shear_prior=[0.05,0.01],ePA_prior=[-90,90],
                sPA_prior=[-90,90],gamma_prior=None,zlens=None,zsrc=None,substructure_model_args={},source_size=0.0012*2.355**-1,
                massprofile='TNFW',raytrace_with='lenstronomy',method='lenstronomy',halo_model=None,multiplane=False,ray_trace=True,
                subhalo_realizations=None,astrometric_perturbation=0.003,LOS_mass_sheet=True,return_system=True):


    realizations = Constructor(zlens, zsrc)
    solver = SolveRoutines(zlens,zsrc)

    if config == 'cusp':
        target = 2
    elif config == 'fold':
        target = 1
    else:
        target = 0

    while True:

        src_r = np.random.uniform(0,0.1**2)
        theta = np.random.uniform(0,2*np.pi)
        srcx,srcy = src_r**0.5*np.cos(theta),src_r**0.5*np.sin(theta)

        R_ein = np.random.normal(b_prior[0],b_prior[1])
        ellip = np.random.normal(ellip_prior[0],ellip_prior[1])
        epa = np.random.uniform(ePA_prior[0],ePA_prior[1])
        shear = np.random.normal(shear_prior[0],shear_prior[1])
        shear_pa = np.random.normal(sPA_prior[0],sPA_prior[1])

        if gamma_prior is None or gamma_prior[1] == 0:
            gamma = 2
        else:
            gamma = np.random.normal(gamma_prior[0],gamma_prior[1])

        start_SIE = Deflector(subclass=SIE(),redshift=zlens,R_ein=R_ein,ellip=ellip,ellip_theta = epa, x=0,
                   y = 0, gamma=gamma,shear=shear,shear_theta=shear_pa)

        halos = realizations.render(massprofile=massprofile, model_name=halo_model, model_args=substructure_model_args,
                                    filter_halo_positions=False, Nrealizations=1)

        data_init = solver.solve_lens_equation(macromodel=start_SIE,realizations=None,multiplane=multiplane,method=method,ray_trace=False,
                                               srcx=srcx,srcy=srcy,grid_rmax=0.12,res=0.001,source_size=source_size)

        if data_init[0].nimg != 4:
            continue

        imgconfig = identify(data_init[0].x,data_init[0].y,R_ein)

        if imgconfig != target:
            continue
        solver = SolveRoutines(zlens, zsrc)
        data = solver.solve_lens_equation(macromodel=start_SIE,realizations=halos,multiplane=multiplane,method=method,ray_trace=True,
                                               srcx=srcx,srcy=srcy,grid_rmax=0.12,res=0.001,source_size=source_size)
        if data[0].nimg != 4:
            continue

        imgconfig = identify(data[0].x, data[0].y, R_ein)

        if imgconfig != target:
            continue
        else:
            break

    solver = SolveRoutines(zlens, zsrc)
    system = solver.build_system(main=start_SIE,additional_halos=halos[0],multiplane=multiplane)

    return data,system,gamma





