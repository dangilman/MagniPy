import numpy as np
from halo_constructor import Realization
from MagniPy.LensBuild.lens_assemble import Deflector
from MagniPy.MassModels.SIE import SIE
from MagniPy.Solver.solveroutines import SolveRoutines
from MagniPy.util import identify
from MagniPy.MassModels.SersicNFW import SersicNFW

def create_data(identifier='create_data',config=None,b_prior=[1,0.2],ellip_prior=[.2,.05],shear_prior=[0.05,0.01],ePA_prior=[-90,90],
                sPA_prior=[-90,90],gamma_prior=None,zlens=None,zsrc=None,substructure_model_args={},source_size=0.0012*2.355**-1,massprofile='TNFW',
                raytrace_with='lenstronomy',method='lenstronomy',halo_model=None,multiplane=False,solver_class=None,
                ray_trace=True,subhalo_realizations=None,astrometric_perturbation=0.003,LOS_mass_sheet=True,return_system=False):

    run = True

    realization = Realization(zlens=zlens,zsrc=zsrc,LOS_mass_sheet=LOS_mass_sheet)

    if config=='cross':
        target = 0
    elif config=='fold':
        target = 1
    elif config == 'cusp':
        target = 2
    else:
        raise Exception('config must be one of cross, cusp, fold')

    if solver_class is None:
        solver = SolveRoutines(zlens=zlens,zsrc=zsrc,temp_folder=identifier)
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
            print 'generating halos... '
            if halo_model is not None:
                subhalo_realizations = realization.halo_constructor(massprofile=massprofile, model_name=halo_model,model_args=substructure_model_args,
                                               Nrealizations=1, zlens=zlens, zsrc=zsrc)
            else:
                subhalo_realizations = None
            print 'done.'
        print 'solving lens equation... '
        dset_v0 = solver.solve_lens_equation(macromodel=main, method=method, realizations=subhalo_realizations,
                                           identifier=identifier,
                                           srcx=src_x, srcy=src_y, grid_rmax=.08,
                                           res=0.001, source_shape='GAUSSIAN', ray_trace=False,
                                           raytrace_with=raytrace_with, source_size=source_size,
                                           multiplane=multiplane)

        if dset_v0[0].nimg != 4 or identify(dset_v0[0].x, dset_v0[0].y, R_ein) != target:

            continue

        else:
            print 'found the image configuration, ray tracing... '
            dset = solver.solve_lens_equation(macromodel=main, method=method, realizations=subhalo_realizations,
                                               identifier=identifier,
                                               srcx=src_x, srcy=src_y, grid_rmax=.08,
                                               res=0.001, source_shape='GAUSSIAN', ray_trace=True,
                                               raytrace_with=raytrace_with, source_size=source_size,
                                               multiplane=multiplane)

            dset[0].x += np.random.normal(0,astrometric_perturbation,size=4)

            dset[0].y += np.random.normal(0,astrometric_perturbation,size=4)

            if return_system:
                system = solver.build_system(main=main, additional_halos=subhalo_realizations[0], multiplane=multiplane)
                return dset[0],system,gamma
            else:
                return dset[0]


def create_data_SERSICNFW(identifier='create_data', config=None, R_ein=[1, 0.2], reff_thetaE_ratio_prior=[1,0], Rs_prior=[50,0],
                ellip_prior=[.2, .05],shear_prior=[0.05, 0.01], ePA_prior=[-90, 90],f_prior=[0.4,0],
                sPA_prior=[-90, 90], n_sersic_prior=[4,0.5],zlens=None, zsrc=None, substructure_model_args={},
                source_size=0.0012 * 2.355 ** -1, massprofile='TNFW',
                raytrace_with='lenstronomy', method='lenstronomy', halo_model=None, multiplane=False, solver_class=None,
                ray_trace=True, subhalo_realizations=None, astrometric_perturbation=0.003, return_snfw=True,
                LOS_mass_sheet=True, return_system=False):

    run = True

    realization = Realization(zlens=zlens, zsrc=zsrc, LOS_mass_sheet=LOS_mass_sheet)

    if config == 'cross':
        target = 0
    elif config == 'fold':
        target = 1
    elif config == 'cusp':
        target = 2
    else:
        raise Exception('config must be one of cross, cusp, fold')

    if solver_class is None:
        solver = SolveRoutines(zlens=zlens, zsrc=zsrc, temp_folder=identifier)
    else:
        solver = solver_class

    while run:

        if config == 'cusp':
            r = np.random.uniform(0.03 ** 2, 0.07 ** 2, 1)
        else:
            r = np.random.uniform(0, .07 ** 2, 1)

        theta = np.random.uniform(0, np.pi * 2, 1)
        src_x, src_y = float(r ** .5 * np.cos(theta)), float(r ** .5 * np.sin(theta))

        R_ein = np.absolute(np.random.normal(R_ein[0], R_ein[1]))
        ellip = np.absolute(np.random.normal(ellip_prior[0], ellip_prior[1]))
        shear = np.absolute(np.random.normal(shear_prior[0], shear_prior[1]))
        epa = np.random.normal(ePA_prior[0], ePA_prior[1])
        spa = np.random.normal(sPA_prior[0], sPA_prior[1])
        reff_thetaE_Ratio = np.absolute(np.random.normal(reff_thetaE_ratio_prior[0],reff_thetaE_ratio_prior[1]))
        Rs = np.absolute(np.random.normal(Rs[0],Rs[1]))
        N_sersic = np.absolute(np.random.normal(n_sersic_prior[0],n_sersic_prior[1]))

        if f_prior[1] == 0:
            f = f_prior[0]
        else:
            f = np.absolute(np.random.normal(f_prior[0],f_prior[1]))
            assert f>0

        truth = {'R_ein': R_ein, 'ellip': ellip, 'ellip_theta': epa, 'x': 0, 'y': 0, 'shear': shear, 'shear_theta': spa,
                 'Rs': Rs,'reff_thetaE_ratio':reff_thetaE_Ratio,'n_sersic':N_sersic}

        main = Deflector(subclass=SersicNFW(f=f), redshift=zlens, tovary=True,
                         varyflags=['1', '1', '1', '1', '1', '0', '0', '0', '0', '0'], **truth)

        if halo_model in ['plaw_LOS', 'composite_plaw']:
            assert multiplane

        if subhalo_realizations is None:
            if halo_model is not None:
                subhalo_realizations = realization.halo_constructor(massprofile=massprofile, model_name=halo_model,
                                                                    model_args=substructure_model_args,
                                                                    Nrealizations=1, zlens=zlens, zsrc=zsrc)
            else:
                subhalo_realizations = None

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
                                              srcx=src_x, srcy=src_y, grid_rmax=.12,
                                              res=0.001, source_shape='GAUSSIAN', ray_trace=True,
                                              raytrace_with=raytrace_with, source_size=source_size,
                                              multiplane=multiplane)

            dset[0].x += np.random.normal(0, astrometric_perturbation, size=4)

            dset[0].y += np.random.normal(0, astrometric_perturbation, size=4)

            if return_snfw:
                return dset[0], [N_sersic,Rs,R_ein,R_ein*reff_thetaE_Ratio]

            else:
                if return_system:
                    return dset[0],[main,subhalo_realizations]
                else:
                    return dset[0]

#data,system,gamma = create_data('test',config='cross',zlens=0.5,zsrc=1.5,substructure_model_args=
#        {'fsub':0.0,'M_halo':10**13,'logmhm':9},multiplane=True,halo_model='composite_plaw',return_system=True)
