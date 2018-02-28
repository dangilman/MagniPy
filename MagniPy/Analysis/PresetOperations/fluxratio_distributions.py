import numpy as np
from MagniPy.util import *
from MagniPy.LensBuild.renderhalos import HaloGen
from MagniPy.Solver.solveroutines import SolveRoutines
from MagniPy.LensBuild.defaults import *
from MagniPy.paths import *

def fluxratio_distributions(subhalo_model_profiles=[],subhalo_model_types=[],subhalo_model_args=[],
                            datatofit=classmethod, Nrealizations=int, outfilenames=[], z_lens=None, z_src=None,
                            start_macromodel=None,identifier=None,grid_rmax=None,res=None,sigmas=None,
                            source_size=None,raytrace_with=None,test_only=False,write_to_file=False):

    assert identifier is not None
    assert len(outfilenames) == len(subhalo_model_profiles)
    assert len(subhalo_model_types) == len(subhalo_model_profiles)

    if write_to_file:
        outfilepath = fluxratio_data_path
        assert os.path.exists(fluxratio_data_path)

    if test_only:
        print outfilenames
        print subhalo_model_profiles
        print subhalo_model_types
        print subhalo_model_args
        return

    halo_generator = HaloGen(zd=z_lens,zsrc=z_src)

    solver = SolveRoutines(zmain=z_lens,zsrc=z_src)

    if start_macromodel is None:
        start_macromodel = default_SIE
        start_macromodel.redshift = z_lens
    if sigmas is None:
        sigmas = default_sigmas
    if grid_rmax is None:
        grid_rmax = default_gridrmax(source_size)
    if res is None:
        res = default_res(source_size)

    for i,model_args in enumerate(subhalo_model_args):

        name = subhalo_model_types[i]
        profile = subhalo_model_profiles[i]

        if name == 'plaw_main':
            spatial_name = 'uniform_cored_nfw'
            multiplane = False
        elif name == 'plaw_LOS':
            spatial_name = 'uniform2d'
            multiplane = True
        elif name == 'delta_LOS':
            spatial_name = 'uniform2d'
            multiplane = True
        elif name == 'composite_plaw':
            spatial_name = 'uniform_cored_nfw'
            multiplane = True

        halos = halo_generator.draw_model(model_name=name, spatial_name=spatial_name,
                                          massprofile=profile, model_kwargs=model_args, Nrealizations=Nrealizations)

        model_data, _ = solver.two_step_optimize(macromodel=start_macromodel,datatofit=datatofit,realizations=halos,
                                                 multiplane=multiplane,method='lensmodel',ray_trace=True,sigmas=sigmas,
                                                 identifier=identifier,grid_rmax=grid_rmax,res=res,source_shape='GAUSSIAN',
                                                source_size=source_size,raytrace_with=raytrace_with)

        if write_to_file:
            write_data(outfilepath+outfilenames[i]+'.txt', model_data, mode='write')
        else:
            return model_data





