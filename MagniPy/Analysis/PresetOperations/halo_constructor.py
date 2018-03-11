from MagniPy.LensBuild.renderhalos import HaloGen

def halo_constructor(massprofile='', model_name='', model_args={},
                   Nrealizations=int, zlens=None, zsrc=None, filter_halo_positions=None,
                   **filter_kwargs):

    assert zlens is not None
    assert zsrc is not None

    halo_generator = HaloGen(zd=zlens, zsrc=zsrc)

    if model_name == 'plaw_main':
        spatial_name = 'uniform_cored_nfw'

    elif model_name == 'plaw_LOS':
        spatial_name = 'uniform2d'

    elif model_name == 'delta_LOS':
        spatial_name = 'uniform2d'

    elif model_name == 'composite_plaw':
        spatial_name = 'uniform_cored_nfw'

    if filter_halo_positions is not None:
        return halo_generator.draw_model(model_name=model_name, spatial_name=spatial_name,
                                          massprofile=massprofile, model_kwargs=model_args, Nrealizations=Nrealizations,
                                              filter_halo_positions=filter_halo_positions, **filter_kwargs)
    else:
        return halo_generator.draw_model(model_name=model_name, spatial_name=spatial_name,
                                         massprofile=massprofile, model_kwargs=model_args, Nrealizations=Nrealizations)