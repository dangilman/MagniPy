from MagniPy.LensBuild.renderhalos import HaloGen

class Realization:

    def __init__(self,zlens,zsrc):

        self.halo_generator = HaloGen(zd=zlens,zsrc=zsrc)

    def halo_constructor(self,massprofile='', model_name='', model_args={},
                       Nrealizations=int, filter_halo_positions=False,
                       **filter_kwargs):


        if model_name == 'plaw_main':
            spatial_name = 'uniform_cored_nfw'

        elif model_name == 'plaw_LOS':
            spatial_name = 'uniform2d'

        elif model_name == 'delta_LOS':
            spatial_name = 'uniform2d'

        elif model_name == 'composite_plaw':
            spatial_name = 'uniform_cored_nfw'

        if filter_halo_positions:
            return self.halo_generator.draw_model(model_name=model_name, spatial_name=spatial_name,
                                              massprofile=massprofile, model_kwargs=model_args, Nrealizations=Nrealizations,
                                                  filter_halo_positions=filter_halo_positions, **filter_kwargs)
        else:
            return self.halo_generator.draw_model(model_name=model_name, spatial_name=spatial_name,
                                             massprofile=massprofile, model_kwargs=model_args, Nrealizations=Nrealizations)