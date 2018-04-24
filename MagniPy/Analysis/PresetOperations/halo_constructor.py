from MagniPy.LensBuild.renderhalos import HaloGen
import numpy as np

class Realization:

    def __init__(self,zlens,zsrc,LOS_mass_sheet=False):

        self.halo_generator = HaloGen(zd=zlens,zsrc=zsrc,LOS_mass_sheet=LOS_mass_sheet)

    def halo_constructor(self,massprofile='', model_name='', model_args={},
                       Nrealizations=int, filter_halo_positions=False,spatial_name=None,
                       **filter_kwargs):


        if model_name == 'plaw_LOS':
            spatial_name = 'uniform2d'

        elif model_name == 'delta_LOS':
            spatial_name = 'uniform2d'

        elif model_name == 'delta_main':
            spatial_name = 'localized_uniform'

        elif model_name == 'delta_LOS':
            spatial_name = 'localized_uniform'

        if spatial_name is None:
            spatial_name = 'uniform_cored_nfw'

        if filter_halo_positions:
            return self.halo_generator.draw_model(model_name=model_name, spatial_name=spatial_name,
                                              massprofile=massprofile, model_kwargs=model_args, Nrealizations=Nrealizations,
                                                  filter_halo_positions=filter_halo_positions, **filter_kwargs)
        else:
            return self.halo_generator.draw_model(model_name=model_name, spatial_name=spatial_name,
                                             massprofile=massprofile, model_kwargs=model_args, Nrealizations=Nrealizations)

def get_redshifts(halos):

    redshifts = []
    for halo in halos:
        if 'center_x' in halo.args.keys():
            redshifts.append(halo.redshift)
    return redshifts

def get_positions(halos):

    x,y = [],[]

    for halo in halos:
        if 'center_x' in halo.args.keys():
            x.append(halo.args['center_x'])
            y.append(halo.args['center_y'])

    return np.array(x),np.array(y)


