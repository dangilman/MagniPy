from MagniPy.LensBuild.renderhalos import *
import numpy as np

class Constructor(object):

    def __init__(self,zlens,zsrc,LOS_mass_sheet=True):

        self.halo_generator = HaloGen(zd=zlens,zsrc=zsrc,LOS_mass_sheet=LOS_mass_sheet)

    def render(self, massprofile='', model_name='', model_args={},
               Nrealizations=int, filter_halo_positions=False, spatial_model=None,
               **filter_kwargs):

        if not isinstance(model_name,list):
            model_name = [model_name]
        if not isinstance(model_args,list):
            model_args = [model_args]
        if not isinstance(massprofile,list):
            massprofile = [massprofile]
        if not isinstance(spatial_model,list):
            spatial_name = [spatial_model]*len(massprofile)

        assert len(model_name)==len(model_args) and len(model_args)==len(massprofile) and len(massprofile),\
            'Must specifiy arguments for each model.'

        halos = []

        for i in range(0,int(len(model_name))):
            mod = model_name[i]
            prof = massprofile[i]
            mod_args = model_args[i]

            if spatial_model is None:
                if mod == 'plaw_LOS':
                    spatial_name[i] = 'uniform2d'
                elif mod == 'delta_LOS':
                    spatial_name[i] = 'uniform2d'

                elif mod == 'delta_main':
                    spatial_name[i] = 'localized_uniform'

                elif mod == 'delta_LOS':
                    spatial_name[i] = 'localized_uniform'
                else:
                    spatial_name[i] = 'NFW'
            else:
                spatial_name[i] = spatial_model[i]

            if filter_halo_positions:
                halos.append(self.halo_generator.draw_model(model_name=mod, spatial_name=spatial_name[i],
                                                  massprofile=prof, model_kwargs=mod_args, Nrealizations=Nrealizations,
                                                      filter_halo_positions=True, **filter_kwargs))
            else:
                halos.append(self.halo_generator.draw_model(model_name=mod, spatial_name=spatial_name[i],
                                                 massprofile=prof, model_kwargs=mod_args, Nrealizations=Nrealizations))

        
        realizations = []
        for i in range(0,Nrealizations):
            all_components = []
            for component in halos:
                all_components += component[i]
            realizations.append(all_components)
        return realizations
        
def get_masses(halos):

    mass = []
    for halo in halos:

        if 'mass' in halo.other_args.keys():
            mass.append(halo.other_args['mass'])
    return np.array(mass)

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
