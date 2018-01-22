from MagniPy.MassModels.SIE import *
from MagniPy.MassModels.ExternalShear import *
from cosmology import Cosmo

class LensSystem:

    """
    This class is a single, complete lens model with a main deflector and halos in the lens plane and along LOS.
    Each element of LensSystem is an isntance of Deflector
    """

    def __init__(self,multiplane=False):

        self.lens_components = []
        self.main = None
        self.multiplane = multiplane

    def main_lens(self, deflector_main):
        self.lens_components += [deflector_main]
        self.main = deflector_main

    def halos(self, halos):

        assert isinstance(halos, list)

        self.lens_components += halos

    def update_component(self, component_index=int, newkwargs={}):
        component = self.lens_components[component_index]

        self.lens_components[component_index] = component.update(**newkwargs)


class Deflector:

    def __init__(self, subclass=classmethod, lensclass=classmethod, trunc=float, xcoord = float, ycoord = float,
                 redshift=float, tovary=False, varyflags = None,
                 **lens_kwargs):

        self.args = subclass.params(**lens_kwargs)
        self.profname = self.args['name']
        self.args['rt'] = trunc
        self.args['x'],self.args['y'] = xcoord,ycoord
        self.args['z'] = redshift

        self.has_shear = True

        if 'shear' not in self.args:
            self.args['shear'] = 0
            self.args['shear_theta'] = 0
            self.has_shear = False

        self.lensing = [lensclass]

        self.tovary = tovary

        if self.tovary:
            self.varyflags = varyflags
        else:
            self.varyflags = ['0']*10

    def print_args(self):
        for item in self.args:
            print item+': '+str(self.args[item])

    def update(self,**newparams):

        self.args.update(**newparams)