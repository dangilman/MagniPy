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
        self.redshift_list = []
        self.main = None
        self.multiplane = multiplane

    def main_lens(self, deflector_main):
        self.lens_components += [deflector_main]
        self.main = deflector_main

        self._redshift_list(deflector_main)

    def halos(self, halos):

        assert isinstance(halos, list)

        self.lens_components += halos

        if self.multiplane:
            for object in halos:
                self._redshift_list(object)

    def update_component(self, component_index=int, newkwargs={}):

        component = self.lens_components[component_index]

        self.lens_components[component_index] = component.update(**newkwargs)

    def _redshift_list(self,component):

        self.redshift_list.append(component.args['z'])


class Deflector:

    def __init__(self, subclass=classmethod, use_lenstronomy_halos = False,
                 redshift=float, tovary=False, varyflags = None,
                 **lens_kwargs):

        self.tovary = tovary

        self.has_shear = False

        self.subclass = subclass

        self.args,self.lenstronomy_args = subclass.params(**lens_kwargs)

        if 'shear' in self.args:
            if self.args['shear'] != 0:
                self.has_shear = True

        self.args['z'] = redshift

        self.lensing = subclass

        self.profname = self.args['name']

        if self.tovary:
            self.varyflags = varyflags
        else:
            self.varyflags = ['0'] * 10


    def print_args(self,method=None):

        if method is None:
            print 'lenstronomy kwargs: '

            for item in self.lenstronomy_args:
                print item+': '+str(self.lenstronomy_args[item])

            print 'gravlens kwargs: '

            for item in self.args:
                print item+': '+str(self.args[item])

        elif method=='lenstronomy':
            print 'lenstronomy kwargs: '

            for item in self.lenstronomy_args:
                print item + ': ' + str(self.lenstronomy_args[item])

        elif method=='lensmodel':

            print 'gravlens kwargs: '

            for item in self.args:
                print item + ': ' + str(self.args[item])


    def update(self,method='',**newparams):

        if method == 'lensmodel':

            self.args.update(**newparams)

            self.lenstronomy_args.update(**self.subclass.translate_to_lenstronomy(**self.args))

        elif method =='lenstronomy':

            self.lenstronomy_args.update(**newparams)

            self.args.update(**self.subclass.translate_to_lensmodel(**self.lenstronomy_args))

        else:
            raise ValueError('must specify which set of kwargs to update')
