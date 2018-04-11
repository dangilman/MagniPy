from MagniPy.util import polar_to_cart,cart_to_polar
from MagniPy.Solver.LenstronomyWrap.kwargs_translate import model_translate_tolenstronomy
from MagniPy.Solver.GravlensWrap.kwargs_translate import model_translate_togravlens

class LensSystem:

    """
    This class is a single, complete lens model with a main deflector and halos in the lens plane and along LOS.
    Each element of LensSystem is an isntance of Deflector
    """

    def __init__(self,multiplane=False,units=None):

        self.units = units
        self.lens_components = []
        self.redshift_list = []
        self.main = None
        self.multiplane = multiplane

    def main_lens(self, deflector_main):
        self.lens_components += [deflector_main]
        self.main = deflector_main
        self.zmain = self.main.redshift

        self._redshift_list(deflector_main)

    def halos(self, halos):

        assert isinstance(halos, list)

        self.lens_components += halos

        if self.multiplane:
            for object in halos:
                self._redshift_list(object)

    def update_component(self, component_index=int, newkwargs={}, method = ''):

        component = self.lens_components[component_index]

        self.lens_components[component_index] = component.update(method=method,**newkwargs)

    def _redshift_list(self,component):

        self.redshift_list.append(component.redshift)

    def print_components(self):

        for component in self.lens_components:
            component.print_args()



class Deflector:

    def __init__(self, subclass=classmethod, use_lenstronomy_halos = False,
                 redshift=float, tovary=False, varyflags = None, is_subhalo = False,
                 **lens_kwargs):

        self.tovary = tovary

        self.is_subhalo = is_subhalo

        self.subclass = subclass

        self.args,self.other_args = subclass.params(**lens_kwargs)

        self.lenstronomy_args = model_translate_tolenstronomy(self.args,name=self.other_args['name'])

        self.profname = self.other_args['name']

        self.redshift = redshift

        self.lensing = subclass

        if self.tovary:
            self.varyflags = varyflags
        else:
            self.varyflags = ['0'] * 10

        if 'shear' in lens_kwargs:
            assert 'shear_theta' in lens_kwargs
            self.has_shear = True
            self.shear = lens_kwargs['shear']
            self.shear_theta = lens_kwargs['shear_theta']
            self.e1,self.e2 = polar_to_cart(self.shear,self.shear_theta)
        else:
            self.has_shear = False
        if self.has_shear:
            from MagniPy.MassModels.ExternalShear import Shear
            self.Shear = Shear()

    def set_varyflags(self,flags):

        self.varyflags = flags

    def print_args(self):

        for item in self.args:
            print item+': '+str(self.args[item])
        for item in self.other_args:
            print item+': '+str(self.other_args[item])
        if self.has_shear:
            print 'shear: ',self.shear
            print 'shear PA: ',self.shear_theta


    def update(self, method=None,is_shear=False,**newparams):

        assert method in ['lensmodel','lenstronomy']

        if method=='lenstronomy':

            if is_shear:
                s, spa = cart_to_polar(newparams['e1'], newparams['e2'])

                self.shear = s
                self.shear_theta = spa
                return

            self.lenstronomy_args.update(newparams)
            self.args = model_translate_togravlens(newparams,self.other_args['name'])

        elif method=='lensmodel':

            self.args.update(newparams)
            self.lenstronomy_args = model_translate_tolenstronomy(self.args,name=self.other_args['name'])
            self.shear = newparams['shear']
            self.shear_theta = newparams['shear_theta']

    def translate_model(self,to='lenstronomy'):

        if to == 'lenstronomy':

            return model_translate_tolenstronomy(self.args,name=self.other_args['name'])

        if to == 'lensmodel':

            return model_translate_togravlens(self.lenstronomy_args, name=self.other_args['name'])

