from MagniPy.util import polar_to_cart,cart_to_polar

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



class Deflector:

    def __init__(self, subclass=classmethod, use_lenstronomy_halos = False,
                 redshift=float, tovary=False, varyflags = None, is_subhalo = False,
                 **lens_kwargs):

        self.tovary = tovary

        self.is_subhalo = is_subhalo

        self.subclass = subclass

        self.args,self.other_args = subclass.params(**lens_kwargs)

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


    def print_args(self):

        for item in self.args:
            print item+': '+str(self.args[item])
        for item in self.other_args:
            print item+': '+str(self.other_args[item])
        if self.has_shear:
            print 'shear: ',self.shear
            print 'shear PA: ',self.shear_theta
            print 'e1: ', self.e1
            print 'e2: ', self.e2


    def get_def_angle(self,method='',x_loc=None,y_loc=None):

        if method=='lensmodel':

            return self.lensing.def_angle(x_loc,y_loc,**self.args)

        elif method == 'lenstronomy':

            if self.has_shear:
                from lenstronomy.LensModel.Profiles.external_shear import ExternalShear
                e1,e2 = polar_to_cart(self.args['shear'],self.args['lenstronomy_shear_theta'])
                ext_shear = ExternalShear()
                xshear,yshear = ext_shear.derivatives(x_loc,y_loc,e1,e2)

            if self.args['lenstronomy_name']=='SPEMD':
                from lenstronomy.LensModel.Profiles.sie import SIE
                sie = SIE()
                xdef,ydef = sie.derivatives(x_loc,y_loc,self.lenstronomy_args['theta_E'],self.lenstronomy_args['q'],
                                            self.lenstronomy_args['phi_G'])

            try:
                return xdef+xshear,ydef+yshear
            except:
                return xdef,ydef

    def update(self, method=None,**newparams):

        assert method is not None
        self.args.update(**newparams)

        if 'shear' in self.args:
            del self.args['shear']
        if 'shear_theta' in self.args:
            del self.args['shear_theta']


        if 'e1' in newparams and 'e2' in newparams:

            s,spa = cart_to_polar(newparams['e1'],newparams['e2'])

            self.shear = s
            self.shear_theta = spa

        if self.has_shear:

            if 'shear' in newparams:
                assert 'shear_theta' in newparams
                self.shear = newparams['shear']
                self.shear_theta = newparams['shear_theta']
                self.e1, self.e2 = polar_to_cart(self.shear, self.shear_theta)
