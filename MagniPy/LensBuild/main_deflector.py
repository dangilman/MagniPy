from MagniPy.MassModels.ExternalShear import Shear
from MagniPy.Solver.GravlensWrap.kwargs_translate import model_translate_togravlens
from MagniPy.Solver.LenstronomyWrap.kwargs_translate import model_translate_tolenstronomy
from MagniPy.util import polar_to_cart, cart_to_polar
import numpy as np

class Deflector(object):

    def __init__(self, subclass=classmethod, redshift=float,
                 tovary=False, varyflags = None, **lens_kwargs):

        self.tovary = tovary

        self.subclass = subclass

        args, self.other_args = subclass.params(**lens_kwargs)
        self.profname = self.other_args['name']

        self.lenstronomy_args = model_translate_tolenstronomy(args, name=self.other_args['name'])
        self.gravlens_args = model_translate_togravlens(self.lenstronomy_args, name=self.other_args['name'])
        self.parameterization = self.other_args['parameterization']

        self.redshift = redshift

        self.lensing = subclass

        if self.tovary:
            self.varyflags = varyflags
        else:
            self.varyflags = ['0'] * 10

        if 'shear' in lens_kwargs:
            assert 'shear_theta' in lens_kwargs
            self.has_shear = True
            self.shear_theta = lens_kwargs['shear_theta']
            self.set_shear(lens_kwargs['shear'])

        else:
            self.has_shear = False

        if self.has_shear:

            self.Shear = Shear()

        else:

            self.shear_args = {'gamma1':1e-10,'gamma2':1e-10}

    def update_lenstronomy_args(self,newargs):

        for key in newargs.keys():
            self.lenstronomy_args[key] = newargs[key]

        self.gravlens_args = model_translate_togravlens(self.lenstronomy_args, name=self.other_args['name'])

    def set_varyflags(self,flags):

        self.varyflags = flags

    def print_args(self):

        for item in self.args:
            print(item+': '+str(self.args[item]))
        for item in self.other_args:
            print(item+': '+str(self.other_args[item]))
        if self.has_shear:
            print('shear: ',self.shear)
            print('shear PA: ',self.shear_theta)

    def set_shear(self,newshear):

        self.shear = newshear
        e1,e2 = polar_to_cart(self.shear,self.shear_theta)
        self.shear_args = {'gamma1':e1,'gamma2':e2}

    def ellip_PA_phiq(self):

        e1, e2 = self.lenstronomy_args['e1'], self.lenstronomy_args['e2']
        phi = np.arctan2(e1, e2) / 2
        c = np.sqrt(e1 ** 2 + e2 ** 2)
        if c > 0.999:
            c = 0.999
        q = (1 - c) / (1 + c)

        return phi * 180 * np.pi ** -1, 1-q

    def ellip_PA_polar(self):

        ellip, ePA = cart_to_polar(self.lenstronomy_args['e1'], self.lenstronomy_args['e2'])
        return ellip, ePA

    def update(self,method=None,is_shear=False,**newparams):

        assert method in ['lensmodel','lenstronomy']

        if method=='lenstronomy':

            if is_shear:

                s, spa = cart_to_polar(newparams['gamma1'], newparams['gamma2'])

                self.shear_theta = spa

                self.set_shear(s)

                return

            if self.parameterization == 'SERSIC_NFW':

                if 'theta_Rs' in newparams.keys():
                    self.lenstronomy_args['NFW'].update(newparams)
                else:
                    self.lenstronomy_args['SERSIC'].update(newparams)

            elif self.parameterization == 'SERSIC_NFW_DISK':

                if 'theta_Rs' in newparams.keys():

                    self.lenstronomy_args['NFW'].update(newparams)

                elif 'n_sersic' in newparams.keys():

                    self.lenstronomy_args['SERSIC'].update(newparams)

            else:

                self.lenstronomy_args.update(newparams)
                self.gravlens_args = model_translate_togravlens(newparams,self.other_args['name'])

        elif method=='lensmodel':

            self.lenstronomy_args = model_translate_tolenstronomy(newparams,name=self.other_args['name'])
            self.gravlens_args = model_translate_togravlens(self.lenstronomy_args,self.other_args['name'])

            self.shear = newparams['shear']
            self.shear_theta = newparams['shear_theta']

            self.set_shear(self.shear)

    def translate_model(self,to='lenstronomy'):

        if to == 'lenstronomy':

            return model_translate_tolenstronomy(self.args,name=self.other_args['name'])

        if to == 'lensmodel':

            return model_translate_togravlens(self.lenstronomy_args, name=self.other_args['name'])