from MagniPy.util import polar_to_cart
import numpy as np

class LensSystem:

    """
    This class is a single, complete lens model with a main deflector and halos in the lens plane and along LOS.
    """

    def __init__(self, main_deflector, realization=None, multiplane=False, LOS_mass_sheet = True):

        self.lens_components = []
        self.redshift_list = []
        self.multiplane = multiplane
        self._LOS_mass_sheet = LOS_mass_sheet

        self.main_lens(main_deflector)
        self.realization = realization

    def _build(self):

        if not hasattr(self, '_halo_names'):
            if self.realization is not None:
                self._halo_names, self._halo_redshifts, self._halo_kwargs = \
                    self.realization.lensing_quantities(mass_sheet_correction=self._LOS_mass_sheet)
            else:
                self._halo_names, self._halo_redshifts, self._halo_kwargs = [], [], []

        main_names, main_redshift, main_args = self._unpack_main(self.main)

        lens_model_names = main_names + self._halo_names
        lens_model_redshifts = np.append(main_redshift, self._halo_redshifts)
        lens_model_kwargs = main_args + self._halo_kwargs

        return lens_model_names, lens_model_redshifts, lens_model_kwargs

    def _unpack_main(self, main):

        if main is None:
            return [], [], []

        if main.parameterization == 'SIE_shear':

            zlist = [self.zmain]*2
            names = ['SPEMD','SHEAR']
            shear_e1, shear_e2 = polar_to_cart(main.shear, main.shear_theta)
            kwargs = [main.lenstronomy_args]
            kwargs.append({'e1': shear_e1, 'e2': shear_e2})

        if main.parameterization == 'SERSIC_NFW':

            zlist = [self.zmain] * 3
            names = ['SERSIC_ELLIPSE', 'SHEAR', 'NFW']
            shear_e1, shear_e2 = polar_to_cart(main.shear, main.shear_theta)
            kwargs = main.lenstronomy_args['SERSIC']
            kwargs.append({'e1': shear_e1, 'e2': shear_e2})
            kwargs.append(main.lenstronomy_args['NFW'])

        return names, zlist, kwargs

    def main_lens(self, deflector_main):

        if deflector_main is None:
            self.main = None
            return
        self.lens_components += [deflector_main]
        self.main = deflector_main
        self.zmain = deflector_main.redshift

    def lenstronomy_lists(self):

        lens_list, zlist, arg_list = self._build()

        return zlist,lens_list,arg_list