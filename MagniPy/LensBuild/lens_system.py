from MagniPy.util import polar_to_cart
import numpy as np

class LensSystem(object):

    """
    This class is a single, complete lens model with a main deflector and halos in the lens plane and along LOS.
    """

    def __init__(self, main_deflector, satellites = None, realization=None, multiplane=False,
                 LOS_mass_sheet_front=7.7, LOS_mass_sheet_back = 8):

        self.lens_components = []
        self.redshift_list = []
        self.multiplane = multiplane

        self._LOS_mass_sheet_front = LOS_mass_sheet_front
        self._LOS_mass_sheet_back = LOS_mass_sheet_back

        self.main_lens(main_deflector)
        self.realization = realization

        self._has_satellites = False
        if satellites is not None:
            self._has_satellites = True
            self.satellites(satellites)

    def satellites(self, satellites):

        mass_model = satellites['lens_model_name']
        redshift = satellites['z_satellite']
        kwargs = satellites['kwargs_satellite']

        self.satellite_mass_model = mass_model
        self.satellite_redshift = redshift
        self.satellite_kwargs = kwargs
        self.satellite_position_lensed = False
        if 'position_convention' in satellites.keys():
            if satellites['position_convention'] == 'lensed':
                self.satellite_position_lensed = True

        if len(self.satellite_kwargs) > 1:
            raise ValueError('more than 1 satellite not currently supported')

        assert len(mass_model) == len(redshift)
        assert len(redshift) == len(kwargs)

        self._satellite_physical_location = np.array([kwargs[0]['center_x'], kwargs[0]['center_y']])

    def _build(self, halos_only = False):

        if not hasattr(self, '_halo_names'):
            if self.realization is not None:
                self._halo_names, self._halo_redshifts, self._halo_kwargs, self.custom_class = \
                    self.realization.lensing_quantities(mass_sheet_correction_front=self._LOS_mass_sheet_front,
                                                        mass_sheet_correction_back=self._LOS_mass_sheet_back)
            else:
                self._halo_names, self._halo_redshifts, self._halo_kwargs, self.custom_class = [], [], [], None

        lensed_inds = False
        if halos_only is False:
            main_names, main_redshift, main_args = self._unpack_main(self.main)

            if self._has_satellites:
                main_names += [model for model in self.satellite_mass_model]
                main_redshift += [red_shift for red_shift in self.satellite_redshift]
                main_args += [kwargs_sat for kwargs_sat in self.satellite_kwargs]

                if self.satellite_position_lensed:
                    lensed_inds = [len(main_names)-1]

            lens_model_names = main_names + self._halo_names
            lens_model_redshifts = np.append(main_redshift, self._halo_redshifts)
            lens_model_kwargs = main_args + self._halo_kwargs

        else:
            lens_model_names = self._halo_names
            lens_model_redshifts = self._halo_redshifts
            lens_model_kwargs = self._halo_kwargs

        return lens_model_names, lens_model_redshifts, lens_model_kwargs, self.custom_class, lensed_inds

    def _unpack_main(self, main):

        if main is None:
            return [], [], []

        if main.parameterization == 'SIE_shear':

            zlist = [self.zmain]*2
            names = ['SPEMD','SHEAR']
            shear_e1, shear_e2 = polar_to_cart(main.shear, main.shear_theta)
            kwargs = [main.lenstronomy_args]
            kwargs.append({'e1': shear_e1, 'e2': shear_e2})

        elif main.parameterization == 'SERSIC_NFW':

            zlist = [self.zmain] * 3
            names = ['SERSIC_ELLIPSE', 'SHEAR', 'NFW']
            shear_e1, shear_e2 = polar_to_cart(main.shear, main.shear_theta)
            kwargs = main.lenstronomy_args['SERSIC']
            kwargs.append({'e1': shear_e1, 'e2': shear_e2})
            kwargs.append(main.lenstronomy_args['NFW'])

        else:
            raise Exception('macromodel parameteriztion '+main.parameterization+' not recognized.')

        return names, zlist, kwargs

    def main_lens(self, deflector_main):

        if deflector_main is None:
            self.main = None
            return
        self.lens_components += [deflector_main]
        self.main = deflector_main
        self.zmain = deflector_main.redshift

    def lenstronomy_lists(self, halos_only = False):

        lens_list, zlist, arg_list, custom_class, lensed_position_inds = self._build(halos_only=halos_only)

        return zlist,lens_list,arg_list, custom_class, lensed_position_inds