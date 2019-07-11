from MagniPy.util import polar_to_cart
import numpy as np
from lenstronomy.LensModel.lens_model import LensModel

class Satellite(object):

    def __init__(self, mass_model, redshift, kwargs, position_convention='phys'):

        self._mass_model = mass_model
        self._redshift = redshift
        self._kwargs = kwargs
        self._position_convention = position_convention

        if self._position_convention != 'lensed':
            x, y = self.location
            self.set_physical_location(x, y)

    @property
    def convention(self):

        return self._position_convention

    def set_physical_location(self, xphys, yphys):

        self._lensed_x, self._lensed_y = xphys, yphys

    @property
    def properties(self):

        return self._mass_model, self._redshift, self._kwargs

    @property
    def location(self):

        if self._position_convention == 'lensed':
            return self._lensed_location
        else:
            return self._physical_location

    @property
    def _physical_location(self):

        return self._kwargs['center_x'], self._kwargs['center_y']

    @property
    def _lensed_location(self):

        return self._lensed_x, self._lensed_y

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

        self._satellite_inds = []
        self._satellites = []
        self._nsat = 0

        for idx, (model_name, zsat, kwargs_sat) in enumerate(zip(satellites['lens_model_name'], satellites['z_satellite'],
                                                satellites['kwargs_satellite'])):
            self._nsat += 1

            if 'position_convention' in satellites.keys() and satellites['position_convention'][idx] == 'lensed':
                convention = 'lensed'
            else:
                convention = 'phys'

            self._satellites.append(Satellite(model_name, zsat, kwargs_sat, convention))

            self._satellite_inds.append(idx + 2)

    def get_satellite_physical_location_fromkwargs(self, z_source, i_max=None):

        kwargs_lens, lensModel = self.getlensModel(z_source)

        kwargs_lens = lensModel.lens_model._convention(kwargs_lens)

        coords = []

        if i_max is None:
            i_max = len(kwargs_lens)

        for i in range(self._nmacro, i_max):

            xphys, yphys = kwargs_lens[i]['center_x'], kwargs_lens[i]['center_y']

            coords.append([xphys, yphys])

        return coords

    def set_satellite_physical_location(self, ind, xphys, yphys):

        self._satellites[ind].set_physical_location(xphys, yphys)

    @property
    def satellite_properties(self):

        mass_model, zsat, kwargssat, convention = [], [], [], []
        if self._has_satellites:
            for sat in self._satellites:
                sat_mass_model, satz, satkwargs = sat.properties
                mass_model.append(sat_mass_model)
                zsat.append(satz)
                kwargssat.append(satkwargs)
                convention.append(sat.convention)
            return mass_model, zsat, kwargssat, convention
        else:
            return None

    def convetion_inds(self, convention_list=None):

        if convention_list is None:
            _, _, _, convention_list = self.satellite_properties

        inds = []

        for i, con in enumerate(convention_list):

            if con == 'lensed':
                inds.append(i+self._nmacro)

        if len(inds) == 0:
            inds = None

        return inds

    def _build(self, halos_only = False):

        if not hasattr(self, '_halo_names'):
            if self.realization is not None:
                self._halo_names, self._halo_redshifts, self._halo_kwargs, self.custom_class = \
                    self.realization.lensing_quantities(mass_sheet_correction_front=self._LOS_mass_sheet_front,
                                                        mass_sheet_correction_back=self._LOS_mass_sheet_back)
            else:
                self._halo_names, self._halo_redshifts, self._halo_kwargs, self.custom_class = [], [], [], None

        lensed_inds = None
        if halos_only is False:
            main_names, main_redshift, main_args = self._unpack_main(self.main)

            if self._has_satellites:

                mass_models, zshifts, satkwargs, convention_list = self.satellite_properties
                lensed_inds = self.convetion_inds(convention_list)

                main_names += mass_models
                main_redshift += zshifts
                main_args += satkwargs

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
            self._nmacro = 2

        elif main.parameterization == 'SERSIC_NFW':

            zlist = [self.zmain] * 3
            names = ['SERSIC_ELLIPSE', 'SHEAR', 'NFW']
            shear_e1, shear_e2 = polar_to_cart(main.shear, main.shear_theta)
            kwargs = main.lenstronomy_args['SERSIC']
            kwargs.append({'e1': shear_e1, 'e2': shear_e2})
            kwargs.append(main.lenstronomy_args['NFW'])
            self._nmacro = 3

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

    def getlensModel(self, z_source):

        zlist, lens_list, arg_list, custom_class, lensed_position_inds = self.lenstronomy_lists()

        lensModel = LensModel(lens_list, z_source=z_source, lens_redshift_list=zlist, multi_plane=True,
                              numerical_alpha_class=custom_class, observed_convention_index=lensed_position_inds)

        return arg_list, lensModel