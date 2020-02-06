import numpy as np
from MagniPy.lensdata import Data
from MagniPy.LensBuild.defaults import get_default_SIE_random, get_default_SIE
from MagniPy.util import approx_theta_E
from MagniPy.Workflow.grism_lenses.quad import Quad

class Lens1413_midIR(Quad):

    _xref, _yref = -0.142, 0.561
    x = np.array([0., -0.744, 0.492, -0.354]) - _xref
    y = np.array([0., 0.168, 0.713, 1.040]) - _yref
    m = np.array([1., 0.84, 0.72, 0.4]) # technically flux ratios

    sigma_x = np.array([0.008]*4)
    sigma_y = np.array([0.008]*4)
    sigma_m = np.zeros_like(sigma_x)
    zlens, zsrc = 0.9, 2.56

    data = Data(x, y, m, None, None,
                         sigma_x = sigma_x, sigma_y = sigma_y,
                         sigma_m=sigma_m)

    identifier = 'lens1413'

    flux_ratio_index = 0

    fluximg = ['A', 'B', 'C', 'D'][flux_ratio_index]

    _macromodel = get_default_SIE(zlens)

    _macromodel.lenstronomy_args['theta_E'] = approx_theta_E(x, y)

    gamma_min = 1.9
    gamma_max = 2.2

    srcmin = 0.02
    srcmax = 0.05

    has_satellite = True

    satellite_mass_model = ['SIS']
    satellite_redshift = [zlens]

    satellite_convention = ['phys']
    # from mass center
    satellite_pos_mass = np.array([-1.87, 4.15]) - np.array([_xref, _yref])
    satellite_pos_mass_effective = satellite_pos_mass
    # from light center
    # satellite_pos_light = [-0.1255, -1.3517]
    satellite_kwargs = [{'theta_E': 0.63, 'center_x': satellite_pos_mass_effective[0],
                         'center_y': satellite_pos_mass_effective[1]}]

class Lens1413_NL(Quad):

    x = np.array([-0.6, -0.22, 0.137, 0.629])
    y = np.array([-0.418, 0.448, -0.588, 0.117])
    m = np.array([0.93, 0.94, 1, 0.41])

    sigma_x = np.array([0.008]*4)
    sigma_y = np.array([0.008]*4)
    sigma_m = np.zeros_like(sigma_x)
    zlens, zsrc = 0.9, 2.56

    data = Data(x, y, m, None, None,
                         sigma_x = sigma_x, sigma_y = sigma_y,
                         sigma_m=sigma_m)

    identifier = 'lens1413'

    flux_ratio_index = 0

    fluximg = ['A', 'B', 'C', 'D'][flux_ratio_index]

    _macromodel = get_default_SIE(zlens)

    _macromodel.lenstronomy_args['theta_E'] = approx_theta_E(x, y)

    gamma_min = 1.9
    gamma_max = 2.2

    srcmin = 0.02
    srcmax = 0.05

    has_satellite = False
    #satellite_mass_model = ['SIS']
    #satellite_redshift = [2]

    #satellite_convention = ['lensed']
    # from mass center
    #satellite_pos_mass = np.array([2.007, 3.552])
    #satellite_pos_mass_effective = np.array([-2.37, 2.08])
    # from light center
    #satellite_pos_light = [-0.1255, -1.3517]
    #satellite_kwargs = [{'theta_E': 1, 'center_x': satellite_pos_mass[0],
    #                     'center_y': satellite_pos_mass[1]}]

    def optimize_fit(self, kwargs_fit={}, macro_init = None, print_output = False):

        if 'datatofit' in kwargs_fit.keys():
            data = kwargs_fit['datatofit']
            del kwargs_fit['datatofit']
        else:
            data = self.data

        satellites = None
        #satellites['lens_model_name'] = self.satellite_mass_model
        #satellites['z_satellite'] = self.satellite_redshift
        #satellites['kwargs_satellite'] = self.satellite_kwargs
        #satellites['position_convention'] = self.satellite_convention

        kwargs_fit.update({'satellites': satellites})

        kwargs_fit.update({'multiplane': True})

        optdata, optmodel = self._fit(data, self.solver, kwargs_fit, macromodel_init=macro_init,
                                      sat_pos_lensed=True)

        if print_output:
            self._print_output(optdata[0], optmodel[0])

        return optdata[0], optmodel[0]

    def optimize_fit_lensmodel(self, kwargs_fit={}, macro_init = None, print_output = False):

        kwargs_fit.update({'identifier': self.identifier})
        optdata, optmodel = self._fit_lensmodel(self.data, self.solver, kwargs_fit, macromodel_init=macro_init)

        if print_output:
            self._print_output(optdata[0], optmodel[0])

        return optdata[0], optmodel[0]

    def _print_output(self, optdata, optmodel):

        macromodel = optmodel.lens_components[0]

        print('optimized mags: ', optdata.m)
        print('observed mags: ', self.data.m)
        print('lensmodel fit: ')
        print('Einstein radius: ', macromodel.lenstronomy_args['theta_E'])
        print('shear, shear_theta:', macromodel.shear, macromodel.shear_theta)
        print('ellipticity, PA:', macromodel.ellip_PA_polar()[0], macromodel.ellip_PA_polar()[1])
        print('centroid: ', macromodel.lenstronomy_args['center_x'],
              macromodel.lenstronomy_args['center_y'])
        print('\n')
        print('flux ratios w.r.t. image '+str(self.fluximg)+':')
        print('observed: ', self.data.compute_flux_ratios(index=self.flux_ratio_index))
        print('recovered: ', optdata.compute_flux_ratios(index=self.flux_ratio_index))