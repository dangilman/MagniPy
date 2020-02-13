import numpy as np
from MagniPy.lensdata import Data
from MagniPy.LensBuild.defaults import get_default_SIE_random, get_default_SIE
from MagniPy.util import approx_theta_E
from MagniPy.Workflow.grism_lenses.quad import Quad

class Lens0435(Quad):

    x = np.array([1.272, 0.306, -1.152, -0.384])
    y = np.array([0.156, -1.092, -0.636, 1.026])
    m = np.array([0.96, 0.976, 1., 0.65])

    time_delay_AB, delta_AB = -8.8, 0.8
    time_delay_AC, delta_AC = -1.1, 0.7
    time_delay_AD, delta_AD = -13.8, 0.9
    delta_time_delay = np.array([delta_AB, delta_AC, delta_AD])

    relative_arrival_times = -np.array([time_delay_AB, time_delay_AC, time_delay_AD])
    sigma_x = np.array([0.008]*4)
    sigma_y = np.array([0.008]*4)
    sigma_m = np.zeros_like(sigma_x)
    zlens, zsrc = 0.45,1.69

    data = Data(x, y, m, None, None,
                         sigma_x = sigma_x, sigma_y = sigma_y,
                         sigma_m=sigma_m)

    identifier = 'lens0435'

    flux_ratio_index = 0

    fluximg = ['A', 'B', 'C', 'D'][flux_ratio_index]

    _macromodel = get_default_SIE(zlens)

    _macromodel.lenstronomy_args['theta_E'] = approx_theta_E(x, y)

    gamma_min = 1.9
    gamma_max = 2.2

    srcmin = 0.02
    srcmax = 0.05

    amp_scale = 1000
    kwargs_lens_light = [{'amp': amp_scale * 1.4, 'R_sersic': 0.4, 'n_sersic': 4., 'center_x': 0., 'center_y': 0.}]
    kwargs_source_light = [
        {'amp': amp_scale * 1.2, 'R_sersic': 0.08, 'n_sersic': 3., 'center_x': None, 'center_y': None,
         'e1': -0.25, 'e2': 0.1}]

    has_satellite = True
    satellite_mass_model = ['SIS']
    satellite_redshift = [0.78]

    satellite_convention = ['phys']
    # from mass center
    satellite_pos_mass_observed = np.array([-2.911, 2.339])
    satellite_pos_mass = np.array([-2.37, 2.08])
    kwargs_satellite_light = [{'amp': amp_scale * 1.2, 'R_sersic': 0.2, 'n_sersic': 4.,
                               'center_x': satellite_pos_mass[0], 'center_y': satellite_pos_mass[1]}]
    # from light center
    #satellite_pos_light = [-0.1255, -1.3517]
    satellite_kwargs = [{'theta_E': 0.37, 'center_x': satellite_pos_mass[0],
                         'center_y': satellite_pos_mass[1]}]

    def optimize_fit(self, kwargs_fit={}, macro_init = None, print_output = False):

        if 'datatofit' in kwargs_fit.keys():
            data = kwargs_fit['datatofit']
            del kwargs_fit['datatofit']
        else:
            data = self.data

        satellites = {}
        satellites['lens_model_name'] = self.satellite_mass_model
        satellites['z_satellite'] = self.satellite_redshift
        satellites['kwargs_satellite'] = self.satellite_kwargs
        satellites['position_convention'] = self.satellite_convention

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