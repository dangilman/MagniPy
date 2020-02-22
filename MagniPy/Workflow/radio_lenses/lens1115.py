import numpy as np
from MagniPy.lensdata import Data
from MagniPy.LensBuild.defaults import get_default_SIE_random, get_default_SIE
from MagniPy.util import approx_theta_E
from MagniPy.Workflow.grism_lenses.quad import Quad

class Lens1115(Quad):

    # x = np.array([0.947, 1.096, -0.722, -0.381])
    # y = np.array([-0.69, -0.232, -0.617, 1.344])
    # m = np.array([1., 0.93, 0.16, 0.21])

    x = np.array([0.947, 1.096, -0.722, -0.381])
    y = np.array([-0.69, -0.232, -0.617, 1.344])
    m = np.array([1., 0.93, 0.16, 0.21])

    # from Impey et al. 1998
    # minus sign is correct
    group_r, group_theta = 10, (-113+90)*np.pi/180
    group_x, group_y = -group_r * np.cos(group_theta), group_r*np.sin(group_theta)

    time_delay_AB, delta_AB = 0.01, 10
    time_delay_AC, delta_AC = 8.3, 1.6
    time_delay_AD, delta_AD = -9.9, 1.1
    delta_time_delay = np.array([delta_AB, delta_AC, delta_AD])
    relative_arrival_times = np.array([time_delay_AB, time_delay_AC, time_delay_AD])

    sigma_x = np.array([0.003]*4)
    sigma_y = np.array([0.003]*4)
    sigma_m = np.zeros_like(sigma_x)
    zlens, zsrc = 0.31, 1.72

    data = Data(x, y, m, None, None,
                         sigma_x = sigma_x, sigma_y = sigma_y,
                         sigma_m=sigma_m)

    # From Chen et al. 2019
    amp_scale = 1000
    kwargs_lens_light = [{'amp': 1000, 'R_sersic': 0.2, 'n_sersic': 4., 'center_x': None, 'center_y': None}]
    kwargs_source_light = [{'amp': 1700, 'R_sersic': 0.125, 'n_sersic': 2., 'center_x': None, 'center_y': None,
                             'e1': 0.15, 'e2': -0.4}]

    kwargs_lens_init = [{'theta_E': 1.0505876627617852, 'center_x': 0.0015838433690165622, 'center_y': 0.0039075583507097575, 'e1': -0.002140732329506273, 'e2': -0.0017325116179204988, 'gamma': 2.2},
     {'gamma1': -0.010208269559630813, 'gamma2': -0.025988491812216158}]

    identifier = 'lens1115'
    has_satellite = True
    satellite_mass_model = ['SIS']
    satellite_redshift = [zlens]
    satellite_convention = ['phys']
    satellite_pos_mass = [group_x, group_y]
    # From Impey et al. 1998 "Einstein ring in..."
    satellite_kwargs = [{'theta_E': 2., 'center_x': satellite_pos_mass[0], 'center_y': satellite_pos_mass[1]}]

    kwargs_satellite_light = [None]
    flux_ratio_index = 0

    fluximg = ['A', 'B', 'C', 'D'][flux_ratio_index]

    _macromodel = get_default_SIE(zlens)

    _macromodel.lenstronomy_args['theta_E'] = approx_theta_E(x, y)

    gamma_min = 1.95
    gamma_max = 2.2

    srcmin = 0.005
    srcmax = 0.025

    def optimize_fit(self, kwargs_fit={}, macro_init = None, print_output = False):

        if 'datatofit' in kwargs_fit.keys():
            data = kwargs_fit['datatofit']
            del kwargs_fit['datatofit']
        else:
            data = self.data

        optdata, optmodel = self._fit(data, self.solver, kwargs_fit, macromodel_init=macro_init)

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