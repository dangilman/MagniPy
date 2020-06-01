import numpy as np
from MagniPy.lensdata import Data
from MagniPy.LensBuild.defaults import get_default_SIE_random, get_default_SIE
from MagniPy.util import approx_theta_E
from MagniPy.Workflow.grism_lenses.quad import Quad

class MockFold(Quad):

    x = np.array([0.71293773,  1.11756729,  0.35865728, - 0.96765226])
    y = np.array([-0.98422971, -0.36175659,  1.35996094, -0.2777054])
    m = np.array([1.,        0.98428725, 0.39814757, 0.2427231 ])
    sigma_x = np.array([0.005]*4)
    sigma_y = np.array([0.005]*4)

    relative_arrival_times = np.array([13.76955768, 14.53781908, 35.69955131])
    delta_time_delay = np.array([2.0, 2., 2.])

    sigma_m = np.zeros_like(sigma_x)

    kwargs_lens_init = [{'theta_E': 1.2, 'center_x': 0., 'center_y': 0., 'e1': 0.06, 'e2': 0.1, 'gamma': 2.0},
               {'gamma1': -0.07, 'gamma2': 0.05}]

    amp_scale = 1000.
    kwargs_lens_light = [{'amp': amp_scale * 1.1, 'R_sersic': 0.4, 'n_sersic': 4., 'center_x': 0., 'center_y': 0.}]
    kwargs_source_light = [{'amp': amp_scale * 1.6, 'R_sersic': 0.12, 'n_sersic': 3., 'center_x': None, 'center_y': None,
                            'e1': -0.1, 'e2': 0.3}]

    zlens, zsrc = 0.5, 1.5

    data = Data(x, y, m, None, None,
                         sigma_x = sigma_x, sigma_y = sigma_y,
                         sigma_m=sigma_m)

    identifier = 'mockfold'

    flux_ratio_index = 0

    fluximg = ['A', 'B', 'C', 'D'][flux_ratio_index]

    _macromodel = get_default_SIE(zlens)

    _macromodel.lenstronomy_args['theta_E'] = approx_theta_E(x, y)

    has_satellite = False

    gamma_min = 1.95
    gamma_max = 2.2

    srcmin = 0.02
    srcmax = 0.05

    @staticmethod
    def relative_time_delays(arrival_times):

        trel = np.array([arrival_times[0], arrival_times[1], arrival_times[3]]) - arrival_times[2]

        return trel

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