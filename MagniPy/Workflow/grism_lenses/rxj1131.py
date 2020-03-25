import numpy as np
from MagniPy.lensdata import Data
from MagniPy.LensBuild.defaults import get_default_SIE_random, get_default_SIE
from MagniPy.util import approx_theta_E
from MagniPy.Workflow.grism_lenses.quad import Quad

class Lens1131(Quad):

    # From Suyu et al. 2016
    # typo in Chen et al. 2016?
    g1x, g1y = 4.420, 3.932
    x = np.array([2.383, 2.344, 2.96, 5.494]) - g1x
    y = np.array([3.412, 4.594, 2.3, 4.288]) - g1y
    m = np.array([1., 0.613497, 0.730061, 0.06135])
    sigma_x = np.array([0.003]*4)
    sigma_y = np.array([0.003]*4)

    time_delay_AB, delta_AB = 0.7, 1.2
    time_delay_AC, delta_AC = 1.1, 1.5
    time_delay_AD, delta_AD = -90.6, 1.4
    delta_time_delay = np.array([delta_AB, delta_AC, delta_AD])
    relative_arrival_times = -np.array([time_delay_AB, time_delay_AC, time_delay_AD])

    kwargs_lens_init = [{'theta_E': 1.58460356403038, 'center_x': -0.005270348888552784, 'center_y': -0.029873551296941633, 'e1': 0.028027358886809944, 'e2': 0.0693670602615151, 'gamma': 1.98},
                        {'gamma1': -0.11734572581060929, 'gamma2': -0.03232611049507928}]

    sigma_m = np.zeros_like(sigma_x)
    zlens, zsrc = 0.3, 0.66

    data = Data(x, y, m, None, None,
                         sigma_x = sigma_x, sigma_y = sigma_y,
                         sigma_m=sigma_m)

    identifier = 'lens1131'

    flux_ratio_index = 1

    fluximg = ['A', 'B', 'C', 'D'][flux_ratio_index]

    _macromodel = get_default_SIE(zlens)

    _macromodel.lenstronomy_args['theta_E'] = approx_theta_E(x, y)

    has_satellite = True
    satellite_mass_model = ['SIS']
    satellite_redshift = [zlens]
    satellite_convention = ['phys']
    satellite_pos_mass = [4.323-g1x, 4.546-g1y]

    # From Chen et al. 2016
    amp_scale = 1000
    kwargs_lens_light = [{'amp': amp_scale * 1.5, 'R_sersic': 0.404, 'n_sersic': 2., 'center_x': 0., 'center_y': 0.}]
    kwargs_source_light = [{'amp': amp_scale * 4, 'R_sersic': 0.1, 'n_sersic': 2., 'center_x': None, 'center_y': None,
                             'e1': -0.2, 'e2': 0.2}]
    kwargs_satellite_light = [{'amp': amp_scale*30,'R_sersic': 0.05, 'n_sersic': 1.,
                             'center_x': satellite_pos_mass[0], 'center_y': satellite_pos_mass[1]}]

    # satellite einstein radius from Chen et al. 2016
    satellite_kwargs = [{'theta_E': 0.28, 'center_x': satellite_pos_mass[0], 'center_y': satellite_pos_mass[1]}]

    gamma_min = 1.95
    gamma_max = 2.2

    srcmin = 0.02
    srcmax = 0.05

    @staticmethod
    def relative_time_delays(arrival_times):

        trel = arrival_times[1:] - arrival_times[0]

        return np.array(trel)

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