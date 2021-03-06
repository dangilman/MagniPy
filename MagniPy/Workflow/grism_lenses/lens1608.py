import numpy as np
from MagniPy.lensdata import Data
from MagniPy.LensBuild.defaults import get_default_SIE_random, get_default_SIE
from MagniPy.util import approx_theta_E
from MagniPy.Workflow.grism_lenses.quad import Quad

class Lens1608(Quad):

    g1x, g1y = 0.4161, -1.0581
    g2x, g2y = -0.2897, -0.9243

    g2x -= g1x
    g2y -= g1y
    # from Fassnacht et al. 2002
    x = np.array([-0.000, -0.738, -0.7446, 1.1284]) - g1x
    y = np.array([0.000, -1.961, -0.4537, -1.2565]) - g1y
    m = np.array([1., 0.5, 0.51, 0.35])
    sigma_x = np.array([0.005]*4)
    sigma_y = np.array([0.005]*4)

    # from Koopmans et al. 2003
    time_delay_AB, delta_AB = -31.5, 1.5
    time_delay_AC, delta_AC = 4.5, 1.5
    time_delay_AD, delta_AD = 45.5, 2.
    # delta_time_delay = np.array([delta_AB, delta_AC, delta_AD])
    # relative_arrival_times = np.array([time_delay_AB, time_delay_AC, time_delay_AD])

    relative_arrival_times = np.array([31., 36., 76.])
    delta_time_delay = np.array([2., 1.5, 2.])

    sigma_m = np.zeros_like(sigma_x)

    kwargs_lens_init = [{'theta_E': 0.9036453181989341, 'center_x': 0.007600860009089998, 'center_y': -0.057685769153856994, 'e1': 0.33480085295892065, 'e2': -0.05097223221504117, 'gamma': 2.08},
                        {'gamma1': 0.06096918564904592, 'gamma2': 0.07721907911829631}]

    amp_scale = 1000.
    kwargs_lens_light = [{'amp': amp_scale * 1.1, 'R_sersic': 0.4, 'n_sersic': 4., 'center_x': 0., 'center_y': 0.}]
    kwargs_source_light = [{'amp': amp_scale * 1.6, 'R_sersic': 0.12, 'n_sersic': 3., 'center_x': None, 'center_y': None,
                            'e1': -0.1, 'e2': 0.3}]

    zlens, zsrc = 0.61, 1.4

    data = Data(x, y, m, None, None,
                         sigma_x = sigma_x, sigma_y = sigma_y,
                         sigma_m=sigma_m)

    identifier = 'lens1608'

    flux_ratio_index = 0

    fluximg = ['A', 'B', 'C', 'D'][flux_ratio_index]

    _macromodel = get_default_SIE(zlens)

    _macromodel.lenstronomy_args['theta_E'] = approx_theta_E(x, y)

    has_satellite = True
    satellite_mass_model = ['SIS']
    satellite_redshift = [zlens]
    satellite_convention = ['phys']
    satellite_pos_mass = [g2x, g2y]
    kwargs_satellite_light = [{'amp': amp_scale * 0.7, 'R_sersic': 0.2, 'n_sersic': 4.,
                               'center_x': g2x, 'center_y': g2y}]
    # satellite einstein radius from Koopmans et al. 2003
    satellite_kwargs = [{'theta_E': 0.26, 'center_x': satellite_pos_mass[0], 'center_y': satellite_pos_mass[1]}]

    gamma_min = 1.95
    gamma_max = 2.2

    srcmin = 0.02
    srcmax = 0.05

    @staticmethod
    def relative_time_delays(arrival_times):

        trel = arrival_times[1:] - arrival_times[0]
        trel = [abs(trel[0]), abs(trel[0]) + trel[1], abs(trel[0]) + abs(trel[2])]
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