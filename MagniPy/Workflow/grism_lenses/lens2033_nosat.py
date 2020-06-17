import numpy as np
from MagniPy.lensdata import Data
from MagniPy.LensBuild.defaults import get_default_SIE_random, get_default_SIE
from MagniPy.util import approx_theta_E
from MagniPy.Workflow.grism_lenses.quad import Quad

class WFI2033_nosat(Quad):

    x = np.array([-0.751, -0.039, 1.445, -0.668])
    y = np.array([0.953, 1.068, -0.307, -0.585])
    m = np.array([1., 0.65, 0.5, 0.53])

    time_delay_AB, delta_AB = 0, 100
    time_delay_AC, delta_AC = -36.2, 0.8
    time_delay_AD, delta_AD = 23.3, 1.4
    # delta_time_delay = np.array([delta_AB, delta_AC, delta_AD])
    # relative_arrival_times = np.array([time_delay_AB, time_delay_AC, time_delay_AD])

    relative_arrival_times = np.array([0.01, 36.2, 23.3 + 36.2])
    delta_time_delay = np.array([delta_AB, delta_AC, np.sqrt(delta_AC**2 + delta_AD**2)])

    sigma_x = np.array([0.005]*4)
    sigma_y = np.array([0.005]*4)
    sigma_m = np.zeros_like(sigma_x)
    zsrc, zlens = 1.66, 0.66
    # source redshift from Motta et al

    data = Data(x, y, m, None, None,
                         sigma_x = sigma_x, sigma_y = sigma_y,
                         sigma_m=sigma_m)

    identifier = 'lens2033'

    flux_ratio_index = 0

    fluximg = ['A', 'B', 'C', 'D'][flux_ratio_index]

    _macromodel = get_default_SIE(zlens)

    _macromodel.lenstronomy_args['theta_E'] = approx_theta_E(x, y)

    gamma_min = 1.9
    gamma_max = 2.1

    kwargs_lens_init = [{'theta_E': 1.1741434797426027, 'center_x': -0.14201456606451995, 'center_y': -0.020836241161107733,
                         'e1': -0.044760164290518836, 'e2': 0.3139742426542731, 'gamma': 1.95},
                        {'gamma1': 0.10751925181736302, 'gamma2': 0.011036722994634627}]

    kwargs_lens_light = [{'amp': 2500, 'R_sersic': 0.2, 'n_sersic': 4., 'center_x': None, 'center_y': None}]
    kwargs_source_light = [{'amp': 1000, 'R_sersic': 0.08, 'n_sersic': 2.5, 'center_x': None, 'center_y': None,
                             'e1': 0.01, 'e2': -0.14}]

    srcmin = 0.02
    srcmax = 0.05
    has_satellite = True
    satellite_mass_model = ['SIS']
    satellite1_pos_mass = [0.245, 2.037]

    satellite_redshift = [zlens]
    satellite_convention = ['phys']

    kwargs_satellite_light = [{'amp': 800, 'R_sersic': 0.1, 'n_sersic': 3.,
                               'center_x': satellite1_pos_mass[0],
                               'center_y': satellite1_pos_mass[1]}
                              ]

    satellite_kwargs = [{'theta_E': 0.03, 'center_x': satellite1_pos_mass[0], 'center_y': satellite1_pos_mass[1]},
                        ]

    @staticmethod
    def relative_time_delays(arrival_times):

        trel = arrival_times[1:] - arrival_times[0]
        trel = [abs(trel[0]), abs(trel[1]), abs(trel[1]) + abs(trel[2])]
        return np.array(trel)

    def optimize_fit(self, kwargs_fit={}, macro_init = None, print_output = False):

        if 'datatofit' in kwargs_fit.keys():
            data = kwargs_fit['datatofit']
            del kwargs_fit['datatofit']
        else:
            data = self.data

        if 'satellites' not in kwargs_fit.keys():
            satellites = {}

            satellites['lens_model_name'] = self.satellite_mass_model
            satellites['z_satellite'] = self.satellite_redshift
            satellites['kwargs_satellite'] = self.satellite_kwargs
            satellites['position_convention'] = self.satellite_convention

            kwargs_fit.update({'satellites': satellites})

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

# lens = WFI2033()
# x, y = lens.x, lens.y
# col = ['k', 'r', 'm', 'g']
# import matplotlib.pyplot as plt
# for l in range(0, 4):
#     plt.scatter(-x[l], y[l], color=col[l])
# plt.show()
