import numpy as np
from MagniPy.lensdata import Data
from MagniPy.LensBuild.defaults import get_default_SIE_random, get_default_SIE
from MagniPy.util import approx_theta_E
from MagniPy.Workflow.grism_lenses.quad import Quad

class Lens0408(Quad):

    x = np.array([1.981, -1.775, -1.895, 0.141])
    y = np.array([-1.495, 0.369, -0.854, 1.466])
    m = np.array([1, 0.7, 0.5, 0.4])

    time_delay_AB, delta_AB = 112, 2.1
    time_delay_AC, delta_AC = 155.5, 12.8
    time_delay_BD = 42.4
    time_delay_AD, delta_AD = time_delay_AB + time_delay_BD, np.sqrt(17.6 ** 2 + 2.1**2)

    delta_time_delay = np.array([delta_AB, delta_AC, delta_AD])
    relative_arrival_times = np.array([time_delay_AB, time_delay_AC, time_delay_AD])

    sigma_x = np.array([0.003]*4)
    sigma_y = np.array([0.003]*4)
    sigma_m = np.zeros_like(sigma_x)
    zsrc, zlens = 2.375, 0.6

    data = Data(x, y, m, None, None,
                         sigma_x = sigma_x, sigma_y = sigma_y,
                         sigma_m=sigma_m)

    identifier = 'lens0408'

    flux_ratio_index = 0

    fluximg = ['A', 'B', 'C', 'D'][flux_ratio_index]

    _macromodel = get_default_SIE(zlens)

    _macromodel.lenstronomy_args['theta_E'] = approx_theta_E(x, y)

    gamma_min = 1.9
    gamma_max = 2.1

    has_satellite = True
    satellite_mass_model = ['SIS', 'SIS', 'SIS', 'SIS', 'SIS']

    sat_1_z, sat_2_z = zlens, 0.769
    sat_1_x, sat_1_y = -1.58, -0.95

    # Satellite Einstein radii
    sat_1_thetaE = 0.22
    sat_2_thetaE = 0.77

    # OBSERVED SATELLITE POSITIONS
    #sat_2_x, sat_2_y = 1.08, -6.52

    # PHYSICAL SATELLITE POSITIONS
    sat_2_x, sat_2_y = 1.13, -7.45

    #satellite2_pos_mass_effective = [-3.63, -0.08]

    kwargs_lens_init = [{'theta_E': 1.7220357060940006, 'center_x': 0.1445702226010889, 'center_y': -3.0844186207127455e-06, 'e1': 0.16516965529124017, 'e2': 0.17318780467502645, 'gamma': 1.91894095496809}, {'gamma1': 0.10994375051894104, 'gamma2': 0.018505364037691943},
                        {'theta_E': 0.22, 'center_x': -1.58, 'center_y': -0.95},
                        {'theta_E': 0.77, 'center_x': 1.13, 'center_y': -7.45}]

    amp_scale = 1000
    kwargs_lens_light = [{'amp': 2500, 'R_sersic': 0.4, 'n_sersic': 3.9, 'center_x': 0., 'center_y': 0.}]
    kwargs_satellite_light = [{'amp': 700., 'R_sersic': 0.15, 'n_sersic': 3.,
                                'center_x': sat_1_x, 'center_y': sat_1_y},
                              None]

    #kwargs_source_light = [{'amp': 1000., 'R_sersic': 0.2, 'n_sersic': 3., 'center_x': None, 'center_y': None,
    #                        'e1': -0.2, 'e2': 0.2}]
    kwargs_source_light = [{'R_sersic': 0.08, 'n_sersic': 4.9, 'center_x': None,
                            'center_y': None, 'amp': 1400, 'e1': 0.045, 'e2': -0.09}]

    satellite_redshift = [zlens, 0.769]
    satellite_convention = ['phys', 'phys']

    satellite_kwargs = [{'theta_E': sat_1_thetaE, 'center_x': sat_1_x, 'center_y': sat_1_y},
                        {'theta_E': sat_2_thetaE, 'center_x': sat_2_x, 'center_y': sat_2_y},
                        ]

    @staticmethod
    def relative_time_delays(arrival_times):

        trel = arrival_times[1:] - arrival_times[0]
        #trel = [abs(trel[0]), abs(trel[1]), abs(trel[0]) + abs(trel[1])]
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
