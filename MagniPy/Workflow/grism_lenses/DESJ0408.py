import numpy as np
from MagniPy.lensdata import Data
from MagniPy.Solver.solveroutines import SolveRoutines
from MagniPy.LensBuild.defaults import get_default_SIE_random, get_default_SIE
from MagniPy.util import approx_theta_E
from MagniPy.Workflow.grism_lenses.quad import Quad

class Lens0408(Quad):

    x = np.array([1.981, -1.775, -1.895, 0.141])
    y = np.array([-1.495, 0.369, -0.854, 1.466])
    m = np.array([1, 0.7, 0.5, 0.4])

    time_delay_AB, delta_AB = 112, 2.6
    time_delay_AC, delta_AC = 155.5, 12.8
    time_delay_BD = 42.4
    time_delay_AD, delta_AD = time_delay_AB + time_delay_BD, np.sqrt(17.6 ** 2 + 2.1**2)
    delta_time_delay = np.array([delta_AB, delta_AC, delta_AD])
    relative_arrival_times = np.array([time_delay_AB, time_delay_AC, time_delay_AD])

    sigma_x = np.array([0.003]*4)
    sigma_y = np.array([0.003]*4)
    sigma_m = np.zeros_like(sigma_x)
    zsrc, zlens = 2.375, 0.6

    solver = SolveRoutines(zlens, zsrc)

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

    srcmin = 0.02
    srcmax = 0.05
    has_satellite = True
    satellite_mass_model = ['SIS', 'SIS', 'SIS', 'SIS', 'SIS']
    satellite1_pos_mass = [-1.58, -0.95]
    satellite2_pos_mass = [1.08, -6.52]
    satellite3_pos_mass = [-0.40, -13.58]
    satellite4_pos_mass = [5.34, -0.78]
    satellite5_pos_mass = [10.9, 5.53]
    theta_E_G2 = 0.21
    theta_E_G3 = 1.5
    theta_E_G4 = 0.35
    theta_E_G5 = 0.075
    theta_E_G6 = 0.1
    #satellite2_pos_mass_effective = [-3.63, -0.08]

    satellite_redshift = [zlens, 0.769, 0.771, 1.032, 0.594]
    satellite_convention = ['phys', 'lensed', 'lensed', 'lensed', 'lensed']

    satellite_kwargs = [{'theta_E': theta_E_G2, 'center_x': satellite1_pos_mass[0], 'center_y': satellite1_pos_mass[1]},
                        {'theta_E': theta_E_G3, 'center_x': satellite2_pos_mass[0], 'center_y': satellite2_pos_mass[1]},
                        {'theta_E': theta_E_G4, 'center_x': satellite3_pos_mass[0], 'center_y': satellite3_pos_mass[1]},
                        {'theta_E': theta_E_G5, 'center_x': satellite4_pos_mass[0], 'center_y': satellite4_pos_mass[1]},
                        {'theta_E': theta_E_G6, 'center_x': satellite5_pos_mass[0], 'center_y': satellite5_pos_mass[1]}]

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
