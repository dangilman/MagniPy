import numpy as np
from MagniPy.lensdata import Data
from MagniPy.Solver.solveroutines import SolveRoutines
from MagniPy.LensBuild.defaults import get_default_SIE_random, get_default_SIE
from MagniPy.util import approx_theta_E
from MagniPy.Workflow.grism_lenses.quad import Quad
from lenstronomy.Util.param_util import phi_q2_ellipticity

class Lens0712(Quad):

    x = np.array([-0.785, -0.729, 0.027, 0.389])
    y = np.array([-0.142, -0.298, -0.805, 0.317])
    m = np.array([1., 0.843, 0.418, 0.082])

    sigma_x = np.array([0.005]*4)
    sigma_y = np.array([0.005]*4)
    sigma_m = np.zeros_like(sigma_x)
    zlens, zsrc = 0.5, 1.5

    solver = SolveRoutines(zlens, zsrc)

    data = Data(x, y, m, None, None,
                         sigma_x = sigma_x, sigma_y = sigma_y,
                         sigma_m=sigma_m)

    identifier = 'lens0712'

    flux_ratio_index = 0

    fluximg = ['A', 'B', 'C', 'D'][flux_ratio_index]

    _macromodel = get_default_SIE(zlens)

    _macromodel.lenstronomy_args['theta_E'] = approx_theta_E(x, y)

    gamma_min = 1.9
    gamma_max = 2.2

    srcmin = 0.001
    srcmax = 0.02

    satellite_mass_model = ['SERSIC_ELLIPSE_GAUSS_DEC']
    # from mass center
    satellite_pos_mass = np.array([0, 0])

    disk_q, disk_angle = 0.23, 90 - 59.7

    disk_angle *= np.pi / 180

    disk_angle *= np.pi / 180
    e1_disk, e2_disk = phi_q2_ellipticity(disk_angle, disk_q)
    satellite_kwargs = [{'k_eff': 0.294, 'R_sersic': 0.389, 'n_sersic': 1, 'e1': e1_disk,
                         'e2': e2_disk, 'center_x':satellite_pos_mass[0], 'center_y':satellite_pos_mass[1]}]

    def optimize_fit(self, kwargs_fit={}, macro_init = None, print_output = False):

        if 'datatofit' in kwargs_fit.keys():
            data = kwargs_fit['datatofit']
            del kwargs_fit['datatofit']
        else:
            data = self.data

        satellites = {}
        satellites['lens_model_name'] = self.satellite_mass_model
        satellites['z_satellite'] = [self.zlens]
        satellites['kwargs_satellite'] = self.satellite_kwargs

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