from MagniPy.LensBuild.defaults import get_default_SIE_random, get_default_SIE
from MagniPy.util import approx_theta_E
import os

class Quad(object):

    default_kwargs_fit = {'multiplane': False, 'n_iterations': 400, 'n_particles': 50, 'tol_centroid': 0.05,
                          'tol_mag': 0.05, 'verbose': False, 'particle_swarm': True,
                          'pso_compute_magnification': 1e+9, 'optimize_routine': 'fixed_powerlaw_shear'}

    default_kwargs_fit_lensmodel = {'method': 'lensmodel', 'ray_trace': False}

    def _fit(self, data_class, solver_class, kwargs_fit, macromodel_init = None):

        if macromodel_init is None:
            macromodel_init = get_default_SIE_random(solver_class.zmain)
            macromodel_init.lenstronomy_args['theta_E'] = approx_theta_E(data_class.x, data_class.y)

        fit_args = {}

        for key in self.default_kwargs_fit.keys():
            fit_args.update({key: self.default_kwargs_fit[key]})
        for key in kwargs_fit.keys():
            fit_args.update({key: kwargs_fit[key]})

        optdata, opt_model = solver_class.optimize_4imgs_lenstronomy(datatofit=data_class,
                                 macromodel=macromodel_init, **fit_args)

        return optdata, opt_model

    def _fit_lensmodel(self, data_class, solver_class, kwargs_fit, macromodel_init=None):

        if macromodel_init is None:
            macromodel_init = get_default_SIE_random(solver_class.zmain)
            macromodel_init.lenstronomy_args['theta_E'] = approx_theta_E(data_class.x, data_class.y)

        fit_args = {}

        for key in self.default_kwargs_fit_lensmodel.keys():
            fit_args.update({key: self.default_kwargs_fit_lensmodel[key]})
        for key in kwargs_fit.keys():
            fit_args.update({key: kwargs_fit[key]})

        optdata, opt_model = solver_class.two_step_optimize(macromodel=macromodel_init,
                                                            datatofit=data_class, **fit_args)

        os.system('rm best.sm')
        os.system('rm chitmp.dat')
        os.system('rm grid.dat')
        os.system('rm crit.dat')

        return optdata, opt_model