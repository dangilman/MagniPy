from MagniPy.util import polar_to_cart
import numpy as np
from copy import deepcopy
from kwargs_translate import gravlens_to_lenstronomy
# import the lens equation solver class (finding image plane positions of a source position)
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.solver4point import Solver4Point
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
import lenstronomy.Util.param_util as param_util

class LenstronomyWrap:

    def __init__(self,multiplane=None,cosmo=None,z_source = None):

        self.xtol = 1e-10
        self.min_distance = 0.01
        self.search_window = 4
        self.precision_limit = 10**-10
        self.num_iter_max = 500
        self.cosmo = cosmo

        assert z_source is not None
        self.zsrc = z_source

        assert multiplane is not None
        self.multiplane = multiplane

        self.lens_model_list,self.lens_model_params = [],[]

    def update_settings(self,xtol=None,min_distance=None,search_window = None,precision_limit=None,num_iter_max = None):

        if xtol is not None:
            self.xtol = xtol
        if min_distance is not None:
            self.min_distance = min_distance
        if search_window is not None:
            self.search_window = search_window
        if precision_limit is not None:
            self.precision_limit = precision_limit
        if num_iter_max is not None:
            self.num_iter_max = num_iter_max

    def assemble(self,system):

        redshift_list = []
        lens_model_list = []
        lens_model_params = []

        for deflector in system.lens_components:

            new_model, new_args, new_z = self.get_lens(deflector)

            lens_model_list += new_model
            redshift_list += new_z
            lens_model_params += new_args

        self.lens_model_list, self.lens_model_params, self.redshift_list = lens_model_list,lens_model_params,redshift_list

    def get_lens(self,deflector):

        model_list = []
        z_list = []
        arg_list = []

        if deflector.profname == 'SERSIC_NFW':
            model_list.append('SERSIC')
            z_list.append(deflector.redshift)
            sersic_arg_names = ['k_eff', 'n_sersic', 'r_eff', 'center_x', 'center_y', 'e1', 'e2']
            sersic_args = {}
            for name in sersic_arg_names:
                sersic_args.update({name: deflector.lenstronomy_args[name]})
            arg_list.append(sersic_args)

            if deflector.has_shear:
                model_list.append('SHEAR')
                z_list.append(deflector.redshift)
                shear_e1, shear_e2 = polar_to_cart(deflector.shear, deflector.shear_theta)
                arg_list.append({'e1': shear_e1, 'e2': shear_e2})

            model_list.append('NFW')
            nfw_arg_names = ['theta_Rs', 'Rs', 'center_x', 'center_y']
            nfw_args = {}
            for name in nfw_arg_names:
                nfw_args.update({name: deflector.lenstronomy_args[name]})
            arg_list.append(nfw_args)
            z_list.append(deflector.redshift)

        else:
            model_list.append(deflector.profname)

            arg_list.append(deflector.lenstronomy_args)
            z_list.append(deflector.redshift)

            if deflector.has_shear:
                model_list.append('SHEAR')

                z_list.append(deflector.redshift)

                shear_e1, shear_e2 = polar_to_cart(deflector.shear, deflector.shear_theta)
                arg_list.append({'e1': shear_e1, 'e2': shear_e2})

        return model_list,arg_list,z_list


    def get_lensmodel(self):

        return LensModel(lens_model_list=self.lens_model_list, z_source=self.zsrc,
                              redshift_list=self.redshift_list,
                              cosmo=self.cosmo, multi_plane=self.multiplane)

    def update_lensparams(self, newparams):

        self.lens_model_params = newparams

    def reset_assemble(self):

        self.lens_model_list, self.lens_model_params = [], []

    def solve_leq(self,xsrc,ysrc):

        lensmodel = LensModel(lens_model_list=self.lens_model_list, z_source=self.zsrc,
                              redshift_list=self.redshift_list,cosmo = self.cosmo, multi_plane = self.multiplane)

        lensEquationSolver = LensEquationSolver(lensModel=lensmodel)

        x_image, y_image = lensEquationSolver.findBrightImage(kwargs_lens=self.lens_model_params, sourcePos_x=xsrc,
                                                              sourcePos_y=ysrc,min_distance=self.min_distance, search_window=self.search_window,
                                                                         precision_limit=self.precision_limit, num_iter_max=self.num_iter_max)

        return x_image,y_image

    def optimize_lensmodel(self, x_image, y_image, solver_type):

        self.model = self.get_lensmodel()

        if self.multiplane:
            decoupling = False
        else:
            decoupling = True

        solver4Point = Solver4Point(lensModel=self.model, decoupling=decoupling, solver_type=solver_type)

        kwargs_fit,acc = solver4Point.constraint_lensmodel(x_pos=x_image, y_pos=y_image, kwargs_list=self.lens_model_params,
                                                       xtol=self.xtol)

        return kwargs_fit

