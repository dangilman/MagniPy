from MagniPy.util import polar_to_cart
import numpy as np
from copy import deepcopy
# import the lens equation solver class (finding image plane positions of a source position)
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.solver4point import Solver4Point
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import lenstronomy.Util.param_util as param_util

class LenstronomyWrap:

    def __init__(self,xtol=1.49012e-10,min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=100,
                 multiplane=None,cosmo=None,z_source = None):

        self.xtol = xtol
        self.min_distance = 0.01
        self.search_window = 5
        self.precision_limit = 10**-10
        self.num_iter_max = 1000
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

        self.redshift_list = []
        lens_model_list = []
        lens_model_params = []

        for deflector in system.lens_components:

            lens_model_list.append(deflector.profname)

            newargs = deepcopy(deflector.args)

            if 'phi_G' in deflector.args:
                newargs['phi_G'] = deflector.args['phi_G'] - 0.5*np.pi

            if 'theta_E' in deflector.args and deflector.profname=='SPEMD':
                q = deflector.args['q']
                newargs['theta_E'] = deflector.args['theta_E']*((1+q**2)*(2*q)**-1)**.5

            lens_model_params.append(newargs)
            self.redshift_list.append(deflector.redshift)

            if deflector.has_shear:

                lens_model_list.append('SHEAR')

                self.redshift_list.append(deflector.redshift)

                e1,e2 = polar_to_cart(deflector.shear,deflector.shear_theta)
                lens_model_params.append({'e1':e1,'e2':e2})

        self.lens_model_list, self.lens_model_params = lens_model_list,lens_model_params

    def get_lensmodel(self):

        return LensModel(lens_model_list=self.lens_model_list, z_source=self.zsrc,
                              redshift_list=self.redshift_list,
                              cosmo=self.cosmo, multi_plane=self.multiplane)

    def update_lensparams(self, newparams):

        self.lens_model_params = newparams
        print self.lens_model_params
        a=input('continue')

    def reset_assemble(self):

        self.lens_model_list, self.lens_model_params = [], []

    def solve_leq(self,xsrc,ysrc):

        lensmodel = LensModel(lens_model_list=self.lens_model_list,z_source=self.zsrc,redshift_list=self.redshift_list,
                              cosmo=self.cosmo,multi_plane=self.multiplane)

        lensEquationSolver = LensEquationSolver(lensModel=lensmodel)

        x_image,y_image =  lensEquationSolver.findBrightImage(kwargs_lens=self.lens_model_params, sourcePos_x=xsrc, sourcePos_y=ysrc,
                                                                         min_distance=self.min_distance, search_window=self.search_window,
                                                                         precision_limit=self.precision_limit, num_iter_max=self.num_iter_max)

        return x_image,y_image

    def optimize_lensmodel(self, x_image, y_image, solver_type):

        self.model = self.get_lensmodel()

        solver4Point = Solver4Point(lensModel=self.model, decoupling=True, solver_type=solver_type)
        kwargs_fit,acc = solver4Point.constraint_lensmodel(x_pos=x_image, y_pos=y_image, kwargs_list=self.lens_model_params,
                                                       xtol=self.xtol)

        return kwargs_fit









