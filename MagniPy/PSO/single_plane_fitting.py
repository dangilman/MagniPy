from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from MagniPy.util import sort_image_index
import numpy as np
import matplotlib.pyplot as plt

class SinglePlaneFit(object):

    def __init__(self,lensmodel, x_pos, y_pos, tol_source, params, \
                 magnification_target, tol_mag, centroid_0, tol_centroid, k_start=0, arg_list=[]):

        self.Params = params
        self.lensModel = lensmodel
        self.solver = LensEquationSolver(self.lensModel)

        self.tol_source = tol_source

        self.magnification_target = magnification_target
        self.tol_mag = tol_mag

        self.centroid_0 = centroid_0
        self.tol_centroid = tol_centroid

        self._x_pos = x_pos
        self._y_pos = y_pos

        self.all_lensmodel_args = arg_list

        if k_start > 0 and len(arg_list)>k_start:

            self.k_sub = np.arange(k_start, len(arg_list))
            self.k = np.arange(0,k_start)

            self.alpha_x_sub, self.alpha_y_sub = self.lensModel.alpha(x_pos, y_pos, arg_list, k=self.k_sub)

            if tol_mag is not None:
                self.sub_fxx, self.sub_fyy, self.sub_fxy, _ = self.lensModel.hessian(x_pos,y_pos,arg_list, k = self.k_sub)

        else:

            self.k,self.k_sub = None,None
            self.alpha_x_sub, self.alpha_y_sub = 0, 0
            self.sub_fxx, self.sub_fyy, self.sub_fxy = 0,0,0

