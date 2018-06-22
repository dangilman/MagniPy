from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from MagniPy.util import sort_image_index
import numpy as np
import matplotlib.pyplot as plt

class SinglePlaneOptimizer(object):

    def __init__(self, lensmodel, x_pos, y_pos, tol_source, params, \
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

    def _get_images(self):


        x_image, y_image = self.solver.image_position_from_source(self.srcx, self.srcy, self.lens_args_latest)

        inds = sort_image_index(x_image, y_image, self._x_pos, self._y_pos)

        return x_image[inds], y_image[inds]

    def _source_position_penalty(self, lens_args_tovary, lens_args_fixed, x_pos, y_pos):

        if len(lens_args_fixed) > 0:
            newargs = lens_args_tovary + lens_args_fixed
        else:
            newargs = lens_args_tovary

        betax, betay = self.lensModel.ray_shooting(x_pos-self.alpha_x_sub, y_pos-self.alpha_y_sub, newargs, k=self.k)

        self.srcx, self.srcy = np.mean(betax), np.mean(betay)

        return (np.std(betax)**2 + np.std(betay)**2) * self.tol_source ** -2

    def _magnification_penalty_long(self,  lens_args_tovary, lens_args_fixed, x_pos, y_pos, magnification_target, tol=0.1):

        newargs = lens_args_tovary + lens_args_fixed

        #if self.arg_list_sub is not None:

        #    newargs += self.arg_list_sub

        magnifications = np.absolute(self.lensModel.magnification(x_pos, y_pos, newargs))

        self.magnifications = magnifications

        dM = []

        for i, target in enumerate(magnification_target):
            mag_tol = tol * target
            dM.append((magnifications[i] - target) * mag_tol ** -1)

        dM = np.array(dM)

        return 0.5*np.sum(dM ** 2)

    def _magnification_penalty(self,  lens_args_tovary, lens_args_fixed, x_pos, y_pos, magnification_target, tol=0.1):

        newargs = lens_args_tovary + lens_args_fixed

        fxx_macro,fyy_macro,fxy_macro,_ = self.lensModel.hessian(x_pos,y_pos,newargs,k=self.k)

        fxx,fyy,fxy = fxx_macro+self.sub_fxx,fyy_macro+self.sub_fyy,fxy_macro+self.sub_fyy

        det_J = (1-fxx)*(1-fyy) - fxy**2

        magnifications = np.absolute(det_J**-1)

        dM = []

        for i, target in enumerate(magnification_target):
            mag_tol = tol * target
            dM.append((magnifications[i] - target) * mag_tol ** -1)

        dM = np.array(dM)

        return 0.5*np.sum(dM ** 2)

    def _centroid_penalty(self, values_dic, tol_centroid):

        d_centroid = ((values_dic[0]['center_x'] - self.centroid_0[0]) * tol_centroid ** -1) ** 2 + \
                     ((values_dic[0]['center_y'] - self.centroid_0[1]) * tol_centroid ** -1) ** 2

        return d_centroid

    def __call__(self, lens_values_tovary):

        params_fixed_dictionary = self.Params.argsfixed_todictionary()
        lens_args_tovary = self.Params.argstovary_todictionary(lens_values_tovary)

        self.lens_args_latest = lens_args_tovary+params_fixed_dictionary

        penalty = self._source_position_penalty(lens_args_tovary, params_fixed_dictionary,
                                                    self._x_pos, self._y_pos)

        if self.tol_mag is not None:
            penalty += self._magnification_penalty(lens_args_tovary,params_fixed_dictionary,
                                                   self._x_pos, self._y_pos, self.magnification_target, self.tol_mag)
        if self.tol_centroid is not None:
            penalty += self._centroid_penalty(self.Params.argstovary_todictionary(lens_args_tovary),
                                              self.tol_centroid)

        return -penalty, None