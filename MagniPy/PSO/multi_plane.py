from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from MagniPy.util import sort_image_index
import numpy as np

class MultiPlaneOptimizer(object):

    def __init__(self, lensmodel, x_pos, y_pos, tol_source, params,\
                 magnification_target, tol_mag, centroid_0, tol_centroid, k_start=0, arg_list=[], z_main=None):

        self.Params = params
        self.lensModel = lensmodel
        self.solver = LensEquationSolver(self.lensModel)
        k_start = k_start

        self.tol_source = tol_source

        self.magnification_target = magnification_target
        self.tol_mag = tol_mag

        self.centroid_0 = centroid_0
        self.tol_centroid = tol_centroid

        if k_start > 0 and len(arg_list) > 2:

            self.k = np.where(self.lensModel.redshift_list>z_main)
            k_front = np.where(self.lensModel.redshift_list<=z_main)

            self.alpha_x_sub, self.alpha_y_sub = self.lensModel.alpha(x_pos, y_pos, arg_list, k=k_front)

            self._x_pos = x_pos - self.alpha_x_sub
            self._y_pos = y_pos - self.alpha_y_sub

        else:

            self.k = None
            self._x_pos, self._y_pos = x_pos, y_pos
            self.alpha_x_sub, self.alpha_y_sub = 0, 0

    def _get_images(self):

        x_image, y_image = self.solver.image_position_from_source(self.srcx, self.srcy, self.lens_args_latest)

        inds = sort_image_index(x_image, y_image, self.x_pos + self.alpha_x_sub, self.y_pos + self.alpha_y_sub)

        return x_image[inds], y_image[inds]

    def _source_position_penalty(self, values_to_vary, lens_args_fixed, x_pos, y_pos):

        lens_args_tovary = self.Params.argstovary_todictionary(values_to_vary)

        if len(lens_args_fixed) > 0:
            newargs = lens_args_tovary + lens_args_fixed
        else:
            newargs = lens_args_tovary

        betax, betay = self.lensModel.ray_shooting(x_pos, y_pos, newargs, k=self.k)

        self.srcx, self.srcy = np.mean(betax), np.mean(betay)

        std1, std2 = np.std(betax), np.std(betay)

        return np.sqrt(std1 ** 2 + std2 ** 2) * self.tol_source ** -1

    def _magnification_penalty(self, values_to_vary, lens_args_fixed, x_pos, y_pos, magnification_target, tol=0.1):

        lens_args_tovary = self.Params.argstovary_todictionary(values_to_vary)

        if len(lens_args_fixed) > 0:
            newargs = lens_args_tovary + lens_args_fixed
        else:
            newargs = lens_args_tovary

        # magnifications = self.lensModel.magnification_finite(x_pos, y_pos, newargs,
        #                                                                 polar_grid=True,aspect_ratio=0.2,window_size=0.08,
        #                                                                 grid_number=80)

        magnifications = self.lensModel.magnification(x_pos, y_pos, newargs)

        # magnifications = self.lensModel.magnification_finite(x_pos,y_pos,newargs,source_sigma=0.0001,polar_grid=True,window_size=0.05)

        magnification_target = np.array(magnification_target)

        magnifications = (magnifications) * np.max(magnifications) ** -1

        self.magnifications = magnifications

        dM = []

        for i, target in enumerate(magnification_target):
            mag_tol = tol * target
            dM.append((magnifications[i] - target) * mag_tol ** -1)

        dM = np.array(dM)

        return np.sum(dM ** 2)

    def _get_magnifications(self, kwargslens, x_pos, y_pos):

        if hasattr(self, 'magnifications'):

            return self.magnifications

        else:
            self.magnifications = self.lensModel.magnification_finite(x_pos=x_pos, y_pos=y_pos, kwargs_lens=kwargslens,
                                                                      polar_grid=True, aspect_ratio=0.2,
                                                                      window_size=0.08, grid_number=80)
            return self.magnifications * np.max(self.magnifications) ** -1

    def _centroid_penalty(self, values_dic, tol_centroid):

        d_centroid = ((values_dic[0]['center_x'] - self.centroid_0[0]) * tol_centroid ** -1) ** 2 + \
                     ((values_dic[0]['center_y'] - self.centroid_0[1]) * tol_centroid ** -1) ** 2

        return d_centroid

    def __call__(self, lens_values_tovary):

        penalty = self._source_position_penalty(lens_values_tovary, self.Params.argsfixed_todictionary(),
                                                self._x_pos, self._y_pos)

        if self.tol_mag is not None:
            penalty += self._magnification_penalty(lens_values_tovary, self.Params.argsfixed_todictionary(),
                                                   self._x_pos, self._y_pos, self.magnification_target, self.tol_mag)
        if self.tol_centroid is not None:
            penalty += self._centroid_penalty(self.Params.argstovary_todictionary(lens_values_tovary),
                                              self.tol_centroid)

        return -penalty, None