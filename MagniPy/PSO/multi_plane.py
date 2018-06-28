from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from MagniPy.util import sort_image_index
import numpy as np
import matplotlib.pyplot as plt


class MultiPlaneOptimizer(object):

    def __init__(self,lensmodel_full,all_args,lensmodel_main,main_args,lensmodel_front,front_args,lensing_components_back,
                 lensModel_interpolated_back,x_pos, y_pos, tol_source, params, \
                 magnification_target, tol_mag, centroid_0, tol_centroid, z_main, z_src):

        self.Params = params

        self.lensmodel = lensmodel_full
        self.all_lensmodel_args = all_args

        self.lensModel_main = lensmodel_main
        self.main_args = main_args

        self.lensModel_front = lensmodel_front
        self.front_args = front_args

        self.lensModel_back = lensModel_interpolated_back
        self.back_args = lensing_components_back

        self.tol_source = tol_source

        self.magnification_target = magnification_target
        self.tol_mag = tol_mag

        self.centroid_0 = centroid_0
        self.tol_centroid = tol_centroid

        self._x_pos = x_pos
        self._y_pos = y_pos

        self.z_main = z_main
        self.zsrc = z_src

        if len(front_args) > 0:
            self.alpha_x_sub_front, self.alpha_y_sub_front = self.lensModel_front.alpha(self._x_pos,self._y_pos,front_args)
        else:
            self.alpha_y_sub_front,self.alpha_y_sub_front = 0,0

    def _shoot_through_lensmodel(self,x_pos,y_pos,tovary_args):

        x_out,y_out,alpha_x_out,alpha_y_out = self.lensModel_main.lens_model.ray_shooting_partial(0, 0, x_pos, y_pos,
                           0, self.z_main, tovary_args)

        alpha_x_out += self.alpha_x_sub_front
        alpha_y_out += self.alpha_y_sub_front

        if self.lensModel_back is None:
            if not hasattr(self,'D_T_lens_source'):
                self.D_T_lens_source = self.lensmodel.lens_model._cosmo_bkg.T_xy(self.z_main, self.zsrc)
            x_out,y_out = x_out - alpha_x_out*self.D_T_lens_source,y_out-alpha_y_out*self.D_T_lens_source
        else:
            x_out, y_out, alpha_x_out, alpha_y_out = self.lensModel_back.lens_model.ray_shooting_partial(x=x_out, y=y_out, alpha_x=alpha_x_out,
                                    alpha_y=alpha_y_out,z_start=self.z_main,z_stop=self.zsrc,
                                    kwargs_lens=self.back_args)

        beta_x, beta_y = self.lensModel_back._co_moving2angle_source(x_out, y_out)

        return beta_x,beta_y


    def _source_position_penalty(self, lens_args_tovary, lens_args_fixed, x_pos, y_pos):

        betax, betay = self._shoot_through_lensmodel(x_pos,y_pos,lens_args_tovary)

        dx = ((betax[0] - betax[1]) ** 2 + (betax[0] - betax[2]) ** 2 + (betax[0] - betax[3]) ** 2 + (
                    betax[1] - betax[2]) ** 2 +
              (betax[1] - betax[3]) ** 2 + (betax[2] - betax[3]) ** 2)
        dy = ((betay[0] - betay[1]) ** 2 + (betay[0] - betay[2]) ** 2 + (betay[0] - betay[3]) ** 2 + (
                    betay[1] - betay[2]) ** 2 +
              (betay[1] - betay[3]) ** 2 + (betay[2] - betay[3]) ** 2)

        return 0.5 * (dx + dy) * self.tol_source ** -2

        # return (np.std(betax)**2 + np.std(betay)**2) * self.tol_source ** -2

    def _get_images(self):

        srcx, srcy = self.lensModel.ray_shooting(self._x_pos, self._y_pos, self.lens_args_latest, None)

        solver = LensEquationSolver(self.lensmodel)
        self.srcx, self.srcy = np.mean(srcx), np.mean(srcy)
        x_image, y_image = solver.image_position_from_source(self.srcx, self.srcy, self.lens_args_latest)

        return x_image, y_image


    def _jacobian(self, x_pos, y_pos, args, k, fxx=None, fyy=None, fxy=None):

        if fxx is None:
            fxx, fyy, fxy, _ = self.lensModel.hessian(x_pos, y_pos, args, k)

        return np.array([[1 - fxx, -fxy], [-fxy, 1 - fyy]])

    def _magnification_penalty(self, lens_args_tovary, lens_args_fixed, x_pos, y_pos, magnification_target, tol):

        newargs = lens_args_tovary + lens_args_fixed

        fxx_macro, fyy_macro, fxy_macro, _ = self.lensModel.hessian(x_pos, y_pos, newargs, k=self.k)

        fxx, fyy, fxy = fxx_macro + self.sub_fxx_front, fyy_macro + self.sub_fyy_front, fxy_macro + self.sub_fyy_front

        det_J = (1 - fxx) * (1 - fyy) - fxy ** 2

        magnifications = np.absolute(det_J ** -1)

        magnifications *= max(magnifications) ** -1

        dM = []

        for i, target in enumerate(magnification_target):
            mag_tol = tol * target
            dM.append((magnifications[i] - target) * mag_tol ** -1)

        dM = np.array(dM)

        return 0.5 * np.sum(dM ** 2)

    def _centroid_penalty(self, values_dic, tol_centroid):

        d_centroid = ((values_dic[0]['center_x'] - self.centroid_0[0]) * tol_centroid ** -1) ** 2 + \
                     ((values_dic[0]['center_y'] - self.centroid_0[1]) * tol_centroid ** -1) ** 2

        return 0.5 * d_centroid

    def __call__(self, lens_values_tovary):

        params_fixed_dictionary = self.Params.argsfixed_todictionary()
        lens_args_tovary = self.Params.argstovary_todictionary(lens_values_tovary)

        self.lens_args_latest = lens_args_tovary + params_fixed_dictionary

        if self.tol_source is not None:
            penalty = self._source_position_penalty(lens_args_tovary, params_fixed_dictionary,
                                                    self._x_pos, self._y_pos)

        if self.tol_mag is not None:
            penalty += self._magnification_penalty(lens_args_tovary, params_fixed_dictionary,
                                                   self._x_pos, self._y_pos, self.magnification_target, self.tol_mag)
        if self.tol_centroid is not None:
            penalty += self._centroid_penalty(lens_args_tovary, self.tol_centroid)

        return -penalty, None