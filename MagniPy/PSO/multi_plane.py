from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from MagniPy.util import sort_image_index
import numpy as np
import matplotlib.pyplot as plt


class MultiPlaneOptimizer(object):

    def __init__(self,lensmodel_full,all_args,lensmodel_main,main_args,lensmodel_front,front_args,lensmodel_back,
                 back_args,x_pos, y_pos, tol_source, Params, magnification_target,
                 tol_mag, centroid_0, tol_centroid, z_main, z_src,comoving_distances,comoving_distances_ij,reduced_to_phys_factors):

        self.Params = Params
        self.lensModel = lensmodel_full
        self.all_lensmodel_args = all_args

        self.lensModel_main = lensmodel_main
        self.main_args = main_args

        self.lensModel_front = lensmodel_front
        self.front_args = front_args

        self.lensModel_back = lensmodel_back
        self.back_args = back_args

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
            self.sub_fxx_front,self.sub_fyy_front,self.sub_fxy_front,self.sub_fyx_front = \
                self.lensModel_front.hessian(self._x_pos,self._y_pos,front_args)
        else:
            self.alpha_y_sub_front,self.alpha_y_sub_front = 0,0
            self.sub_fxx_front, self.sub_fyy_front, self.sub_fxy_front, self.sub_fyx_front = 0, 0, 0, 0

        self.comoving_distances = comoving_distances
        self.comoving_ij = comoving_distances_ij
        self.reduced_to_phys_factors = reduced_to_phys_factors

    def _shoot_through_lensmodel(self,x_pos,y_pos,args_main):

        """
        three lens models; shoot through them a-la multi-plane
        :param x_pos: observed sky position (arcsec)
        :param y_pos: observed sky position (arcsec)
        :param tovary_args:
        :return:
        """
        x,y = np.zeros_like(x_pos),np.zeros_like(y_pos)
        alpha_x,alpha_y = x_pos,y_pos

        delta_T = self.comoving_distances_ij[0]
        x = x + alpha_x*delta_T
        y = y + alpha_y*delta_T
        alpha_x, alpha_y = self._add_deflection(x,y,alpha_x,alpha_y,self.lensModel_main,
                                                      args_main,0,xdef=self.alpha_x_sub_front,
                                                      ydef=self.alpha_y_sub_front)

        delta_T = self.comoving_distances_ij[1]
        x = x + alpha_x * delta_T
        y = y + alpha_y * delta_T
        alpha_x, alpha_y = self._add_deflection(x, y, alpha_x, alpha_y,self.lensModel_main, args_main, 1)

        delta_T = self.comoving_ij[2]
        x = x + alpha_x*delta_T
        y = y + alpha_y*delta_T
        beta_x,beta_y = x*self.comoving_distances[2]**-1,y*self.comoving_distances[2]**-1

        return beta_x,beta_y


    def _add_deflection(self,x,y,alpha_x,alpha_y,model,kwargs_lens,i,xdef=0,ydef=0):

        theta_x = x*self.comoving_distances[i]**-1
        theta_y = y*self.comoving_distances[i]**-1

        alphax_red_macro,alphay_red_macro = model.alpha(theta_x,theta_y,kwargs_lens)
        alphax_red = xdef + alphax_red_macro
        alphay_red = ydef + alphay_red_macro

        alpha_x_phys = alphax_red * self.reduced_to_phys_factors[i]
        alpha_y_phys = alphay_red * self.reduced_to_phys_factors[i]

        return alpha_x - alpha_x_phys, alpha_y - alpha_y_phys

    def _source_position_penalty(self, lens_args_tovary, x_pos, y_pos):

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

        srcx, srcy = self.lensModel.lens_model.ray_shooting(self._x_pos, self._y_pos, self.all_lensmodel_args, None)

        solver = LensEquationSolver(self.lensModel)
        self.srcx, self.srcy = np.mean(srcx), np.mean(srcy)
        x_image, y_image = solver.image_position_from_source(self.srcx, self.srcy, self.all_lensmodel_args)

        return x_image, y_image

    def _magnification_penalty(self, lens_args_tovary, lens_args_fixed, x_pos, y_pos, magnification_target, tol):

        pass

    def _centroid_penalty(self, values_dic, tol_centroid):

        d_centroid = ((values_dic[0]['center_x'] - self.centroid_0[0]) * tol_centroid ** -1) ** 2 + \
                     ((values_dic[0]['center_y'] - self.centroid_0[1]) * tol_centroid ** -1) ** 2

        return 0.5 * d_centroid

    def __call__(self, lens_values_tovary,sign_swith=-1):

        params_fixed_dictionary = self.Params.argsfixed_todictionary()
        lens_args_tovary = self.Params.argstovary_todictionary(lens_values_tovary)

        self.all_lensmodel_args[0:self.Params.Nprofiles_to_vary] = lens_args_tovary
        print lens_args_tovary
        if self.tol_source is not None:
            penalty = self._source_position_penalty(lens_args_tovary, params_fixed_dictionary,
                                                    self._x_pos, self._y_pos)

        if self.tol_mag is not None:
            penalty += self._magnification_penalty(lens_args_tovary, params_fixed_dictionary,
                                                   self._x_pos, self._y_pos, self.magnification_target, self.tol_mag)
        if self.tol_centroid is not None:
            penalty += self._centroid_penalty(lens_args_tovary, self.tol_centroid)

        if sign_swith == -1:
            return sign_swith*penalty, None
        else:
            return sign_swith*penalty