from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.Solver.solver4point import Solver4Point
from MagniPy.util import sort_image_index
import numpy as np
import matplotlib.pyplot as plt

class SinglePlaneOptimizer_SHEAR(object):

    def __init__(self, lensmodel, x_pos, y_pos, tol_source, params, \
                 magnification_target, tol_mag, centroid_0, tol_centroid, arg_list=[]):

        self.lensModel = lensmodel
        self.tol_source = tol_source
        self.magnification_target = magnification_target
        self.tol_mag = tol_mag
        self.arg_list = arg_list

        self.x_pos,self.y_pos = x_pos,y_pos

        if len(arg_list) > 2:
            self.k = [0,1]
            self.k_sub = np.arrange(2,len(arg_list))
            self.alpha_x_sub,self.alpha_y_sub = self.lensModel.ray_shooting(x_pos, y_pos, self.arg_list, k=self.k_sub)
            self.sub_fxx, self.sub_fyy, self.sub_fxy, _ = self.lensModel.hessian(x_pos, y_pos, arg_list, k=self.k_sub)
        else:
            self.k_sub = []
            self.k = [0,1]
            self.alpha_x_sub, self.alpha_y_sub = 0,0
            self.sub_fxx, self.sub_fyy, self.sub_fxy = 0,0,0

        self._x_pos, self._y_pos = x_pos - self.alpha_x_sub, y_pos - self.alpha_y_sub
        self.solver = Solver4Point(self.lensModel,solver_type='PROFILE_SHEAR')

    def replace_shear(self,args, shear_e1_e2):

        args[1] = {'e1':shear_e1_e2[0],'e2':shear_e1_e2[1]}
        self.arg_list = args
        return args

    def source_position_penaly(self):

        betax,betay = self.lensModel.ray_shooting(self._x_pos,self._y_pos,self.arg_list)

        betax += self.alpha_x_sub
        betay += self.alpha_y_sub

        dx = ((betax[0] - betax[1]) ** 2 + (betax[0] - betax[2]) ** 2 + (betax[0] - betax[3]) ** 2 + (
                    betax[1] - betax[2]) ** 2 +
              (betax[1] - betax[3]) ** 2 + (betax[2] - betax[3]) ** 2)
        dy = ((betay[0] - betay[1]) ** 2 + (betay[0] - betay[2]) ** 2 + (betay[0] - betay[3]) ** 2 + (
                    betay[1] - betay[2]) ** 2 +
              (betay[1] - betay[3]) ** 2 + (betay[2] - betay[3]) ** 2)

        return 0.5 * (dx + dy) * self.tol_source ** -2

    def _magnification_penalty(self):

        fxx_macro,fyy_macro,fxy_macro,_ = self.lensModel.hessian(self.x_pos,self.y_pos,self.arg_list,k=self.k)

        fxx,fyy,fxy = fxx_macro+self.sub_fxx,fyy_macro+self.sub_fyy,fxy_macro+self.sub_fyy

        det_J = (1-fxx)*(1-fyy) - fxy**2

        magnifications = np.absolute(det_J**-1)

        magnifications *= max(magnifications)**-1

        dM = []

        for i, target in enumerate(self.magnification_target):
            mag_tol = self.tol_mag * target
            dM.append((magnifications[i] - target) * mag_tol ** -1)

        dM = np.array(dM)

        return 0.5*np.sum(dM ** 2)

    def __call__(self, shear_e1_e2):

        self.replace_shear(self.arg_list, shear_e1_e2)
        newkwargs,_ = self.solver.constraint_lensmodel(self._x_pos,self._y_pos, self.arg_list)
        self.arg_list = newkwargs

        if self.tol_mag is not None:

            penalty = self._magnification_penalty()

        return -penalty,None




class SinglePlaneOptimizer(object):

    def __init__(self, lensmodel, x_pos, y_pos, tol_source, params, \
                 magnification_target, tol_mag, centroid_0, tol_centroid, k_start=0, arg_list=[],return_sign=-1,
                 return_array = False):

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

        self.return_sign = return_sign
        self.retrun_array = return_array

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

        srcx, srcy = self.lensModel.ray_shooting(self._x_pos, self._y_pos, self.lens_args_latest, None)

        self.srcx, self.srcy = np.mean(srcx), np.mean(srcy)
        x_image, y_image = self.solver.image_position_from_source(self.srcx, self.srcy, self.lens_args_latest)

        return x_image, y_image

    def _source_position_penalty(self, lens_args_tovary, lens_args_fixed, x_pos, y_pos):

        if len(lens_args_fixed) > 0:
            newargs = lens_args_tovary + lens_args_fixed
        else:
            newargs = lens_args_tovary

        betax, betay = self.lensModel.ray_shooting(x_pos-self.alpha_x_sub, y_pos-self.alpha_y_sub, newargs, k=self.k)
        betax,betay = betax + self.alpha_x_sub, betay + self.alpha_y_sub

        if self.retrun_array:
            dx = np.array([betax[0] - betax[1],betax[0] - betax[2],betax[0] - betax[3],betax[1] - betax[2],
                           betax[1] - betax[3],betax[2] - betax[3]])
            dy = np.array([betay[0] - betay[1],betay[0] - betay[2],betay[0] - betay[3],betay[1] - betay[2],betay[1] - betay[3],
                           betay[2] - betay[3]])

            return np.append(dx,dy)*self.tol_source**-1

        dx = ((betax[0] - betax[1]) ** 2 + (betax[0] - betax[2]) ** 2 + (betax[0] - betax[3]) ** 2 + (
                    betax[1] - betax[2]) ** 2 +
              (betax[1] - betax[3]) ** 2 + (betax[2] - betax[3]) ** 2)
        dy = ((betay[0] - betay[1]) ** 2 + (betay[0] - betay[2]) ** 2 + (betay[0] - betay[3]) ** 2 + (
                    betay[1] - betay[2]) ** 2 +
              (betay[1] - betay[3]) ** 2 + (betay[2] - betay[3]) ** 2)

        return 0.5*(dx+dy)*self.tol_source**-2

        #return (np.std(betax)**2 + np.std(betay)**2) * self.tol_source ** -2

        return 0.5*np.sum(dM ** 2)

    def _jacobian(self, x_pos, y_pos,args,k, fxx=None, fyy = None, fxy = None):

        if fxx is None:
            fxx,fyy,fxy,_ = self.lensModel.hessian(x_pos,y_pos,args,k)

        return np.array([[1-fxx, -fxy],[-fxy, 1-fyy]])

    def _magnification_penalty(self,  lens_args_tovary, lens_args_fixed, x_pos, y_pos, magnification_target, tol=0.1):

        newargs = lens_args_tovary + lens_args_fixed

        fxx_macro,fyy_macro,fxy_macro,_ = self.lensModel.hessian(x_pos,y_pos,newargs,k=self.k)

        fxx,fyy,fxy = fxx_macro+self.sub_fxx,fyy_macro+self.sub_fyy,fxy_macro+self.sub_fyy

        det_J = (1-fxx)*(1-fyy) - fxy**2

        magnifications = np.absolute(det_J**-1)

        magnifications *= max(magnifications)**-1

        dM = []

        for i, target in enumerate(magnification_target):
            mag_tol = tol * target
            dM.append((magnifications[i] - target) * mag_tol ** -1)

        dM = np.array(dM)

        if self.retrun_array:
            return dM

        return 0.5*np.sum(dM ** 2)

    def _centroid_penalty(self, values_dic, tol_centroid):

        dx = (values_dic[0]['center_x'] - self.centroid_0[0])*self.tol_centroid**-1
        dy = (values_dic[0]['center_y'] - self.centroid_0[1])*self.tol_centroid**-1

        if self.retrun_array:
            return np.append(dx,dy)

        return 0.5*(dx**2+dy**2)

    def __call__(self, lens_values_tovary):

        params_fixed_dictionary = self.Params.argsfixed_todictionary()
        lens_args_tovary = self.Params.argstovary_todictionary(lens_values_tovary)

        self.lens_args_latest = lens_args_tovary+params_fixed_dictionary

        if self.retrun_array:

            penalty = None

            if self.tol_source is not None:
                _penalty = self._source_position_penalty(lens_args_tovary, params_fixed_dictionary,
                                                        self._x_pos, self._y_pos)
                if penalty is None:
                    penalty = _penalty
                else:
                    penalty = np.append(penalty,_penalty)

            if self.tol_mag is not None:

                _penalty = self._magnification_penalty(lens_args_tovary, params_fixed_dictionary,
                                                       self._x_pos, self._y_pos, self.magnification_target,
                                                       self.tol_mag)
                if penalty is None:
                    penalty = _penalty
                else:
                    penalty = np.append(penalty,_penalty)

            if self.tol_centroid is not None:

                _penalty = self._centroid_penalty(lens_args_tovary, self.tol_centroid)

                if penalty is None:
                    penalty = _penalty
                else:
                    penalty = np.append(penalty,_penalty)

            if self.return_sign == -1:
                return -penalty, None
            else:
                return penalty

        else:

            penalty = 0

            if self.tol_source is not None:
                penalty += self._source_position_penalty(lens_args_tovary, params_fixed_dictionary,
                                                        self._x_pos, self._y_pos)

            if self.tol_mag is not None:
                penalty += self._magnification_penalty(lens_args_tovary,params_fixed_dictionary,
                                                       self._x_pos, self._y_pos, self.magnification_target, self.tol_mag)
            if self.tol_centroid is not None:
                penalty += self._centroid_penalty(lens_args_tovary,self.tol_centroid)

            if self.return_sign == -1:
                return -penalty,None
            else:

                return penalty