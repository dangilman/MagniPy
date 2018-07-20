from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.Solver.solver4point import Solver4Point
from MagniPy.util import sort_image_index
import numpy as np
import matplotlib.pyplot as plt

class SinglePlaneOptimizer(object):

    def __init__(self, lensmodel, x_pos, y_pos, tol_source, params, \
                 magnification_target, tol_mag, centroid_0, tol_centroid, k_start=0, arg_list=[],return_sign=-1,
                 mag_penalty=False,return_mode='PSO'):

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

        self._return_mode = return_mode

        self._compute_mags = mag_penalty

        self.mag_penalty, self.src_penalty, self.parameters = [], [], []

        self._counter = 1

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

    def _init_particles(self,n_particles,n_iterations):

        self._n_total_iter = n_iterations*n_particles
        self._n_particles = n_particles
        self._mag_penalty_switch = 1

    def _get_images(self,args):

        srcx, srcy = self.lensModel.ray_shooting(self._x_pos, self._y_pos, args, None)

        self.srcx, self.srcy = np.mean(srcx), np.mean(srcy)
        x_image, y_image = self.solver.image_position_from_source(self.srcx, self.srcy,args,precision_limit=10**-11)

        return x_image, y_image

    def _source_position_penalty(self, lens_args):

        betax, betay = self.lensModel.ray_shooting(self._x_pos - self.alpha_x_sub, self._y_pos - self.alpha_y_sub, lens_args, k=self.k)
        betax,betay = betax + self.alpha_x_sub, betay + self.alpha_y_sub

        dx = ((betax[0] - betax[1]) ** 2 + (betax[0] - betax[2]) ** 2 + (betax[0] - betax[3]) ** 2 + (
                    betax[1] - betax[2]) ** 2 +
              (betax[1] - betax[3]) ** 2 + (betax[2] - betax[3]) ** 2)
        dy = ((betay[0] - betay[1]) ** 2 + (betay[0] - betay[2]) ** 2 + (betay[0] - betay[3]) ** 2 + (
                    betay[1] - betay[2]) ** 2 +
              (betay[1] - betay[3]) ** 2 + (betay[2] - betay[3]) ** 2)

        return 0.5*(dx+dy)*self.tol_source**-2

    def _jacobian(self, x_pos, y_pos,args,k, fxx=None, fyy = None, fxy = None):

        if fxx is None:
            fxx,fyy,fxy,_ = self.lensModel.hessian(x_pos,y_pos,args,k)

        return np.array([[1-fxx, -fxy],[-fxy, 1-fyy]])

    def _magnification_penalty(self,  args, magnification_target, tol=0.1):

        fxx_macro,fyy_macro,fxy_macro,_ = self.lensModel.hessian(self._x_pos,self._y_pos,args,k=self.k)

        fxx,fyy,fxy = fxx_macro+self.sub_fxx,fyy_macro+self.sub_fyy,fxy_macro+self.sub_fyy

        det_J = (1-fxx)*(1-fyy) - fxy**2

        magnifications = np.absolute(det_J**-1)

        #fxx, fyy, fxy,_ = self.lensModel.hessian(self._x_pos,self._y_pos,args)
        #det_J = (1 - fxx) * (1 - fyy) - fxy ** 2
        #magnifications = np.absolute(det_J ** -1)

        magnifications *= max(magnifications)**-1



        dM = []

        for i, target in enumerate(magnification_target):
            mag_tol = tol * target
            dM.append((magnifications[i] - target) * mag_tol ** -1)

        dM = np.array(dM)

        return 0.5*np.sum(dM ** 2)

    def _centroid_penalty(self, values_dic, tol_centroid):

        dx = (values_dic[0]['center_x'] - self.centroid_0[0])*self.tol_centroid**-1
        dy = (values_dic[0]['center_y'] - self.centroid_0[1])*self.tol_centroid**-1

        return 0.5*(dx**2+dy**2)

    def _log(self,src_penalty,mag_penalty):

        if mag_penalty is None:
            mag_penalty = np.inf
        if src_penalty is None:
            src_penalty = np.inf

        self.src_penalty.append(src_penalty)
        self.mag_penalty.append(mag_penalty)
        self.parameters.append(self.lens_args_latest)

    def _compute_mags_criterion(self):

        if self._compute_mags:
            return True

        if self._counter > self._n_particles and np.mean(self.src_penalty[-self._n_particles:]) < 1:
            return True
        else:
            return False

    def __call__(self, lens_values_tovary,src_penalty=None,mag_penalty=None,centroid_penalty=None):

        self._counter += 1

        params_fixed = self.Params.argsfixed_todictionary()
        lens_args_tovary = self.Params.argstovary_todictionary(lens_values_tovary)

        if self.tol_source is not None:
            src_penalty = self._source_position_penalty(lens_args_tovary+params_fixed)

            self._compute_mags = self._compute_mags_criterion()

        if self._compute_mags and self.tol_mag is not None:
            mag_penalty = self._magnification_penalty(lens_args_tovary + params_fixed, self.magnification_target,
                                                      self.tol_mag)

        if self.tol_centroid is not None:
            centroid_penalty = self._centroid_penalty(lens_args_tovary, self.tol_centroid)

        _penalty = [src_penalty, mag_penalty, centroid_penalty]

        penalty = 0
        for pen in _penalty:
            if pen is not None:
                penalty += pen

        if self._counter % 500 == 0:

            print 'source penalty: ', src_penalty
            if self.mag_penalty is not None:
                print 'mag penalty: ', mag_penalty

        self.lens_args_latest = lens_args_tovary + params_fixed

        self._log(src_penalty, mag_penalty)

        if self._return_mode == 'PSO':
            return -1 * penalty, None
        else:
            return penalty