__author__ = 'dgilman'

import os
import sys
import shutil
import tempfile
import time

import emcee
from emcee.utils import MPIPool
import numpy as np
from cosmoHammer import CosmoHammerSampler
from cosmoHammer import LikelihoodComputationChain
from cosmoHammer import MpiCosmoHammerSampler
from cosmoHammer import MpiParticleSwarmOptimizer
from cosmoHammer import ParticleSwarmOptimizer
from cosmoHammer.util import InMemoryStorageUtil
from cosmoHammer.util import MpiUtil
from Params import Params

from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from MagniPy.lensdata import Data
from MagniPy.util import sort_image_index
from MagniPy.util import cart_to_polar

class QuadPSOLike(object):

    def __init__(self, zlist, lens_list, arg_list, z_source, x_pos, y_pos, tol_source, magnification_target, tol_mag, centroid_0,
                 tol_centroid, x_tol=None, y_tol = None, chi_mode=1,k=None,multiplane=False,params_init=None):

        self.lensModel = LensModelExtensions(lens_model_list=lens_list,z_source=z_source,redshift_list=zlist,multi_plane=multiplane)

        self.extension = LensEquationSolver(self.lensModel)

        self._x_pos,self._y_pos,self.tol_source,self.magnification_target,self.tol_mag,self.centroid_0,self.tol_centroid = x_pos,y_pos,tol_source,\
                                                                                magnification_target,tol_mag,centroid_0,tol_centroid
        self.k = k

        self.Params = Params(zlist, lens_list, arg_list, params_init)

        self.lower_limit = self.Params.tovary_lower_limit

        self.upper_limit = self.Params.tovary_upper_limit

        self.chi_mode = chi_mode # source plane

        self.x_tol = x_tol
        self.y_tol = y_tol

    def _set_chi_mode(self,mode):

        if mode == 'src_plane_chi2':

            self.chi_mode = 1

        else:

            self.chi_mode = 2

    def _get_images(self, lens_args):

        lens_args_fixed = self.Params.argsfixed_todictionary()

        if hasattr(self, 'x_image') and hasattr(self,'y_image'):
            return self.x_image,self.y_image

        if len(lens_args_fixed) > 0:
            newargs = lens_args + lens_args_fixed
        else:
            newargs = lens_args

        self.x_image, self.y_image = self.extension.image_position_from_source(sourcePos_x=self.srcx,
                                                                               sourcePos_y=self.srcy, kwargs_lens=newargs)

        indexes = sort_image_index(self.x_image,self.y_image,self._x_pos,self._y_pos)
        self.x_image = self.x_image[indexes]
        self.y_image = self.y_image[indexes]

        return self.x_image[indexes],self.y_image[indexes]

    def _source_position_penalty(self,values_to_vary,lens_args_fixed, x_pos, y_pos, tol=10**-10):

        lens_args_tovary = self.Params.argstovary_todictionary(values_to_vary)

        if len(lens_args_fixed)>0:
            newargs = lens_args_tovary+lens_args_fixed
        else:
            newargs = lens_args_tovary


        betax,betay = self.lensModel.ray_shooting(x_pos,y_pos,newargs,k=self.k)

        self.srcx,self.srcy = np.mean(betax),np.mean(betay)

        dx1 = (betax[0] - betax[1])
        dx2 = (betax[0] - betax[2])
        dx3 = (betax[0] - betax[3])
        dx4 = (betax[1] - betax[2])
        dx5 = (betax[1] - betax[3])
        dx6 = (betax[2] - betax[3])

        dy1 = (betay[0] - betay[1])
        dy2 = (betay[0] - betay[2])
        dy3 = (betay[0] - betay[3])
        dy4 = (betay[1] - betay[2])
        dy5 = (betay[1] - betay[3])
        dy6 = (betay[2] - betay[3])

        dx_2 = dx1**2 + dx2 **2 +dx3 **2 + dx4**2 + dx5**2 + dx6**2
        dy_2 = dy1**2 + dy2 **2 +dy3 **2 + dy4**2 + dy5**2 + dy6**2

        dr = dx_2 + dy_2

        return dr * tol ** -2

    def _img_position_penalty(self,values_to_vary,lens_args_fixed,x_pos,y_pos,xtol,ytol):

        lens_args_tovary = self.Params.argstovary_todictionary(values_to_vary)

        if len(lens_args_fixed)>0:
            newargs = lens_args_tovary+lens_args_fixed
        else:
            newargs = lens_args_tovary

        betax, betay = self.lensModel.ray_shooting(x_pos, y_pos, newargs, k=self.k)

        self.srcx, self.srcy = np.mean(betax), np.mean(betay)

        ximg,yimg = self.extension.image_position_from_source(sourcePos_x=self.srcx,
                                                                               sourcePos_y=self.srcy, kwargs_lens=newargs)

        nimg = len(self._x_pos)

        if len(ximg) != nimg:
            return 1e+11

        sort_inds = sort_image_index(ximg,yimg,self._x_pos,self._y_pos)
        self.x_image,self.y_image = ximg[sort_inds],yimg[sort_inds]

        dx,dy = [],[]
        for i in range(0,nimg):
            dx.append((self.x_image[i] - self._x_pos[i])**2*xtol[i]**-2)
            dy.append((self.y_image[i] - self._y_pos[i]) ** 2*xtol[i]**-2)

        return 0.5*np.sum(np.array(dx)+np.array(dy))

    def _magnification_penalty(self,values_to_vary,lens_args_fixed,x_pos, y_pos, magnification_target, tol=0.1):

        lens_args_tovary = self.Params.argstovary_todictionary(values_to_vary)

        if len(lens_args_fixed)>0:
            newargs = lens_args_tovary+lens_args_fixed
        else:
            newargs = lens_args_tovary

        #magnifications = self.lensModel.magnification_finite(x_pos, y_pos, newargs,
        #                                                                 polar_grid=True,aspect_ratio=0.2,window_size=0.08,
        #                                                                 grid_number=80)

        magnifications = self.lensModel.magnification(x_pos,y_pos,newargs)

        magnification_target = np.array(magnification_target)

        magnifications = (magnifications)*np.max(magnifications)**-1

        self.magnifications = magnifications

        dM = []

        for i,target in enumerate(magnification_target):
            mag_tol = tol*target
            dM.append((magnifications[i] - target)*mag_tol**-1)

        dM = np.array(dM)

        return np.sum(dM**2)

    def get_magnifications(self,kwargslens,x_pos,y_pos):

        if hasattr(self,'magnifications'):

            return self.magnifications

        else:
            self.magnifications = self.lensModel.magnification_finite(x_pos=x_pos,y_pos=y_pos,kwargs_lens=kwargslens,
                                                                      polar_grid=True,aspect_ratio=0.2,window_size=0.08,grid_number=80)
            return self.magnifications*np.max(self.magnifications)**-1

    def compute_img_plane(self,args):

        x_image,y_image = self.extension.image_position_from_source(sourcePos_x=self.srcx,sourcePos_y=self.srcy,kwargs_lens=args)

        inds_sorted = sort_image_index(x_image,y_image,self._x_pos,self._y_pos)

        self.x_image,self.y_image = x_image,y_image

        return x_image[inds_sorted],y_image[inds_sorted],self.get_magnifications(args,x_image,y_image)[inds_sorted],\
               [self.srcx,self.srcy]

    def _centroid_penalty(self,values_dic,tol_centroid):

        d_centroid = ((values_dic[0]['center_x'] - self.centroid_0[0])*tol_centroid**-1)**2 + \
                     ((values_dic[0]['center_y'] - self.centroid_0[1])*tol_centroid**-1)**2

        return d_centroid

    def __call__(self,lens_values_tovary):

        if self.chi_mode == 1:

            penalty = self._source_position_penalty(lens_values_tovary,self.Params.argsfixed_todictionary(),self._x_pos,self._y_pos,self.tol_source)

        else:

            penalty = self._img_position_penalty(lens_values_tovary,self.Params.argsfixed_todictionary(),self._x_pos,
                                                 self._y_pos,self.x_tol,self.y_tol)

        if self.tol_mag is not None:
            penalty += self._magnification_penalty(lens_values_tovary,self.Params.argsfixed_todictionary(),
                                                        self._x_pos,self._y_pos,self.magnification_target,self.tol_mag)
        if self.tol_centroid is not None:
            penalty += self._centroid_penalty(self.Params.argstovary_todictionary(lens_values_tovary),self.tol_centroid)

        return -penalty,None


class QuadSampler(object):
    """
    class which executes the different sampling  methods
    """

    def __init__(self, zlist, lens_list, arg_list, z_source,x_pos,y_pos,tol_source,magnification_target,
                 tol_mag=None,tol_centroid=None,centroid_0=None,x_tol=None,y_tol=None,params_init=None):
        """
        initialise the classes of the chain and for parameter options
        """
        assert len(x_pos) == 4
        assert len(y_pos) == 4
        assert len(magnification_target) == len(x_pos)

        self.chain = QuadPSOLike(zlist, lens_list, arg_list, z_source, x_pos,y_pos,tol_source,magnification_target,tol_mag,
                                 k=None,centroid_0=centroid_0,tol_centroid=tol_centroid,x_tol=x_tol,y_tol=y_tol,params_init=params_init)

    def pso(self, n_particles, n_iterations,run_mode='src_plane_chi2',):

        assert run_mode in ['img_plane_chi2','src_plane_chi2']

        optimized_args = self._pso(n_particles,n_iterations,chi_mode=run_mode)

        ximg,yimg = self.chain._get_images(optimized_args)

        return optimized_args,[self.chain.srcx, self.chain.srcy], {'x_image':ximg,'y_image':yimg}


    def _pso(self, n_particles, n_iterations, lowerLimit=None, upperLimit=None, threadCount=1,mpi=False,
             chi_mode='src_plane_chi2'):

        """
        returns the best fit for the lense model on catalogue basis with particle swarm optimizer
        """

        self.chain._set_chi_mode(chi_mode)

        if lowerLimit is None or upperLimit is None:
            lowerLimit, upperLimit = self.chain.lower_limit, self.chain.upper_limit
            print("PSO initialises its particles with default values")
        else:
            lowerLimit = np.maximum(lowerLimit, self.chain.lower_limit)
            upperLimit = np.minimum(upperLimit, self.chain.upper_limit)
        if mpi is True:
            pso = MpiParticleSwarmOptimizer(self.chain, lowerLimit, upperLimit, n_particles, threads=1)
            self.PSO = pso
            if pso.isMaster():
                print('MPI option chosen')

        pso = ParticleSwarmOptimizer(self.chain, lowerLimit, upperLimit, n_particles, threads=threadCount)

        X2_list = []
        vel_list = []
        pos_list = []
        time_start = time.time()

        num_iter = 0
        for swarm in pso.sample(n_iterations):
            X2_list.append(pso.gbest.fitness * 2)
            vel_list.append(pso.gbest.velocity)
            pos_list.append(pso.gbest.position)
            num_iter += 1
            if pso.isMaster():
                if num_iter % 10 == 0:
                    print(num_iter)
        if not mpi:
            result = pso.gbest.position
        else:
            result = MpiUtil.mpiBCast(pso.gbest.position)

        return self.chain.Params.argstovary_todictionary(result)
