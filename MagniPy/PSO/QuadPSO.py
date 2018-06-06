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
                 tol_centroid,k=None,multiplane=False):

        self.lensModel = LensModelExtensions(lens_model_list=lens_list,z_source=z_source,redshift_list=zlist,multi_plane=multiplane)

        self.extension = LensEquationSolver(self.lensModel)

        self._x_pos,self._y_pos,self.tol_source,self.magnification_target,self.tol_mag,self.centroid_0,self.tol_centroid = x_pos,y_pos,tol_source,\
                                                                                magnification_target,tol_mag,centroid_0,tol_centroid
        self.k = k

        self.Params = Params(zlist, lens_list, arg_list)

        self.lower_limit = self.Params.tovary_lower_limit

        self.upper_limit = self.Params.tovary_upper_limit

    #def _img_to_source(self,):

    def _source_position_penalty(self,values_to_vary,lens_args_fixed, x_pos, y_pos, tol=10**-10):

        lens_args_tovary = self.Params.argstovary_todictionary(values_to_vary)

        if len(lens_args_fixed)>0:
            newargs = lens_args_tovary + [lens_args_fixed]
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

    def _magnification_penalty(self,values_to_vary,lens_args_fixed,x_pos, y_pos, magnification_target, tol=0.1):

        lens_args_tovary = self.Params.argstovary_todictionary(values_to_vary)

        if len(lens_args_fixed) > 0:
            newargs = lens_args_tovary + [lens_args_fixed]
        else:
            newargs = lens_args_tovary

        magnifications = self.lensModel.magnification_finite(x_pos, y_pos, newargs,
                                                                         polar_grid=True,aspect_ratio=0.2,window_size=0.08,
                                                                         grid_number=80)

        magnification_target = np.array(magnification_target)*np.max(magnification_target)**-1

        magnifications = (magnifications)*np.max(magnifications)**-1

        self.magnifications = magnifications

        dM = (magnifications - np.array(magnification_target))*tol**-1

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

        penalty = self._source_position_penalty(lens_values_tovary,self.Params.argsfixed_todictionary(),self._x_pos,self._y_pos,self.tol_source)

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
                 tol_mag=None,tol_centroid=None,centroid_0=None):
        """
        initialise the classes of the chain and for parameter options
        """
        assert len(x_pos) == 4
        assert len(y_pos) == 4
        assert len(magnification_target) == len(x_pos)

        self.chain = QuadPSOLike(zlist, lens_list, arg_list, z_source, x_pos,y_pos,tol_source,magnification_target,tol_mag,
                                 k=None,centroid_0=centroid_0,tol_centroid=tol_centroid)

    def pso(self, n_particles, n_iterations,run_mode='src_plane_chi2',sigma_posx=[0.003]*4,sigma_posy=[0.003]*4,
            N_iter_max=4):

        assert run_mode in ['img_plane_chi2','src_plane_chi2']

        if run_mode == 'img_plane_chi2':

            optimized_args = self.fit_image_plane(n_particles, n_iterations,sigma_posx=sigma_posx,sigma_posy=sigma_posy,
                                 N_iter_max=N_iter_max)

            return optimized_args, [self.chain.srcx, self.chain.srcy], {'x_image':self.chain.x_image,'y_image':self.chain.y_image}

        elif run_mode == 'src_plane_chi2':

            optimized_args = self._pso(n_particles, n_iterations)

            return optimized_args,[self.chain.srcx,self.chain.srcy],None

    def fit_image_plane(self,n_particles, n_iterations,sigma_posx=[0.003]*4,sigma_posy=[0.003]*4,
            N_iter_max=5,n_iterations_secondary=50):

        N_iter = 0

        optimized_args = self._pso(n_particles, n_iterations)

        d_of_f = 7

        while True:

            x_image, y_image, mags, src = self.chain.compute_img_plane(optimized_args)

            self.chain.tol_source *= 0.1

            dx2,dy2 = [],[]

            for i,xi in enumerate(x_image):
                dx2.append(((xi - self.chain._x_pos[i])*sigma_posx[i]**-1)**2)
            for i,yi in enumerate(y_image):
                dy2.append(((yi - self.chain._y_pos[i])*sigma_posy[i]**-1)**2)

            img_chi2 = d_of_f**-1*np.sum(np.array(dx2) + np.array(dy2))**0.5

            print img_chi2

            if img_chi2 <= 2:
                break
            elif N_iter > N_iter_max:
                break
            else:
                N_iter += 1

                optimized_args = self._pso(n_particles,n_iterations_secondary,re_init=True)

        return optimized_args,x_image,y_image,np.array(mags)*np.max(mags)**-1,[self.chain.srcx,self.chain.srcy]

    def _pso(self, n_particles, n_iterations, lowerLimit=None, upperLimit=None, threadCount=1,re_init=False,mpi=False,
             print_key='default'):
        """
        returns the best fit for the lense model on catalogue basis with particle swarm optimizer
        """
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

        elif re_init is True:

            pso = self.PSO

        else:

            pso = ParticleSwarmOptimizer(self.chain, lowerLimit, upperLimit, n_particles, threads=threadCount)
            self.PSO = pso

        X2_list = []
        vel_list = []
        pos_list = []
        time_start = time.time()
        if pso.isMaster():
            print('Computing the %s ...' % print_key)
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

if False:
    from MagniPy.Analysis.PresetOperations.halo_constructor import Realization
    from MagniPy.LensBuild.defaults import *
    from MagniPy.Solver.solveroutines import SolveRoutines
    from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

    lens_params = {'R_ein':1.04,'x':0.001,'y':0.008,'ellip':0.3,'ellip_theta':0.01,'shear':0.07,'shear_theta':34,'gamma':2}
    start = Deflector(subclass=SIE(),redshift=0.5,**lens_params)
    real = Realization(0.5,1.5)
    halos = real.halo_constructor('NFW','plaw_main',{'fsub':0.00,'M_halo':10**13,'r_core':'0.5Rs','tidal_core':True},Nrealizations=1)

    solver = SolveRoutines(0.5,1.5)

    data = solver.solve_lens_equation(macromodel=start,realizations=halos,multiplane=False,srcx=0.03,srcy=0.01,ray_trace=True)

    lens_sys = solver.build_system(start,multiplane=False)
    redshift_list,lens_list,arg_list = lens_sys.lenstronomy_lists()

    lensModel = LensModelExtensions(lens_model_list=lens_list, multi_plane=False,
                                    redshift_list=redshift_list, z_source=1.5)

    optimizer = QuadSampler(zlist=redshift_list,lens_list=lens_list,arg_list=arg_list,z_source=1.5,x_pos=data[0].x,y_pos=data[0].y,
                            tol_source=0.0001,magnification_target=data[0].m,tol_mag=None,tol_centroid=0.02,centroid_0=[0,0])

    optimized_args,ximg,yimg,mags,src = optimizer.pso(350,200,run_mode='src_plane_chi2')

    for arg in optimized_args[0].keys():
        print arg,optimized_args[0][arg] - start.lenstronomy_args[arg]
    shear,shear_theta = cart_to_polar(optimized_args[1]['e1'],optimized_args[1]['e2'])
    print shear,shear-start.shear
    print shear_theta,shear_theta-start.shear_theta

    print ximg,data[0].x
    print yimg,data[0].y
    print mags,data[0].m