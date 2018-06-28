__author__ = 'dgilman'

import time
import numpy as np
from cosmoHammer import ParticleSwarmOptimizer
from Params import Params
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.lens_model import LensModel
from single_plane import SinglePlaneOptimizer
from scipy.optimize import minimize
from multi_plane import MultiPlaneOptimizer

class Optimizer(object):
    """
    class which executes the different sampling  methods
    """

    def __init__(self, zlist=[], lens_list=[], arg_list=[], z_source=None, x_pos=None, y_pos=None, tol_source=1e-16,magnification_target=None,
                 tol_mag=0.1, tol_centroid=None, centroid_0=None, initialized=False, astropy_instance=None, optimizer_routine=str,
                 z_main=None, multiplane=None,lenstronomy_wrap = None, interp_range = 3, interp_resolution = 1e-3):
        """
        initialise the classes for parameter options and solvers
        """
        assert len(x_pos) == 4
        assert len(y_pos) == 4
        assert len(magnification_target) == len(x_pos)
        assert optimizer_routine in ['optimize_SIE_shear','optimize_plaw_shear']
        assert tol_source is not None
        assert len(zlist) == len(lens_list) == len(arg_list)

        if multiplane is True:
            assert z_source is not None
            assert z_main is not None
            assert astropy_instance is not None

        lensModel = LensModelExtensions(lens_model_list=lens_list,redshift_list=zlist,z_source=z_source,
                                        cosmo=astropy_instance,multi_plane=multiplane)

        self.Params = Params(zlist=lensModel.redshift_list, lens_list=lensModel.lens_model_list, arg_list=arg_list,
                             optimizer_routine=optimizer_routine, initialized=initialized)

        self.lower_limit = self.Params.tovary_lower_limit
        self.upper_limit = self.Params.tovary_upper_limit

        if multiplane is False:

            self.optimizer = SinglePlaneOptimizer(lensModel, x_pos, y_pos, tol_source, self.Params, \
                                                  magnification_target, tol_mag, centroid_0, tol_centroid,
                                                  k_start=self.Params.Nprofiles_to_vary, arg_list=arg_list)

        else:
            assert z_main is not None
            assert z_source > z_main

            exclude_k = np.arange(0,self.Params.Nprofiles_to_vary)

            lensmodel_main,main_args = lenstronomy_wrap.split_lensmodel(lensModel,arg_list,z_main,z_main,keep_k = exclude_k, multiplane=True)

            lensmodel_front,front_args = lenstronomy_wrap.split_lensmodel(lensModel,arg_list,0,z_main,exclude_k=exclude_k,
                                                                                           multiplane=True)

            x_values = y_values = np.linspace(-interp_range, interp_range,
                                              int(2 * interp_range * interp_resolution ** -1))

            print 'computing background deflections... '

            lensing_components_back, lensModel_interpolated_back = lenstronomy_wrap.interpolate_LOS(x_values, y_values,
                                                                                           lensModel,arg_list,z_main, z_source,
                                                                                           exclude_k=exclude_k)
            print 'done.'

            self.optimizer = MultiPlaneOptimizer(lensModel,arg_list,lensmodel_main,main_args,lensmodel_front,front_args,lensing_components_back,
                                                 lensModel_interpolated_back,x_pos, y_pos, tol_source, self.Params,
                                                 magnification_target, tol_mag,centroid_0, tol_centroid, z_main, z_source)


    def optimize(self,n_particles=None,n_iterations=None,method='PS'):

        optimized_args = self._pso(n_particles, n_iterations)

        if method == 'optimize':
            print 'optimizing... '
            opt = minimize(self.optimizer,x0=optimized_args,args=(1),tol=1e-25)

            optimized_args = opt['x']

        optimized_args = self.optimizer.Params.argstovary_todictionary(optimized_args)

        optimized_args += self.optimizer.Params.argsfixed_todictionary()

        ximg, yimg = self.optimizer._get_images()

        return optimized_args, [self.optimizer.srcx, self.optimizer.srcy], [ximg, yimg]

    def _pso(self, n_particles, n_iterations, lowerLimit=None, upperLimit=None, threadCount=1):

        """
        returns the best fit for the lense model on catalogue basis with particle swarm optimizer
        """

        if lowerLimit is None or upperLimit is None:
            lowerLimit, upperLimit = self.lower_limit, self.upper_limit

        else:
            lowerLimit = np.maximum(lowerLimit, self.lower_limit)
            upperLimit = np.minimum(upperLimit, self.upper_limit)

        pso = ParticleSwarmOptimizer(self.optimizer, lowerLimit, upperLimit, n_particles, threads=threadCount)

        swarms, gBests = pso.optimize(maxIter=n_iterations)

        likelihoods = [particle.fitness for particle in gBests]
        ind = np.argmax(likelihoods)

        return gBests[ind].position
        #likelihoods = np.array(likelihoods)
        #print np.argmax(likelihoods)
        #ind = np.argmax(likelihoods)
        #print gBests[ind].position

        #exit(1)
        #for swarm in pso.sample(n_iterations):
        #    X2_list.append(pso.gbest.fitness * 2)
        #    vel_list.append(pso.gbest.velocity)
        #    pos_list.append(pso.gbest.position)
        #    num_iter += 1
        #    if num_iter % 50 == 0:
        #        print(num_iter)

        #result = pso.gbest.position

        #return self.optimizer.lens_args_latest
