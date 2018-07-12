__author__ = 'dgilman'

import time
import numpy as np
from cosmoHammer import ParticleSwarmOptimizer
from Params import Params
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.lens_model import LensModel
from single_plane import SinglePlaneOptimizer,SinglePlaneOptimizer_SHEAR
from scipy.optimize import minimize
from scipy.linalg import lstsq
from multi_plane import MultiPlaneOptimizer

class Optimizer(object):
    """
    class which executes the different sampling  methods
    """

    def __init__(self, zlist=[], lens_list=[], arg_list=[], z_source=None, x_pos=None, y_pos=None, tol_source=1e-16,magnification_target=None,
                 tol_mag=0.1, tol_centroid=None, centroid_0=None, initialized=False, astropy_instance=None, optimizer_routine=str,
                 z_main=None, multiplane=None,lenstronomy_wrap = None, interp_range = 3, interp_resolution = 4e-2):
        """
        initialise the classes for parameter options and solvers
        """
        assert len(x_pos) == 4
        assert len(y_pos) == 4
        assert len(magnification_target) == len(x_pos)
        assert optimizer_routine in ['optimize_SIE_shear','optimize_plaw_shear','optimize_SHEAR']
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

            if optimizer_routine == 'optimize_SHEAR':
                self.optimizer = SinglePlaneOptimizer_SHEAR(lensModel, x_pos, y_pos, tol_source, self.Params, \
                                                  magnification_target, tol_mag, centroid_0, tol_centroid, arg_list=arg_list)
            else:
                self.optimizer = SinglePlaneOptimizer(lensModel, x_pos, y_pos, tol_source, self.Params, \
                                                  magnification_target, tol_mag, centroid_0, tol_centroid,
                                                  k_start=self.Params.vary_inds[1], arg_list=arg_list)
                self.optimizer_2 = SinglePlaneOptimizer(lensModel, x_pos, y_pos, tol_source, self.Params, \
                                                  magnification_target, tol_mag, centroid_0, tol_centroid,
                                                  k_start=self.Params.vary_inds[1], arg_list=arg_list,return_sign=1)

        else:

            x_interp = y_interp = np.linspace(-interp_range,interp_range,2*interp_range*interp_resolution**-1)

            [lensmodel_main, main_args], [lensmodel_front, front_args], \
            [back_interpolated, back_args_interpolated],comoving_distances,comoving_distances_ij,reduced_to_phys_factors = lenstronomy_wrap.composite_multiplane_model(x_interp,y_interp,
                                                   lensModel,arg_list,z_main,z_source)

            self.optimizer = MultiPlaneOptimizer(lensModel,arg_list,lensmodel_main,main_args,lensmodel_front,front_args,back_interpolated,
                 back_args_interpolated,x_pos, y_pos, tol_source, self.Params, magnification_target,
                 tol_mag, centroid_0, tol_centroid, z_main, z_source,comoving_distances,comoving_distances_ij,reduced_to_phys_factors)


    def optimize(self,n_particles=None,n_iterations=None,method='PS'):

        optimized_args = self._pso(n_particles, n_iterations)

        if method == 'optimize':

            opt = minimize(self.optimizer_2,x0=optimized_args,tol=1e-20)

            optimized_args = opt['x']

        optimized_args = self.optimizer.Params.argstovary_todictionary(optimized_args)

        optimized_args += self.optimizer.Params.argsfixed_todictionary()

        ximg, yimg = self.optimizer._get_images()

        return optimized_args, [self.optimizer.srcx, self.optimizer.srcy], [ximg, yimg]

    def _pso(self, n_particles, n_iterations, lowerLimit=None, upperLimit=None, threadCount=1,social_influence = 0.9,
             personal_influence=1.3):

        """
        returns the best fit for the lense model on catalogue basis with particle swarm optimizer
        """

        if lowerLimit is None or upperLimit is None:
            lowerLimit, upperLimit = self.lower_limit, self.upper_limit

        else:
            lowerLimit = np.maximum(lowerLimit, self.lower_limit)
            upperLimit = np.minimum(upperLimit, self.upper_limit)

        pso = ParticleSwarmOptimizer(self.optimizer, lowerLimit, upperLimit, n_particles, threads=threadCount)

        swarms, gBests = pso.optimize(maxIter=n_iterations,c1=social_influence,c2=personal_influence)

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
