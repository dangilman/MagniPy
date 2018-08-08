from MagniPy.util import polar_to_cart
import numpy as np
from copy import deepcopy
from MagniPy.Solver.LenstronomyWrap.kwargs_translate import gravlens_to_lenstronomy
# import the lens equation solver class (finding image plane positions of a source position)
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.solver4point import Solver4Point
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Optimizer.optimizer import Optimizer

class LenstronomyWrap:

    def __init__(self,cosmo=None,z_source = None):

        self.xtol = 1e-10
        self.min_distance = 0.01
        self.search_window = 4
        self.precision_limit = 10**-10
        self.num_iter_max = 500
        self.astropy_instance = cosmo

        assert z_source is not None
        self.zsrc = z_source

    def assemble(self,system):

        zlist, lens_list, arg_list = system.lenstronomy_lists()

        return {'lens_model_list':lens_list,'kwargs_lens':arg_list,'redshift_list':zlist}

    def get_lensmodel(self,lens_system):

        lists = self.assemble(lens_system)

        return LensModelExtensions(lens_model_list=lists['lens_model_list'],redshift_list=lists['redshift_list'],z_source=self.zsrc,cosmo=self.astropy_instance ,
                         multi_plane=lens_system.multiplane),lists['kwargs_lens']

    def solve_leq(self,xsrc,ysrc,lensmodel,lens_model_params):

        lensEquationSolver = LensEquationSolver(lensModel=lensmodel)

        #x_image, y_image = lensEquationSolver.image_position_from_source(kwargs_lens=lens_model_params, sourcePos_x=xsrc,
        #                                                      sourcePos_y=ysrc,min_distance=self.min_distance, search_window=self.search_window,
        #                                                                 precision_limit=self.precision_limit, num_iter_max=self.num_iter_max,
        #                                                                 arrival_time_sort=False)
        #if len(x_image) != 4:
        x_image,y_image = lensEquationSolver.findBrightImage(xsrc,ysrc,lens_model_params,arrival_time_sort=False)

        return x_image,y_image

    def fit_lensmodel(self, x_image, y_image, lensModel, solver_type, lens_model_params):

        solver4Point = Solver4Point(lensModel=lensModel, solver_type=solver_type)

        kwargs_fit,acc = solver4Point.constraint_lensmodel(x_pos=x_image, y_pos=y_image, kwargs_list=lens_model_params,
                                                       xtol=self.xtol)

        return kwargs_fit

    def compute_mags(self,x_pos,y_pos,lensmodel,lens_args,source_size,grid_rmax,grid_number,source_shape):

        magnification = lensmodel.magnification_finite(x_pos,y_pos,kwargs_lens=lens_args,source_sigma=source_size,window_size=grid_rmax,grid_number=grid_number,
                                                       shape=source_shape,polar_grid=True)

        return np.absolute(magnification)

    def run_optimize(self,lens_system,z_source,x_pos,y_pos,tol_source,magnification_target,tol_mag,tol_centroid,centroid_0,
                     optimizer_routine,z_main,interpolate,n_particles,n_iterations,verbose,restart,re_optimize,particle_swarm):

        lensmodel_kwargs = self.assemble(lens_system)

        optimizer = Optimizer(x_pos,y_pos,magnification_target=magnification_target,redshift_list=lensmodel_kwargs['redshift_list'],
                              lens_model_list=lensmodel_kwargs['lens_model_list'],kwargs_lens=lensmodel_kwargs['kwargs_lens'],
                              optimizer_routine=optimizer_routine,multiplane=lens_system.multiplane,z_main=z_main,z_source=z_source,
                              tol_source=tol_source,tol_mag=tol_mag,tol_centroid=tol_centroid,centroid_0=centroid_0,astropy_instance=self.astropy_instance,
                              interpolate=interpolate,verbose=verbose,re_optimize=re_optimize,particle_swarm=particle_swarm,
                              pso_convergence_standardDEV=0.01, pso_convergence_mean=1, pso_compute_magnification=6)

        optimized_args, source, images = optimizer.optimize(n_particles,n_iterations,restart)

        return optimized_args, source, images, optimizer.optimizer.lensModel

