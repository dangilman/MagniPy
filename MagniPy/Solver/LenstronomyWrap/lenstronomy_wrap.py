from MagniPy.util import polar_to_cart
import numpy as np
from copy import deepcopy
from kwargs_translate import gravlens_to_lenstronomy
# import the lens equation solver class (finding image plane positions of a source position)
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.solver4point import Solver4Point
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
import lenstronomy.Util.param_util as param_util

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

        self.lens_model_list,self.lens_model_params = [],[]
        self.current_lensmodel = None

    def update_settings(self,xtol=None,min_distance=None,search_window = None,precision_limit=None,num_iter_max = None):

        if xtol is not None:
            self.xtol = xtol
        if min_distance is not None:
            self.min_distance = min_distance
        if search_window is not None:
            self.search_window = search_window
        if precision_limit is not None:
            self.precision_limit = precision_limit
        if num_iter_max is not None:
            self.num_iter_max = num_iter_max

    def assemble(self,system):

        zlist, lens_list, arg_list = system.lenstronomy_lists()

        return {'lens_model_list':lens_list,'kwargs_lens':arg_list,'redshift_list':zlist}

    def get_lensmodel(self,lens_system):

        lists = self.assemble(lens_system)

        return LensModelExtensions(lens_model_list=lists['lens_model_list'],redshift_list=lists['redshift_list'],z_source=self.zsrc,cosmo=self.astropy_instance ,
                         multi_plane=lens_system.multiplane),lists['kwargs_lens']

    def solve_leq(self,xsrc,ysrc,lensmodel,lens_model_params):


        lensEquationSolver = LensEquationSolver(lensModel=lensmodel)

        x_image, y_image = lensEquationSolver.image_position_from_source(kwargs_lens=lens_model_params, sourcePos_x=xsrc,
                                                              sourcePos_y=ysrc,min_distance=self.min_distance, search_window=self.search_window,
                                                                         precision_limit=self.precision_limit, num_iter_max=self.num_iter_max)

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

    def interpolate_lensmodel(self,lens_model_list,x_values,y_values,redshift_list=None,kwargs_lens=None,multiplane=False,
                              z_source=None):

        xx,yy = np.meshgrid(x_values,y_values)
        xx,yy = xx.ravel(),yy.ravel()

        if len(lens_model_list)>0:
            if multiplane is False:

                lenmodel_init = LensModel(lens_model_list)

                lensmodel_interpolated = LensModel(['INTERPOL'])

                f_x, f_y = lenmodel_init.alpha(xx, yy, kwargs_lens)
                deflection_angles = [{'f_x': f_x, 'f_y': f_y}]

            else:

                lenmodel_init = LensModel(lens_model_list,redshift_list=redshift_list,z_source=z_source,
                                          cosmo=self.astropy_instance,multi_plane=True)

                lensmodel_interpolated = LensModel(['INTERPOL'])

                f_x, f_y = lenmodel_init.alpha(xx, yy, kwargs_lens)
                deflection_angles = [{'f_x': f_x, 'f_y': f_y}]

        else:

            lensmodel_interpolated = None

            deflection_angles = None

        return deflection_angles,lensmodel_interpolated

    def interpolate_LOS(self,x_values,y_values,lensModel,kwargs_lens,z_start,z_end,exclude_k=None):

        lists = self._multi_plane_partition(lensModel,kwargs_lens,z_start,z_end,exclude_k)

        lensing,lensModel_interpolated = self.interpolate_lensmodel(lists['lens_model_list'],x_values,y_values,
                                        redshift_list=lists['redshift_list'],kwargs_lens=lists['kwargs_lens'],
                                          multiplane=True,z_source=self.zsrc)

        return lensing,lensModel_interpolated

    def split_lensmodel(self,lensModel,kwargs_lens,z_start,z_end,exclude_k=None,keep_k=None,multiplane=None):

        lists = self._multi_plane_partition(lensModel,kwargs_lens,z_start,z_end,exclude_k,keep_k)

        return LensModelExtensions(lists['lens_model_list'],z_source=self.zsrc,redshift_list=lists['redshift_list'],
                                   cosmo=self.astropy_instance,multi_plane=multiplane),lists['kwargs_lens']

    def _multi_plane_partition(self,lensModel,kwargs_lens,z_start,z_end,exclude_k=None,keep_k=None):

        zlist = lensModel.redshift_list
        lens_list = lensModel.lens_model_list

        zlist = np.array(zlist)

        redshifts, lenses, lens_kwargs = [], [], []

        if exclude_k is None and keep_k is None:

            for i in range(0, len(lens_list)):
                if zlist[i] > z_start and zlist[i] <= z_end:
                    redshifts.append(zlist[i])
                    lenses.append(lens_list[i])
                    lens_kwargs.append(kwargs_lens[i])

        else:

            for i in range(0, len(lens_list)):

                if keep_k is not None and i in keep_k:

                    redshifts.append(zlist[i])
                    lenses.append(lens_list[i])
                    lens_kwargs.append(kwargs_lens[i])

                elif zlist[i] > z_start and zlist[i] <= z_end:

                    if i not in exclude_k:
                        redshifts.append(zlist[i])
                        lenses.append(lens_list[i])
                        lens_kwargs.append(kwargs_lens[i])

        return {'redshift_list': redshifts, 'lens_model_list': lenses, 'kwargs_lens': lens_kwargs}



