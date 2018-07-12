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

    def interpolate_lensmodel(self,x_values,y_values,lensModel,kwargs_lens,multi_plane,z_model=None,z_source=None):

        xx,yy = np.meshgrid(x_values,y_values)
        L = int(np.shape(xx)[0])
        xx,yy = xx.ravel(),yy.ravel()

        if len(lensModel.lens_model_list)>0 and multi_plane is True:

            lensmodel_interpolated = LensModel(['INTERPOL'],multi_plane=multi_plane,z_source=z_source,cosmo=self.astropy_instance,
                                               redshift_list=[z_model])

            f_x, f_y = lensModel.alpha(xx, yy, kwargs_lens)
            f_xx, f_yy, f_xy, f_yx = lensModel.hessian(xx, yy, kwargs_lens)
            args = [{'f_x': f_x.reshape(L,L), 'f_y': f_y.reshape(L,L), 'f_xx': f_xx.reshape(L,L), 'f_yy': f_yy.reshape(L,L),
                     'f_xy': f_xy.reshape(L,L), 'f_yx': f_yx.reshape(L,L),
                     'grid_interp_x':x_values,'grid_interp_y':y_values}]

        elif len(lensModel.lens_model_list)>0 and multi_plane is False:

            raise Exception('single plane interpolation not implmeneted.')

        else:

            raise Exception('length of lens_model_list must be > 0.')

        return lensmodel_interpolated,args

    def multi_plane_partition(self,lensModel_full,kwargs_lens_full,z_main,z_source,macro_inds=[0,1]):
        """
        Splits a full LOS lens model into
        1) front-of-macromodel+main lens plane subhalos
        2) macromodel
        3) behind lens halos
        :param lensModel:
        :param kwargs_lens:
        :param z_start:
        :param z_end:
        :param exclude_k:
        :param keep_k:
        :return:
        """

        lens_list = lensModel_full.lens_model_list
        z_list = lensModel_full.redshift_list

        macromodel_kwargs,macromodel_lens_list,macromodel_redshift = [], [], []
        front_kwargs, front_lens_list, front_redshift = [], [], []
        back_kwargs, back_lens_list, back_redshift = [], [], []

        for index in range(0,len(kwargs_lens_full)):

            if index in macro_inds:
                macromodel_kwargs.append(kwargs_lens_full[index])
                macromodel_lens_list.append(lens_list[index])
                macromodel_redshift.append(z_main)
            else:
                if z_list[index]<=z_main:
                    front_kwargs.append(kwargs_lens_full[index])
                    front_lens_list.append(lens_list[index])
                    front_redshift.append(z_list[index])
                else:
                    back_kwargs.append(kwargs_lens_full[index])
                    back_lens_list.append(lens_list[index])
                    back_redshift.append(z_list[index])

        macromodel_lensmodel = LensModel(lens_model_list=macromodel_lens_list,redshift_list=macromodel_redshift,
                                         z_source=z_source,multi_plane=True,cosmo=self.astropy_instance)
        if len(front_lens_list) == 0:
            front_lensmodel = LensModel(lens_model_list=['NONE'], redshift_list=[z_main],
                                        z_source=z_source, multi_plane=True, cosmo=self.astropy_instance)
        else:
            front_lensmodel = LensModel(lens_model_list=front_lens_list, redshift_list=front_redshift,
                                         z_source=z_source, multi_plane=True, cosmo=self.astropy_instance)

        if len(back_lens_list) == 0:
            back_lensmodel = LensModel(lens_model_list=['NONE'],redshift_list=[z_main],z_source=z_source,
                                       cosmo=self.astropy_instance,multi_plane=True)
        else:
            back_lensmodel = LensModel(lens_model_list=back_lens_list, redshift_list=back_redshift,
                                   z_source=z_source,cosmo=self.astropy_instance,multi_plane=True)

        return [macromodel_lensmodel,macromodel_kwargs],[front_lensmodel,front_kwargs],[back_lensmodel,back_kwargs]

    def behind_main_plane(self,lensModel,z_main):

        assert lensModel.multi_plane is True

        for idx in lensModel.lens_model._sorted_redshift_index:

            if lensModel.redshift_list[idx]>z_main:
                return lensModel.redshift_list[idx]

        raise Exception('no halos behind main lens plane')

    def composite_multiplane_model(self,x_interp,y_interp,lensModel_full,kwargs_lens_full,z_main,z_source):

        comvoing_distances_ij,comvoing_distances, reduced_to_phys_factors = [],[], []

        T_zlens = lensModel_full.lens_model._cosmo_bkg.T_xy(0, z_main)
        comvoing_distances.append(T_zlens)
        comvoing_distances_ij.append(T_zlens)
        reduced_to_phys_factors.append(lensModel_full.lens_model._cosmo_bkg.D_xy(0, z_source) / lensModel_full.lens_model._cosmo_bkg.D_xy(z_main, z_source))

        [macromodel_lensmodel, macromodel_kwargs], [front_lensmodel, front_kwargs],\
        [back_lensmodel,back_kwargs] = self.multi_plane_partition(lensModel_full, kwargs_lens_full, z_main, z_source)

        back_z = self.behind_main_plane(lensModel_full, z_main)

        T_back_z = lensModel_full.lens_model._cosmo_bkg.T_xy(0,back_z)
        T_main_back = lensModel_full.lens_model._cosmo_bkg.T_xy(z_main,back_z)
        comvoing_distances.append(T_back_z)
        comvoing_distances_ij.append(T_main_back)
        reduced_to_phys_factors.append(lensModel_full.lens_model._cosmo_bkg.D_xy(0,
                            z_source) / lensModel_full.lens_model._cosmo_bkg.D_xy(back_z, z_source))

        lensmodel_interpolated, lensmodel_interp_args = self.interpolate_lensmodel(x_interp, y_interp,
                                                                                      back_lensmodel, back_kwargs,
                                                                                      True, z_model=back_z,
                                                                                      z_source=z_source)
        T_back_source = lensModel_full.lens_model._cosmo_bkg.T_xy(back_z,z_source)
        T_source = lensModel_full.lens_model._cosmo_bkg.T_xy(0,z_source)
        comvoing_distances.append(T_source)
        comvoing_distances_ij.append(T_back_source)

        return [macromodel_lensmodel, macromodel_kwargs], [front_lensmodel, front_kwargs],\
        [lensmodel_interpolated,lensmodel_interp_args],comvoing_distances,comvoing_distances_ij,reduced_to_phys_factors