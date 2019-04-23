from MagniPy.magnipy import Magnipy
from MagniPy.Solver.RayTrace.raytrace import RayTrace
import copy
import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from MagniPy.LensBuild.defaults import *
from scipy.ndimage.filters import gaussian_filter

class Analysis(Magnipy):

    def hessian_with_scale(self, lens_system=None,main=None,halos=None,x_img=None,
                       y_img=None, convolve_scale = None, step = 0.003, window_size = 0.05):

        if lens_system is None:
            lens_system = self.build_system(main=main, realization=halos, multiplane=True)

        lenstronomy = self.lenstronomy_build()

        lensmodel, lensmodel_params = lenstronomy.get_lensmodel(lens_system)

        x = np.arange(x_img - 0.5*window_size, x_img + 0.5 * window_size, step)
        y = np.arange(y_img - 0.5*window_size, y_img + 0.5 * window_size, step)
        xx, yy = np.meshgrid(x, y)
        original_shape = xx.shape
        xx = xx.ravel()
        yy = yy.ravel()

        alpha_ra, alpha_dec = lensmodel.alpha(xx, yy, lensmodel_params)

        alpha_ra_dx, alpha_dec_dx = lensmodel.alpha(xx + convolve_scale, yy, lensmodel_params)
        alpha_ra_dy, alpha_dec_dy = lensmodel.alpha(xx, yy + convolve_scale, lensmodel_params)

        dalpha_rara = (alpha_ra_dx - alpha_ra) / convolve_scale
        dalpha_radec = (alpha_ra_dy - alpha_ra) / convolve_scale
        dalpha_decra = (alpha_dec_dx - alpha_dec) / convolve_scale
        dalpha_decdec = (alpha_dec_dy - alpha_dec) / convolve_scale

        fxx = dalpha_rara
        fyy = dalpha_decdec
        fxy = dalpha_radec
        fyx = dalpha_decra

        return fxx.reshape(original_shape), fxy.reshape(original_shape), \
               fyx.reshape(original_shape), fyy.reshape(original_shape)

    def shear_with_scale(self, lens_system=None,main=None,halos=None,x_img=None,
                       y_img=None, convolve_scale = None, step = 0.003, window_size = 0.05):

        fxx, fxy, fyx, fyy = self.hessian_with_scale(lens_system, main, halos, x_img, y_img,
                                                     convolve_scale, step, window_size)

        shear1 = 0.5*(fxx - fyy)
        shear2 = np.sqrt(0.25 * fxy * fyx)

        return shear1, shear2

    def sersicNFW_effective_slope(self,params):

        from MagniPy.MassModels.SersicNFW import SersicNFW
        s = SersicNFW(R0_fac=0.5)
        x = np.linspace(params['theta_E'] * 0.7, params['theta_E'] * 1.3, 200)
        kappa = s.kappa(x, 0, theta_E=params['theta_E'], Rs=params['Rs'], reff_thetaE_ratio=params['ratio'],
                        n_sersic=params['n_sersic'], q=1, separate=False,f=params['f'])
        # plt.plot(np.log10(x),np.log10((nfw_kappa+sersic_kappa)),color=cmap(np.absolute(1-mean*r**-1)))

        return np.polyfit(np.log10(kappa), np.log10(x), 1)[0]

    def get_hessian(self,lens_system=None,main=None,halos=None,x=None,y=None,multiplane=True):

        if lens_system is None:
            lens_system = self.build_system(main=main, realization=halos, multiplane=multiplane)

        lenstronomy = self.lenstronomy_build()

        lensmodel, lensmodel_params = lenstronomy.get_lensmodel(lens_system)

        if isinstance(x,int) or isinstance(x,float):

            f_xx, f_xy, f_yx, f_yy = lensmodel.hessian(x,y,kwargs=lensmodel_params)

            return f_xx, f_xy,f_yx,f_yy
        else:
            initial_shape = x.shape
            f_xx, f_xy, f_yx, f_yy = lensmodel.hessian(x.ravel(), y.ravel(), kwargs=lensmodel_params)

            return f_xx.reshape(initial_shape), f_xy.reshape(initial_shape), \
                   f_yx.reshape(initial_shape), f_yy.reshape(initial_shape)

    def get_shear(self,lens_system=None,main=None,halos=None,x=None,y=None,multiplane=None):

        f_xx, f_xy, f_yx, f_yy = self.get_hessian(lens_system,main,halos,x,y,multiplane)

        g1 = 0.5*(f_xx - f_yy)
        g2 = -0.5*f_xy
        g3 = -0.5*f_yx

        return g1,g2,g3

    def critical_cruves_caustics(self,lens_system=None,main=None,halos=None,multiplane=None,compute_window=1.5,
                                 scale=0.5,max_order=10,method=None,grid_scale=0.005):

        if lens_system is None:
            lens_system = self.build_system(main=main, realization=halos, multiplane=multiplane)

        lenstronomy = self.lenstronomy_build()

        lensmodel, lensmodel_params = lenstronomy.get_lensmodel(lens_system)

        extension = LensModelExtensions(lensmodel)

        if method=='tiling':
            xcrit, ycrit = extension.critical_curve_tiling(lensmodel_params,compute_window=compute_window,
                                                       start_scale=scale,max_order=max_order)
            return xcrit, ycrit
        else:
           
            ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = \
                extension.critical_curve_caustics(lensmodel_params,compute_window=5,grid_scale=grid_scale)

            return ra_crit_list, dec_crit_list,ra_caustic_list, dec_caustic_list

    def rayshoot(self,lens_system=None,main=None,halos=None,x=None,y=None,multiplane=None):

        if lens_system is None:
            lens_system = self.build_system(main=main, realization=halos, multiplane=multiplane)

        lenstronomy = self.lenstronomy_build()

        lensmodel, lensmodel_params = lenstronomy.get_lensmodel(lens_system)

        xnew,ynew = lensmodel.ray_shooting(x,y,kwargs=lensmodel_params)

        return xnew,ynew

    def get_deflections(self,lens_system=None,main=None,halos=None,x=None,y=None,multiplane=None):

        if lens_system is None:

            lens_system = self.build_system(main=main, realization=halos, multiplane=multiplane)

        lenstronomy = self.lenstronomy_build()

        lensmodel, lensmodel_params = lenstronomy.get_lensmodel(lens_system)

        if x.ndim >1:
            dx,dy = lensmodel.alpha(x.ravel(), y.ravel(),lensmodel_params)
            return dx.reshape(len(x), len(y)), dy.reshape(len(x), len(y))
        else:
            dx, dy = lensmodel.alpha(x, y, lensmodel_params)

            return dx,dy

    def get_kappa(self,lens_list=None,x=None,y=None,multiplane=False,method='lenstronomy',lens_system=None):

        if lens_system is None:
            if len(lens_list)==1:
                lens_system = self.build_system(lens_list[0],multiplane=multiplane)
            else:
                lens_system = self.build_system(lens_list[0], realization=lens_list[1:], multiplane=multiplane)

        lenstronomy_instance = self.lenstronomy_build()

        lensmodel,lensmodel_params= lenstronomy_instance.get_lensmodel(lens_system)

        return lensmodel.kappa(x.ravel(),y.ravel(),lensmodel_params).reshape(len(x),len(x))


    def get_magnification(self,lens_list=None,x=None,y=None,multiplane=False,method='lenstronomy',lens_system=None):

        if lens_system is None:
            if len(lens_list)==1:
                lens_system = self.build_system(lens_list[0],multiplane=multiplane)
            else:
                lens_system = self.build_system(lens_list[0], realization=lens_list[1:], multiplane=multiplane)

        lenstronomy_instance = self.lenstronomy_build()

        lensmodel, lensmodel_params = lenstronomy_instance.get_lensmodel(lens_system)

        return lensmodel.magnification(x.ravel(),y.ravel(),lensmodel_params).reshape(len(x),len(y))

    def azimuthal_kappa_avg(self,lens_list,rmax,N=1000,multiplane=False,pad=10):

        x,y = np.linspace(-rmax,rmax,N),np.linspace(-rmax,rmax,N)
        xx,yy = np.meshgrid(x,y)

        kappa = self.get_kappa(lens_list,xx,yy,multiplane=multiplane)

        r_bins = np.linspace(0,rmax,int(N*.5)+pad)
        r_step = r_bins[1]
        r_bins = r_bins[pad:]
        r_center = r_bins+r_step

        rr = np.sqrt(xx**2+yy**2)
        conv = []
        for r in r_bins:

            inds = np.where(np.absolute(rr-r)<r_step)

            mass = kappa[inds]
            mass_total = np.sum(mass)
            if mass_total==0:
                conv.append(0)
            else:
                conv.append(mass_total*len(mass)**-2)

        return r_center,conv

    def raytrace_images(self, full_system=None, macromodel=None, xcoord=None, ycoord=None, realizations=None,
                        multiplane=None,
                        identifier=None, srcx=None, srcy=None, res=None, method='lenstronomy',
                        source_shape='GAUSSIAN', source_size_kpc=None, **kwargs):

        lens_systems = []

        if source_size_kpc is None:
            raise Exception('specify source size')

        source_scale = self.cosmo.kpc_per_asec(self.zsrc)
        source_size = source_size_kpc * source_scale ** -1

        if res is None:
            res = default_res(source_size)

        if full_system is None:

            assert macromodel is not None

            if realizations is not None:
                assert len(realizations) == 1
                lens_system = self.build_system(main=copy.deepcopy(macromodel), realization=realizations[0],
                                                multiplane=multiplane)

        else:

            lens_system=copy.deepcopy(full_system)

        lenstronomy = self.lenstronomy_build()

        lensmodel, kwargs_lens = lenstronomy.get_lensmodel(lens_system)

        trace = RayTrace(xsrc=srcx, ysrc=srcy, multiplane=multiplane, method=method, res=res,
                         source_shape=source_shape,raytrace_with=method,
                         cosmology=self.cosmo, source_size=source_size,polar_grid=False,**kwargs)

        magnifications, image = trace.get_images(xcoord, ycoord, lensmodel, kwargs_lens,
                                                 return_image=True)

        return magnifications, image








