import numpy as np
from source_models import *
from MagniPy.util import *
from MagniPy.LensBuild.Cosmology.cosmology import Cosmo
import matplotlib.pyplot as plt
from lenstronomy.LensModel.Profiles.sie import SPEMD
from lenstronomy.LensModel.Profiles.nfw import NFW
from MagniPy.Solver.LenstronomyWrap import generate_input,MultiLensWrap

class RayTrace:

    def __init__(self, xsrc=float, ysrc=float, multiplane=False, grid_rmax=int, res=0.0005, source_shape='', cosmology=None, raytrace_with=None,
                 polar_grid=False, **kwargs):

        """
        :param xsrc: x coordinate for grid center (arcseconds)
        :param ysrc: ""
        :param multiplane: multiple lens plane flag
        :param size: width of the box in asec
        :param res: pixel resolution asec per pixel
        """
        self.raytrace_with = raytrace_with

        self.polar_grid = polar_grid
        self.grid_rmax = grid_rmax
        self.res = res
        self.xsrc,self.ysrc = xsrc,ysrc
        self.multiplane = multiplane

        if source_shape == 'GAUSSIAN':
            self.source = GAUSSIAN(x=xsrc,y=ysrc,width=kwargs['source_size'])
        else:
            raise ValueError('other source models not yet implemented')

        self.cosmo = cosmology
        self.x_grid_0, self.y_grid_0 = np.meshgrid(np.linspace(-self.grid_rmax, self.grid_rmax, 2*self.grid_rmax*res**-1),
                                                   np.linspace(-self.grid_rmax, self.grid_rmax, 2*self.grid_rmax*res**-1))



        if self.raytrace_with == 'lenstronomy':
            self.multlenswrap = MultiLensWrap.MultiLensWrapper(multiplane=self.multiplane,astropy_class=self.cosmo.cosmo,
                                                               z_source=self.cosmo.zsrc,source_shape=source_shape,
                                                               gridsize=2*self.grid_rmax,res=self.res,
                                                               source_size=kwargs['source_size'])


    def get_images(self,xpos,ypos,lens_system,**kwargs):

        return self.compute_mag(xpos,ypos,lens_system,**kwargs)

    def compute_mag(self,xpos,ypos,lens_system,print_mag=False,**kwargs):

        if self.multiplane is False:
            if self.raytrace_with == 'lenstronomy':
                return self.multlenswrap.compute_mag(xpos,ypos,lens_system,)

            else:
                return self._single_plane_trace(xpos,ypos,lens_system,print_mag,**kwargs)
        else:
            if self.raytrace_with == 'lenstronomy':
                return self.multlenswrap.compute_mag(xpos, ypos, lens_system)
            else:
                return self._mult_plane_trace(xpos,ypos,lens_system,**kwargs)

    def _single_plane_trace(self,xpos,ypos,lens_system,print_mag=False,return_image=False,which_image=None):

        magnification = []

        if print_mag:
            print 'computing mag...'

        for i in range(0,len(xpos)):

            x_loc = xpos[i]*np.ones_like(self.x_grid_0)+self.x_grid_0
            y_loc = ypos[i]*np.ones_like(self.y_grid_0)+self.y_grid_0

            if self.polar_grid:

                x_loc = x_loc[self.r_indicies]
                y_loc = y_loc[self.r_indicies]

            xdef = np.zeros_like(x_loc)
            ydef = np.zeros_like(y_loc)

            for count,deflector in enumerate(lens_system.lens_components):

                xplus,yplus = deflector.lensing.def_angle(x=x_loc,y=y_loc,**deflector.args)


                if deflector.has_shear:

                    shearx, sheary = deflector.Shear.def_angle(x_loc, y_loc,deflector.shear,deflector.shear_theta)

                    xplus += shearx
                    yplus += sheary

                xdef += xplus
                ydef += yplus

                if print_mag and i==0:
                    print count+1

            x_source = x_loc - xdef
            y_source = y_loc - ydef

            if return_image:

                image = self._eval_src(x_source,y_source)

            magnification.append(np.sum(self._eval_src(x_source,y_source)*self.res**2))

        if return_image:
            return np.array(magnification),image
        else:
            return np.array(magnification)

    def _eval_src(self,betax,betay):

        #print 'lensmodel'

        return self.source.source_profile(betax=betax,betay=betay)

    def _mult_plane_trace(self,xpos,ypos,lens_system):

        magnification = []

        for i in range(0,len(xpos)):

            xcoords = self.x_grid_0 + xpos[i]
            ycoords = self.y_grid_0 + ypos[i]

            x_source, y_source = self._multi_plane_shoot(lens_system,xcoords,ycoords)

            magnification.append(np.sum(self._eval_src(x_source, y_source) * self.res ** 2))

        return np.array(magnification)

    def _index_ordering(self, redshift_list):
        """

        :param redshift_list: list of redshifts
        :return: indexes in acending order to be evaluated (from z=0 to z=z_source)
        """
        redshift_list = np.array(redshift_list)
        sort_index = np.argsort(redshift_list[redshift_list < self.cosmo.zsrc])
        if len(sort_index) < 1:
            raise ValueError("There is no lens object between observer at z=0 and source at z=%s" % self._z_source)
        return sort_index

    def _ray_step(self,x,y,alpha_x,alpha_y,DT):

        return x+alpha_x*DT,y+alpha_y*DT

    def _comoving2angle(self,d,z,zstart=0):

        return d*self.cosmo.T_xy(zstart,z)**-1

    def _reduced2phys(self,reduced,z):

        scale = self.cosmo.D_s*self.cosmo.D_A(z,self.cosmo.zsrc)**-1

        return reduced*scale

    def _next_plane(self,x,y,xdef,ydef,d):

        return x + xdef*d, y +ydef*d

    def _multi_plane_shoot(self, lens_system, x_obs, y_obs):

        sorted_indexes = self._index_ordering(lens_system.redshift_list)

        x,y = np.zeros_like(x_obs),np.zeros_like(y_obs)

        xdef_angle,ydef_angle = x_obs,y_obs

        zstart = 0

        for i,deflector in enumerate(lens_system.lens_components):

            z = lens_system.redshift_list[sorted_indexes[i]]

            D_to_plane = self.cosmo.T_xy(zstart,z)

            x,y = self._next_plane(x,y,xdef_angle,ydef_angle,D_to_plane)

            x_angle_plane = self._comoving2angle(x,z)
            y_angle_plane = self._comoving2angle(y,z)

            xdef_new,ydef_new = deflector.lensing.def_angle(x_angle_plane,y_angle_plane,**deflector.args)

            if deflector.has_shear:

                xshear,yshear = deflector.Shear.def_angle(x_angle_plane,y_angle_plane,deflector.shear,deflector.shear_theta)

                xdef_new += xshear
                ydef_new += yshear

            xdef_physical = self._reduced2phys(xdef_new, z)
            ydef_physical = self._reduced2phys(ydef_new, z)

            xdef_angle -= xdef_physical
            ydef_angle -= ydef_physical

            zstart = z

        D_to_src = self.cosmo.T_xy(zstart,self.cosmo.zsrc)

        x_on_src,y_on_src = x + D_to_src*xdef_angle, y + D_to_src*ydef_angle

        betax = self._comoving2angle(x_on_src,self.cosmo.zsrc)
        betay = self._comoving2angle(y_on_src, self.cosmo.zsrc)

        return betax,betay


















