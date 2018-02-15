import numpy as np
from MagniPy.Solver.LenstronomyWrap.MultiLensWrap import MultiLensWrapper
from source_models import *
from MagniPy.util import *
from MagniPy.LensBuild.Cosmology.cosmology import Cosmo
import matplotlib.pyplot as plt
from lenstronomy.LensModel.Profiles.sie import SPEMD
from lenstronomy.LensModel.Profiles.nfw import NFW

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
        self.x_grid_0, self.y_grid_0 = np.meshgrid(np.linspace(-self.grid_rmax, self.grid_rmax, self.grid_rmax*res**-1),
                                                   -np.linspace(-self.grid_rmax, self.grid_rmax, self.grid_rmax*res**-1))

        self.raytrace_with = raytrace_with

        if self.polar_grid:

            self.r_indicies = np.where(self.grid_rmax > (self.x_grid_0 ** 2 + self.y_grid_0 ** 2) ** .5)

        if self.raytrace_with == 'lenstronomy':
            self.multilens_wrap = MultiLensWrapper(gridsize=grid_rmax, res=res, source_shape=source_shape,
                                                   astropy_class=self.cosmo.cosmo, z_source=self.cosmo.zsrc,
                                                   source_size=kwargs['source_size'])

    def get_images(self,xpos,ypos,lens_system,print_mag=False):

        assert self.multiplane is False

        images,magnifications = [],[]

        for i in range(0,len(xpos)):

            x_loc = xpos[i]*np.ones_like(self.x_grid_0)+self.x_grid_0
            y_loc = ypos[i]*np.ones_like(self.y_grid_0)+self.y_grid_0

            xdef = np.zeros_like(x_loc)
            ydef = np.zeros_like(y_loc)

            for count,deflector in enumerate(lens_system.lens_components):

                xplus,yplus = deflector.lensing.def_angle(x_loc,y_loc,**deflector.args)

                if deflector.has_shear:

                    shearx, sheary = deflector.Shear.def_angle(x_loc, y_loc,deflector.shear,deflector.shear_theta)

                    xplus += shearx
                    yplus += sheary

                xdef+=xplus
                ydef+=yplus

            x_source = x_loc - xdef
            y_source = y_loc - ydef

            source_light = self.source.source_profile(betax=x_source, betay=y_source)
            magnifications = np.sum(source_light)*self.res**2
            image = source_light * (int(np.shape(source_light)[0]) ** 2 * np.sum(source_light)) ** -1

        return magnifications,image

    def compute_mag(self,xpos,ypos,lens_system,print_mag=False):

        if self.multiplane is False:
            if self.raytrace_with == 'lenstronomy':
                return self.multilens_wrap.compute_mag(xpos,ypos,lens_system)

            else:
                return self._single_plane_trace(xpos,ypos,lens_system,print_mag)
        else:
            if self.raytrace_with == 'lenstronomy':
                return self.multilens_wrap.compute_mag(xpos, ypos, lens_system)
            else:
                return self._mult_plane_trace(xpos,ypos,lens_system)

    def _single_plane_trace(self,xpos,ypos,lens_system,print_mag=False):

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
            
            magnification.append(np.sum(self.source.source_profile(betax=x_source,betay=y_source))*self.res**2)

        return np.array(magnification)

    def _mult_plane_trace(self,xpos,ypos,lens_system):

        magnification = []

        for i in range(0,len(xpos)):
            if i==0:
                show=True
            else:
                show=False

            xcoords = self.x_grid_0 + xpos[i]
            ycoords = self.y_grid_0 + ypos[i]

            if self.polar_grid:
                xcoords = xcoords[self.r_indicies]
                ycoords = ycoords[self.r_indicies]

            x_source, y_source = self._multi_plane_shoot(lens_system,xcoords,ycoords,show=show)

            magnification.append(np.sum(self.source.source_profile(betax=x_source, betay=y_source)) * self.res ** 2)

        return np.array(magnification)

    def _comoving2angle(self,x,y,z):

        dt = self.cosmo.T_xy(0,z)
        return x*dt**-1,y*dt**-1

    def _reduced2phys(self,reduced_angle,z,zsrc):

        return reduced_angle*self.cosmo.D_xy(0,zsrc)*\
               self.cosmo.D_xy(z,zsrc)**-1

    def add_deflection(self, x, y, defx, defy, lens, z, is_shear=False,show=False):

        xangle,yangle = self._comoving2angle(x,y,z)

        dx_angle, dy_angle = lens.lensing.def_angle(xangle, yangle, **lens.args)

        if lens.has_shear:

            dx_angle_shear,dy_angle_shear = lens.Shear.def_angle(xangle,yangle,lens.shear,lens.shear_theta)
            dx_angle += dx_angle_shear
            dy_angle += dy_angle_shear

        dx_phys = self._reduced2phys(dx_angle,z,self.cosmo.zsrc)
        dy_phys = self._reduced2phys(dy_angle, z, self.cosmo.zsrc)

        return defx - dx_phys, defy - dy_phys

    def _ray_step(self,x,y,alpha_x,alpha_y,DT):

        return x+alpha_x*DT,y+alpha_y*DT

    def _multi_plane_shoot(self, lens_system, theta_x, theta_y, show):

        sorted_indexes = np.argsort(lens_system.redshift_list)

        zstart = 0

        alpha_x,alpha_y = theta_x, theta_y

        x,y = np.zeros_like(alpha_x),np.zeros_like(alpha_y)

        for i,index in enumerate(sorted_indexes):

            z = lens_system.redshift_list[index]

            lens = lens_system.lens_components[index]

            DT = self.cosmo.T_xy(zstart, z)

            x, y = self._ray_step(x,y,alpha_x,alpha_y,DT)

            alpha_x,alpha_y = self.add_deflection(x,y,alpha_x,
                                                            alpha_y,lens,z)

            zstart = z

        DT = self.cosmo.T_xy(zstart, self.cosmo.zsrc)

        x, y = x + alpha_x * DT, y + alpha_y * DT

        betax,betay = self._comoving2angle(x,y,self.cosmo.zsrc)

        return betax,betay










