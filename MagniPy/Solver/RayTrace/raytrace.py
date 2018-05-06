import numpy as np
from source_models import *
from MagniPy.util import *
import matplotlib.pyplot as plt
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

        self.xsrc,self.ysrc = xsrc,ysrc

        self.cosmo = cosmology

        self.x_grid_0, self.y_grid_0 = np.meshgrid(np.linspace(-self.grid_rmax, self.grid_rmax, 2*self.grid_rmax*res**-1),
                                                   np.linspace(-self.grid_rmax, self.grid_rmax, 2*self.grid_rmax*res**-1))

        polar_grid = True
        self.x_grid_0 = self.x_grid_0.ravel()
        self.y_grid_0 = self.y_grid_0.ravel()

        if polar_grid:

            inds = np.where(np.sqrt(self.x_grid_0**2+self.y_grid_0**2)<=self.grid_rmax)
            self.x_grid_0 = self.x_grid_0[inds]
            self.y_grid_0 = self.y_grid_0[inds]

        if source_shape == 'GAUSSIAN':
            self.source = GAUSSIAN(x=xsrc,y=ysrc,width=kwargs['source_size'],xgrid0=self.x_grid_0,ygrid0=self.y_grid_0)
        else:
            raise ValueError('other source models not yet implemented')

        if self.raytrace_with == 'lenstronomy':

            self.multilenswrap = MultiLensWrap.MultiLensWrapper()


    def get_images(self,xpos,ypos,lens_system,**kwargs):

        if self.raytrace_with == 'lensmodel':
            return self.compute_mag(xpos,ypos,lens_system,**kwargs)
        else:
            return self.multilenswrap.rayshoot(xpos,ypos,lens_system,source_function=self.source)

    def compute_mag(self,xpos,ypos,lensmodel=None,lens_model_params=None,lens_system=None,
                    print_mag=False,**kwargs):

        if self.raytrace_with == 'lenstronomy':
            return self.multilenswrap.magnification(xpos,ypos,self.x_grid_0,self.y_grid_0,lensmodel,lens_model_params,
                                                        self.source,self.res)

        else:
            if self.multiplane:
                return self._multi_plane_trace(xpos, ypos, lens_system, **kwargs)
            else:
                return self._single_plane_trace(xpos,ypos,lens_system,**kwargs)

    def _single_plane_trace_full(self,xx,yy,lens_system,to_img_plane=False,print_mag=False,return_image=False):

        if print_mag:
            print 'computing mag...'

        x_loc = xx
        y_loc = yy

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

        if to_img_plane:

            return xdef,ydef

        else:
            x_source = x_loc - xdef
            y_source = y_loc - ydef
            return x_source,y_source

    def _single_plane_trace(self,xpos,ypos,lens_system,print_mag=False,return_image=False):

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

        #plt.imshow(self.source.source_profile(betax=betax,betay=betay))
        #plt.show()

        return self.source(betax=betax,betay=betay)

    def multi_plane_trace_full(self,xx,yy,lens_system):

        betax,betay = self._multi_plane_shoot(lens_system,xx,yy)

        return betax,betay

    def _multi_plane_trace(self,xpos,ypos,lens_system,return_image=False):

        magnification = []

        for i in range(0,len(xpos)):

            xcoords = self.x_grid_0 + xpos[i]
            ycoords = self.y_grid_0 + ypos[i]

            x_source, y_source = self._multi_plane_shoot(lens_system,xcoords,ycoords)

            magnification.append(np.sum(self._eval_src(x_source, y_source) * self.res ** 2))

            if return_image:

                image = self._eval_src(x_source,y_source)

        if return_image:
            return np.array(magnification),image
        else:
            return np.array(magnification)

    def sort_redshift_indexes(self, redshift_list):
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

    def _comoving2angle(self,d,z):

        return d*self.cosmo.T_xy(0,z)**-1

    def _reduced2phys(self,reduced,z,z_source):

        factor = self.cosmo.D_xy(0, z_source) / self.cosmo.D_xy(z, z_source)
        return reduced * factor

    def _next_plane(self,x,y,xdef,ydef,d):

        return x + xdef*d, y +ydef*d

    def _add_deflection(self,x_comoving,y_comoving,deflector,xdeflection,ydeflection,z):

        x_angle = self._comoving2angle(x_comoving,z)
        y_angle = self._comoving2angle(y_comoving,z)

        xdef_new,ydef_new = deflector.lensing.def_angle(x_angle,y_angle,**deflector.args)

        if deflector.has_shear:
            xshear,yshear = deflector.Shear.def_angle(x_angle,y_angle,deflector.shear,deflector.shear_theta)
            xdef_new += xshear
            ydef_new += yshear

        xdef_physical = self._reduced2phys(xdef_new, z, self.cosmo.zsrc)
        ydef_physical = self._reduced2phys(ydef_new, z, self.cosmo.zsrc)

        alpha_x_new = xdeflection - xdef_physical
        alpha_y_new = ydeflection - ydef_physical

        return alpha_x_new, alpha_y_new

    def _add_deflection_write(self,x_comoving,y_comoving,deflector,xdeflection,ydeflection,z,file):

        x_angle = self._comoving2angle(x_comoving,z)
        y_angle = self._comoving2angle(y_comoving,z)

        xdef_new,ydef_new = deflector.lensing.def_angle(x_angle,y_angle,**deflector.args)

        with open(file,'a') as f:
            f.write('reduced_def '+str(deflector.profname)+': '+str(xdef_new)+' '+str(ydef_new)+'\n\n')

        if deflector.has_shear:
            xshear,yshear = deflector.Shear.def_angle(x_angle,y_angle,deflector.shear,deflector.shear_theta)
            xdef_new += xshear
            ydef_new += yshear
            with open(file, 'a') as f:
                f.write('reduced_def (shear) ' + str('Shear') + ': ' + str(xshear) + ' ' + str(yshear) + '\n\n')

        xdef_physical = self._reduced2phys(xdef_new,z)
        ydef_physical = self._reduced2phys(ydef_new,z)

        with open(file, 'a') as f:
            f.write('physical_def: ' + str(xdef_physical) + ' ' + str(ydef_physical) + '\n\n')

        alpha_x_new = xdeflection - xdef_physical
        alpha_y_new = ydeflection - ydef_physical

        return alpha_x_new, alpha_y_new

    def _multi_plane_shoot(self, lens_system, x_obs, y_obs):

        sorted_indexes = self.sort_redshift_indexes(lens_system.redshift_list)

        x,y = np.zeros_like(x_obs),np.zeros_like(y_obs)

        xdef_angle,ydef_angle = x_obs,y_obs

        zstart = 0

        for index in sorted_indexes:

            z = lens_system.redshift_list[index]

            deflector = lens_system.lens_components[index]

            D_to_plane = self.cosmo.T_xy(zstart,z)

            x,y = self._next_plane(x,y,xdef_angle,ydef_angle,D_to_plane)

            xdef_angle,ydef_angle = self._add_deflection(x,y,deflector,xdef_angle,ydef_angle,z)

            zstart = z

        D_to_src = self.cosmo.T_xy(zstart,self.cosmo.zsrc)

        x_on_src,y_on_src = x + D_to_src*xdef_angle, y + D_to_src*ydef_angle

        betax = self._comoving2angle(x_on_src,self.cosmo.zsrc)
        betay = self._comoving2angle(y_on_src, self.cosmo.zsrc)

        return betax,betay

    def _multi_plane_shoot_write(self, lens_system, x_obs, y_obs,file):

        sorted_indexes = self.sort_redshift_indexes(lens_system.redshift_list)

        x,y = np.zeros_like(x_obs),np.zeros_like(y_obs)

        xdef_angle,ydef_angle = x_obs,y_obs

        with open(file,'a') as f:
            f.write('x_obs,y_obs: '+str(x_obs)+' '+str(y_obs)+'\n\n')

        zstart = 0

        for index in sorted_indexes:

            z = lens_system.redshift_list[index]

            with open(file, 'a') as f:
                f.write('xdef angle/ydef angle: '+str(xdef_angle)+' '+str(ydef_angle)+'\n')
                f.write('zstart,z: '+str(zstart)+' '+str(z)+'\n')

            deflector = lens_system.lens_components[index]

            D_to_plane = self.cosmo.T_xy(zstart,z)

            with open(file, 'a') as f:
                f.write('D_co: '+str(D_to_plane)+'\n\n')

            x,y = self._next_plane(x,y,xdef_angle,ydef_angle,D_to_plane)

            with open(file,'a') as f:
                f.write('next plane: '+str(x)+' '+str(y)+'\n\n')

            xdef_angle,ydef_angle = self._add_deflection_write(x,y,deflector,xdef_angle,ydef_angle,z,file)

            with open(file,'a') as f:
                f.write('updated xdef: '+str(xdef_angle)+'\n')
                f.write('updated ydef: '+str(ydef_angle)+'\n')

            zstart = z

        D_to_src = self.cosmo.T_xy(zstart,self.cosmo.zsrc)

        with open(file,'a') as f:
            f.write('D_co (src): ' +str(D_to_src)+'\n\n')

        x_on_src,y_on_src = x + D_to_src*xdef_angle, y + D_to_src*ydef_angle

        betax = self._comoving2angle(x_on_src,self.cosmo.zsrc)
        betay = self._comoving2angle(y_on_src, self.cosmo.zsrc)

        with open(file,'a') as f:
            f.write('src pos: '+str(betax)+ ' ' + str(betay)+'\n')
        print 'src pos (true): ',str(self.xsrc) + ' ' + str(self.ysrc)

        return betax,betay




