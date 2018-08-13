import numpy as np
from MagniPy.Solver.RayTrace.source_models import *
from MagniPy.util import *
import matplotlib.pyplot as plt
from MagniPy.Solver.LenstronomyWrap import lenstronomy_wrap,MultiLensWrap

class RayTrace:

    def __init__(self, xsrc=float, ysrc=float, multiplane=False, grid_rmax=int, res=0.0005, source_shape='', cosmology=None,
                 raytrace_with=None,
                 polar_grid=False, polar_q = 0.4, **kwargs):

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


        if polar_grid:
            self.polar_q = polar_q
            self.x_grid_0 = self.x_grid_0.ravel()
            self.y_grid_0 = self.y_grid_0.ravel()

        if source_shape == 'GAUSSIAN':
            self.source = GAUSSIAN(x=xsrc,y=ysrc,width=kwargs['source_size'])
        elif source_shape == 'TORUS':
            self.source = TORUS(x=xsrc,y=ysrc,inner=kwargs['inner_radius'],outer=kwargs['outer'])
        elif source_shape == 'GAUSSIAN_TORUS':
            self.source = GAUSSIAN_TORUS(x=xsrc, y=ysrc, width=kwargs['source_size'],inner=kwargs['inner_radius'],outer=kwargs['outer_radius'])
        elif source_shape == 'SERSIC':
            self.source = SERSIC(x=xsrc,y=ysrc,r_half=kwargs['r_half'],n_sersic=kwargs['n_sersic'])
        elif source_shape == 'GAUSSIAN_SERSIC':
            self.source = GAUSSIAN_SERSIC(x=xsrc,y=ysrc,width=kwargs['source_size'],r_half=kwargs['r_half'],n_sersic=kwargs['n_sersic'])
        else:
            raise ValueError('other source models not yet implemented')

        if self.raytrace_with == 'lenstronomy':

            self.multilenswrap = MultiLensWrap.MultiLensWrapper()

    def _get_grids(self,xpos,ypos,Nimg):

        x_loc,y_loc = [],[]
        for i in range(0,Nimg):

            if self.polar_grid:
                #ellipse_inds = np.where(np.sqrt(self.x_grid_0 ** 2 + self.y_grid_0 ** 2) <= self.grid_rmax)
                ellipse_inds = ellipse_coordinates(self.x_grid_0, self.y_grid_0, self.grid_rmax,q=self.polar_q,
                                                   theta=np.arctan2(ypos[i], xpos[i])+0.5*np.pi)
                x_loc.append(xpos[i] + self.x_grid_0[ellipse_inds])
                y_loc.append(ypos[i] + self.y_grid_0[ellipse_inds])

            else:
                x_loc.append(xpos[i] + self.x_grid_0)
                y_loc.append(ypos[i] + self.y_grid_0)
        return x_loc,y_loc

    def get_images(self,xpos,ypos,lens_system,**kwargs):

        if isinstance(xpos,float) or isinstance(xpos,int):
            xpos,ypos = self._get_grids([xpos],[ypos],1)
            xpos = xpos[0]
            ypos = ypos[0]
        else:
            xpos,ypos = self._get_grids(xpos,ypos,len(xpos))

        if self.raytrace_with == 'lensmodel':

            return self.compute_mag(xpos,ypos,lens_system,**kwargs)

        else:

            img = self.multilenswrap.rayshoot(xpos,ypos,lens_system,source_function=self.source,
                                              astropy=self.cosmo.cosmo,zsrc=self.cosmo.zsrc)

            if self.polar_grid:
                return np.sum(img)*self.res**2,img
            else:
                return np.sum(img)*self.res**2,array2image(img)

    def compute_mag(self,xpos,ypos,lensmodel=None,lens_model_params=None,lens_system=None,ray_shooting_function=None):

        x_loc, y_loc = self._get_grids(xpos, ypos, len(xpos))
        N = len(xpos)

        if self.raytrace_with == 'lenstronomy':

            return self.multilenswrap.magnification(x_loc,y_loc,lensmodel,lens_model_params,
                                                        self.source,self.res,n=N,ray_shooting_function=ray_shooting_function)

        else:
            raise Exception('not yet implemented')
            if self.multiplane:
                return self._multi_plane_trace(x_loc, y_loc, lens_system, **kwargs)
            else:
                return self._single_plane_trace(x_loc,y_loc,lens_system,**kwargs)

    def _single_plane_trace_full(self,xx,yy,lens_system,to_img_plane=False,print_mag=False,return_image=False):

        if print_mag:
            print('computing mag...')

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

            if print_mag and count==0:
                print(count+1)

        if to_img_plane:

            return xdef,ydef

        else:
            x_source = x_loc - xdef
            y_source = y_loc - ydef
            return x_source,y_source

    def _single_plane_trace(self,xpos,ypos,lens_system,print_mag=False,return_image=False):

        magnification = []

        if print_mag:
            print('computing mag...')

        for i in range(0,len(xpos)):

            x_loc = xpos[i]
            y_loc = ypos[i]

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
                    print(count+1)

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

            xcoords = xpos[i]
            ycoords = ypos[i]

            x_source, y_source = self._multi_plane_shoot(lens_system,xcoords,ycoords)

            magnification.append(np.sum(self._eval_src(x_source, y_source) * self.res ** 2))

            if return_image:

                image = self._eval_src(x_source,y_source)

        if return_image:
            return np.array(magnification),image
        else:
            return np.array(magnification)

    def _sort_redshift_indexes(self, redshift_list):
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

        sorted_indexes = self._sort_redshift_indexes(lens_system.redshift_list)

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

        sorted_indexes = self._sort_redshift_indexes(lens_system.redshift_list)

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
        print('src pos (true): ',str(self.xsrc) + ' ' + str(self.ysrc))

        return betax,betay






