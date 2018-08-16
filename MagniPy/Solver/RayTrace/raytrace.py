import numpy as np
from MagniPy.Solver.RayTrace.source_models import *
from MagniPy.util import *

class RayTrace:

    def __init__(self, xsrc=float, ysrc=float, multiplane=False, grid_rmax_arcsec_source=float, res=0.001, source_shape='',
                 polar_grid=False, polar_q = 0.5, **kwargs):

        """
        :param xsrc: x coordinate for grid center (arcseconds)
        :param ysrc: ""
        :param multiplane: multiple lens plane flag
        :param size: width of the box in asec
        :param res: pixel resolution asec per pixel
        """

        self.polar_grid = polar_grid
        self.grid_rmax = grid_rmax_arcsec_source
        self.res = res
        self.xsrc,self.ysrc = xsrc,ysrc
        self.multiplane = multiplane

        self.xsrc,self.ysrc = xsrc,ysrc

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

    def get_images(self,xpos,ypos,lens_system):

        if isinstance(xpos,float) or isinstance(xpos,int):
            xpos,ypos = self._get_grids([xpos],[ypos],1)
            xpos = xpos[0]
            ypos = ypos[0]

        xpos,ypos = self._get_grids(xpos,ypos,len(xpos))

        img = self.rayshoot(xpos,ypos,lens_system)

        if self.polar_grid:
            return np.sum(img)*self.res**2,img
        else:
            return np.sum(img)*self.res**2,array2image(img)

    def magnification(self,xpos,ypos,lensModel,kwargs_lens):

        flux = []

        for i in range(0,len(xpos)):

            image = self.rayshoot(xpos[i],ypos[i],lensModel,kwargs_lens)

            flux.append(np.sum(image*self.res**2))

        return np.array(flux)

    def rayshoot(self,x,y,lensModel,kwargs_lens):

        xnew,ynew = lensModel.ray_shooting(x.ravel(),y.ravel(),kwargs_lens)

        beta = self.source(xnew,ynew)

        return beta

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


