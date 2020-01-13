import numpy as np
from MagniPy.Solver.RayTrace.source_models import *
from MagniPy.util import *
import matplotlib.pyplot as plt

class RayShootingGrid(object):

    def __init__(self, side_length, grid_res, adaptive, rot=0):

        N = int(2*side_length*grid_res**-1)

        self.x_grid_0, self.y_grid_0 = np.meshgrid(
            np.linspace(int(-side_length+grid_res), int(side_length-grid_res), N),
            np.linspace(int(-side_length+grid_res), int(side_length-grid_res), N))

        self.radius = side_length

        self._adaptive = False

        self._rot = rot

    @property
    def grid_at_xy_unshifted(self):
        return self.x_grid_0, self.y_grid_0

    def _grid_at_xy(self, x, y):

        return x + self.x_grid_0, y + self.y_grid_0

    def grid_at_xy(self, xloc, yloc, x_other_list, y_other_list):

        theta = self._rot

        cos_phi, sin_phi = np.cos(theta), np.sin(theta)

        gridx0, gridy0 = self.grid_at_xy_unshifted

        #ellipse_inds = ellipse_coordinates(gridx0, ygrid0, self.radius, q=1,
        #                                   theta=np.arctan2(xloc, yloc) + 0.5 * np.pi)

        _xgrid, _ygrid = (cos_phi * gridx0 + sin_phi * gridy0), (-sin_phi * gridx0 + cos_phi * gridy0)
        xgrid, ygrid = _xgrid + xloc, _ygrid + yloc

        xgrid, ygrid = xgrid.ravel(), ygrid.ravel()

        if self._adaptive:
            for j, (xo, yo) in enumerate(zip(x_other_list, y_other_list)):
                sep = dr(xloc, xo, yloc, yo)
                if sep < 2*self.radius:
                    xgrid, ygrid = self.filter_xy(xgrid, ygrid, xo, yo, xloc, yloc)

        return xgrid, ygrid

    def filter_xy(self, xgrid, ygrid, x_center_other, y_center_other, xloc, yloc):

        delta_r_1 = np.sqrt((xgrid - x_center_other)**2 + (ygrid - y_center_other) ** 2)
        delta_r_2 = np.sqrt((xgrid - xloc) ** 2 + (ygrid - yloc) ** 2)

        inds = np.where(delta_r_1 > delta_r_2)
        #plt.figure(1)
        #ax=plt.gca()
        #print(xloc, yloc, x_center_other, y_center_other)
        #plt.scatter(xgrid[inds], ygrid[inds], color='m', marker='x')
        #plt.scatter(xloc, yloc, color='g')
        #ax.set_aspect('equal')
        #plt.show()
        #a=input('continue')

        return xgrid[inds], ygrid[inds]

class RayTrace(object):

    def __init__(self, xsrc=float, ysrc=float, multiplane=False, res=0.001, source_shape='',
                 polar_grid=False, polar_q = 1, minimum_image_sep = None, adaptive_grid=True,
                 grid_rmax_scale=1, **kwargs):

        """
        :param xsrc: x coordinate for grid center (arcseconds)
        :param ysrc: ""
        :param multiplane: multiple lens plane flag
        :param size: width of the box in asec
        :param res: pixel resolution asec per pixel
        """

        self.polar_grid = polar_grid

        self.xsrc,self.ysrc = xsrc,ysrc
        self.multiplane = multiplane

        self.xsrc,self.ysrc = xsrc,ysrc

        self._source_size_kpc = kwargs['source_size']

        if source_shape == 'GAUSSIAN':
            self.source = GAUSSIAN(x=xsrc,y=ysrc,width=kwargs['source_size'])
            self.grid_rmax, self.res = self._grid_rmax(kwargs['source_size'],res)
        else:
            raise ValueError('other source models not yet implemented')

        self.grid_rmax *= grid_rmax_scale
        self.grid_rmax = int(self.grid_rmax)

        if adaptive_grid is True:
            #print('warning: adaptive grid not yet implemented')
            pass
            #adaptive_grid = False

        if adaptive_grid is False:
            self.grid = []

            if minimum_image_sep is not None:
                for j in range(0,len(minimum_image_sep[0])):
                    sep = minimum_image_sep[0][j]
                    theta = minimum_image_sep[1][j]
                    L = int(0.5*sep)
                    self.grid.append(RayShootingGrid(int(min(self.grid_rmax, L)), self.res, adaptive_grid, rot=theta))
        else:
            self.grid = [RayShootingGrid(self.grid_rmax, self.res, adaptive_grid)]*4

        self._adaptive_grid = adaptive_grid

    def _grid_rmax(self,size_asec,res):

        if size_asec < 0.0002:
            s = 0.005
        elif size_asec < 0.0005:
            s = 0.03
        elif size_asec < 0.001:
            s = 0.08
        elif size_asec < 0.002:
            s = 0.2
        elif size_asec < 0.003:
            s = 0.28
        elif size_asec < 0.005:
            s = 0.35

        else:
            s = 0.48

        return s,res

    def get_images(self,xpos,ypos,lensModel,kwargs_lens,return_image=False):

        if isinstance(xpos,float) or isinstance(xpos,int):
            xpos,ypos = self._get_grids(np.array([xpos]),np.array([ypos]),1)
            xpos = xpos[0]
            ypos = ypos[0]
        else:
            xpos,ypos = self._get_grids(xpos,ypos,len(xpos))
        #del kwargs_lens[0]['source_size_kpc']
        img = self.rayshoot(xpos,ypos,lensModel,kwargs_lens)
        #try:
        #n = int(len(img) ** 0.5)
        #print('npixels: ' , n)
        #plt.imshow(img.reshape(n,n)); plt.show()
        #a=input('continue')
        #except:
        #    pass
        if return_image:
            return np.sum(img)*self.res**2,array2image(img)
        else:
            return np.sum(img)*self.res**2

    def magnification(self,xpos,ypos,lensModel,kwargs_lens):

        if self._source_size_kpc == 0:
            flux = lensModel.magnification(xpos, ypos, kwargs_lens)

            return np.absolute(flux)

        flux = []
        xgrids, ygrids = self._get_grids(xpos, ypos, len(xpos))

        for i in range(0,len(xpos)):

            image = self.rayshoot(xgrids[i],ygrids[i],lensModel,kwargs_lens)

            #n = int(np.sqrt(len(image)))
            #print('npixels: ' , n)
            #plt.imshow(image.reshape(n,n)); plt.show()
            #a=input('continue')
            #blended = flux_at_edge(image.reshape(n,n))
            #blended = False
            #if blended:
            #    flux.append(np.nan)
            #else:
            flux.append(np.sum(image*self.res**2))

            #plt.imshow(image.reshape(n,n))
            #plt.show()
            #a=input('continue')

        return np.array(flux)

    def rayshoot(self,x,y,lensModel,kwargs_lens):

        xnew,ynew = lensModel.ray_shooting(x.ravel(),y.ravel(),kwargs_lens)

        beta = self.source(xnew,ynew)

        return beta

    def _get_grids(self, xpos, ypos, nimg):

        xgrid, ygrid = [], []

        for i, (xi, yi) in enumerate(zip(xpos, ypos)):

            xother, yother = [], []
            for j in range(0, len(xpos)):
                if j != i:
                    xother.append(xpos[j])
                    yother.append(ypos[j])

            xg, yg = self.grid[i].grid_at_xy(xi, yi, xother, yother)

            xgrid.append(xg)
            ygrid.append(yg)

        return xgrid, ygrid

