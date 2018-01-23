import numpy as np
from source_models import *
from MagniPy.util import *

class RayTrace:

    def __init__(self,xsrc=float,ysrc=float,multiplane=False,gridsize=int,res=0.001,source_shape='',**kwargs):

        """

        :param xsrc: x coordinate for grid center (arcseconds)
        :param ysrc: ""
        :param multiplane: multiple lens plane flag
        :param size: width of the box in asec
        :param res: pixel resolution asec per pixel
        """

        self.gridsize = gridsize
        self.res = res
        self.multiplane = multiplane

        if source_shape == 'GAUSSIAN':
            self.source = GAUSSIAN(x=xsrc,y=ysrc,width=kwargs['source_size'])
        else:
            raise ValueError('other source models not yet implemented')

        self.x_grid_0, self.y_grid_0 = make_grid(gridsize=self.gridsize * self.res ** -1, deltapix=self.res)

    def compute_mag(self,xpos,ypos,lens_system):

        if self.multiplane:
            return self._single_plane_trace(xpos,ypos,lens_system)
        else:
            raise ValueError('multiplane lensing not yet implemented')

    def _single_plane_trace(self,xpos,ypos,lens_system):

        magnification = []

        for i in range(0,len(xpos)):

            xdef = self.x_grid_0
            ydef = self.y_grid_0

            for deflector in lens_system.lens_components:

                xplus,yplus = deflector.lensing.def_angle(xdef,ydef,**deflector.args)

                xdef+=xplus
                ydef+=yplus

            magnification.append(np.sum(self.source.source_profile(x=xdef,y=ydef))*self.res**2)

        return np.array(magnification)
