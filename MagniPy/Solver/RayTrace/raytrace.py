import numpy as np
from MagniPy.Solver.LenstronomyWrap.MultiLensWrap import MultiLensWrapper
from source_models import *
from MagniPy.util import *
import matplotlib.pyplot as plt

class RayTrace:

    def __init__(self,xsrc=float,ysrc=float,multiplane=False,gridsize=int,res=0.0005,source_shape='',cosmology=None,**kwargs):

        """

        :param xsrc: x coordinate for grid center (arcseconds)
        :param ysrc: ""
        :param multiplane: multiple lens plane flag
        :param size: width of the box in asec
        :param res: pixel resolution asec per pixel
        """

        self.gridsize = gridsize
        self.res = res
        self.xsrc,self.ysrc = xsrc,ysrc
        self.multiplane = multiplane

        if multiplane:
            self.multilens = MultiLensWrapper(gridsize=self.gridsize,res=self.res,source_shape=source_shape,
                                         astropy_class=cosmology,z_source=kwargs['zsrc'],source_size=kwargs['source_size'])

        if source_shape == 'GAUSSIAN':
            self.source = GAUSSIAN(x=xsrc,y=ysrc,width=kwargs['source_size'])
        else:
            raise ValueError('other source models not yet implemented')

        self.cosmo = cosmology

        self.x_grid_0, self.y_grid_0 = make_grid(numPix=self.gridsize * self.res ** -1, deltapix=self.res)
        self.y_grid_0 *= -1

    def compute_mag(self,xpos,ypos,lens_system,print_mag=False):

        if self.multiplane is False:
            return self._single_plane_trace(xpos,ypos,lens_system,print_mag)
        else:
            return self.multilens.compute_mag(xpos,ypos,lens_system)

    def _single_plane_trace(self,xpos,ypos,lens_system,print_mag=False):

        magnification = []

        if print_mag:
            print 'computing mag...'
        for i in range(0,len(xpos)):

            x_loc = xpos[i]*np.ones_like(self.x_grid_0)+self.x_grid_0
            y_loc = ypos[i]*np.ones_like(self.y_grid_0)+self.y_grid_0

            xdef = np.zeros_like(x_loc)
            ydef = np.zeros_like(y_loc)

            for count,deflector in enumerate(lens_system.lens_components):

                xplus,yplus = deflector.lensing.def_angle(x_loc,y_loc,**deflector.args)

                xdef+=xplus
                ydef+=yplus
                if print_mag and i==0:
                    print count+1

            x_source = x_loc - xdef
            y_source = y_loc - ydef

            magnification.append(np.sum(self.source.source_profile(betax=x_source,betay=y_source))*self.res**2)

        return np.array(magnification)





