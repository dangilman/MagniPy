import numpy as np
from magnipy.MassModels.grid_location import Local
from magnipy.MassModels import NFW,PJaffe,PTmass,SIE
from DarkHalo.Cosmo import cosmo


class DeflectionPlane:

    def __init__(self,x_c=float,y_c=float,redshift=float,size=int,res=0.0004):

        """

        :param x_c: x coordinate for grid center (arcseconds)
        :param y_c: ""
        :param redshift: redshift of lens plane
        :param size: 1/2 the width of the box in arcseconds
        :param res: pixel resolution m.a.s. per pixel
        """

        self.x,self.y = x_c,y_c
        self.z = redshift
        self.size = size

        size *= .001
        self.gridsize = size
        steps = max(1, 2 * round(size * res ** -1))

        self.grid = Local(x0=self.x,y0=self.y,redshift=redshift,res=res,size=size)

        self.xdef,self.ydef = np.zeros_like(self.grid.xgrid),np.zeros_like(self.grid.ygrid)

        self.subs_inplane = []

    def add_deflectors(self,halos=classmethod):

        """
        :param halos: instances of Halo classes
        :return:
        """

        for halo in halos:
            dx,dy = halo.lensing.def_angle(self.grid.xgrid,self.grid.ygrid,**halo.lens_args)
            self.xdef += dx
            self.ydef += dy
