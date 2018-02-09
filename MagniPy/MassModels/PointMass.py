from MagniPy.LensBuild.cosmology import Cosmo
import numpy as np

class PointMass:

    #pcrit = 2.77536627e+11

    def __init__(self,z1=0.5,z2=1.5,cosmology = None):
        """
        adopting a standard cosmology, other cosmologies not yet implemented
        :param z1: lens redshift
        :param z2: source redshift
        :param h: little h
        """
        if cosmology is None:
            self.cosmology = Cosmo(zd=z1, zsrc=z2)
        else:
            self.cosmology = cosmology

        self.rmin = 10**-8

    def def_angle(self, x, y, center_x=0, center_y=0, R_ein = None, **kwargs):

        x_grid = x - center_x
        y_grid = y - center_y

        r = np.sqrt(x_grid ** 2 + y_grid ** 2)
        r[np.where(r<self.rmin)] = self.rmin
        magdef = R_ein**2*r**-1

        return magdef * x_grid * r ** -1, magdef * y_grid * r ** -1

    def params(self,x=None,y=None,mass=None):

        subkwargs = {}
        otherkwargs = {}
        otherkwargs['name'] = 'POINT_MASS'
        otherkwargs['mass'] = mass

        subkwargs['R_ein'] = self.R_ein(mass)
        subkwargs['center_x'] = x
        subkwargs['center_y'] = y

        return subkwargs,otherkwargs

    def R_ein(self,M):
        const = 4*self.cosmology.G*self.cosmology.D_ds*(self.cosmology.c**2*self.cosmology.D_d*self.cosmology.D_s)**-1*\
                (self.cosmology.kpc_convert*self.cosmology.arcsec**-1)**2 # [Msun ^-1 arcsec ^ 2]
        return (M*const)**.5
