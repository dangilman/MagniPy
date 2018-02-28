from MagniPy.LensBuild.Cosmology.cosmology import Cosmo
import numpy as np

class PointMass:

    #pcrit = 2.77536627e+11

    def __init__(self,z=None,zsrc=None,cosmology = None):
        """
        adopting a standard cosmology, other cosmologies not yet implemented
        :param z1: lens redshift
        :param z2: source redshift
        :param h: little h
        """
        if cosmology is None:
            self.cosmology = Cosmo(zd=z, zsrc=zsrc,compute=False)
        else:
            self.cosmology = cosmology

        self.D_s,self.D_d,self.D_ds = self.cosmology.D_A(0,zsrc),self.cosmology.D_A(0,z),self.cosmology.D_A(z,zsrc)

        self.rmin = 10**-9

    def def_angle(self, x, y, center_x=0, center_y=0, theta_E = None, **kwargs):

        x_grid = x - center_x
        y_grid = y - center_y

        r = np.sqrt(x_grid ** 2 + y_grid ** 2)
        r[np.where(r<self.rmin)] = self.rmin
        magdef = theta_E**2*r**-1

        return magdef * x_grid * r ** -1, magdef * y_grid * r ** -1

    def params(self,x=None,y=None,mass=None):

        subkwargs = {}
        otherkwargs = {}
        otherkwargs['name'] = 'POINT_MASS'
        otherkwargs['mass'] = mass

        subkwargs['theta_E'] = self.R_ein(mass)
        subkwargs['center_x'] = x
        subkwargs['center_y'] = y

        return subkwargs,otherkwargs

    def R_ein(self,M):

        const = 4*self.cosmology.G*self.cosmology.c**-2*self.D_ds*(self.D_d*self.D_s)**-1
        return self.cosmology.arcsec**-1*(M*const)**.5
