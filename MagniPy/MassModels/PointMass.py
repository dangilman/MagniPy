from MagniPy.LensBuild.cosmology import Cosmo
import numpy as np

class PointMass(Cosmo):

    #pcrit = 2.77536627e+11

    def __init__(self,z1=0.5,z2=1.5,c_turnover=True):
        """
        adopting a standard cosmology, other cosmologies not yet implemented
        :param z1: lens redshift
        :param z2: source redshift
        :param h: little h
        """
        Cosmo.__init__(self, zd=z1, zsrc=z2)
        self.c_turnover=c_turnover

    def def_angle(self, x_grid, y_grid, x=None, y=None, R_ein = None, **kwargs):

        x = x_grid - x
        y = y_grid - y

        r = np.sqrt(x ** 2 + y ** 2 + 0.0000000000001)
        magdef = R_ein**2*r**-1

        return magdef * x * r ** -1, magdef * y * r ** -1

    def params(self,x,y,M):

        subkwargs = {}
        subkwargs['R_ein'] = self.R_ein(M)
        subkwargs['x'] = x
        subkwargs['y'] = y


        return subkwargs,subkwargs

    def R_ein(self,M):
        const = 4*self.G*self.D_ds*(self.c**2*self.D_d*self.D_s)**-1*(self.kpc_convert*self.arcsec**-1)**2 # [Msun ^-1 arcsec ^ 2]
        return (M*const)**.5
