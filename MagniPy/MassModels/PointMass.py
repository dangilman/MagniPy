from MagniPy.LensBuild.cosmology import Cosmo
import numpy as np

class PointMass(Cosmo):

    #pcrit = 2.77536627e+11

    def __init__(self,z1=0.5,z2=1.5):
        """
        adopting a standard cosmology, other cosmologies not yet implemented
        :param z1: lens redshift
        :param z2: source redshift
        :param h: little h
        """
        Cosmo.__init__(self, zd=z1, zsrc=z2)

        self.rmin = 10**-8

    def translate_to_lensmodel(self,**args):

        newargs = {}
        newargs['R_ein'] = args['R_ein']
        newargs['x'] = args['center_x']
        newargs['y'] = args['center_y']
        return newargs

    def translate_to_lenstronomy(self,**args):

        newargs = {}
        newargs['R_ein'] = args['R_ein']
        newargs['x'] = args['center_x']
        newargs['y'] = args['center_y']

        return newargs

    def def_angle(self, x_grid, y_grid, x=None, y=None, R_ein = None, **kwargs):

        x = x_grid - x
        y = y_grid - y

        r = np.sqrt(x ** 2 + y ** 2)
        r[np.where(r<self.rmin)] = self.rmin
        magdef = R_ein**2*r**-1

        return magdef * x * r ** -1, magdef * y * r ** -1

    def params(self,x=None,y=None,mass=None):

        subkwargs = {}
        subkwargs['name'] = 'ptmass'
        subkwargs['R_ein'] = self.R_ein(mass)
        subkwargs['lenstronomy_name'] = 'POINTMASS'
        subkwargs['x'] = x
        subkwargs['y'] = y

        lenstronomykwargs = {}
        lenstronomykwargs['R_ein'] = subkwargs['R_ein']
        lenstronomykwargs['x'] = x
        lenstronomykwargs['y'] = y

        return subkwargs,lenstronomykwargs

    def R_ein(self,M):
        const = 4*self.G*self.D_ds*(self.c**2*self.D_d*self.D_s)**-1*(self.kpc_convert*self.arcsec**-1)**2 # [Msun ^-1 arcsec ^ 2]
        return (M*const)**.5
