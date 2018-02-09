import numpy as np
from MagniPy.LensBuild.cosmology import Cosmo
from MagniPy.util import *
from MagniPy.MassModels.ExternalShear import Shear
import matplotlib.pyplot as plt

class SIE(Cosmo):

    def __init__(self,z1=0.5,z2=1.5):
        """
        adopting a standard cosmology, other cosmologies not yet implemented
        :param z1: lens redshift
        :param z2: source redshift
        :param h: little h
        """
        pass

    def kappa(self,x,y,theta_E,q,phi_G,center_x=0, center_y=0,gamma=2):

        rtol = 1e-6
        r = np.sqrt(x**2+y**2)
        r[np.where(r<rtol)] = rtol

        return 0.5*r**-1

    def def_angle(self, x, y, theta_E, q, phi_G, center_x=0, center_y=0, gamma=2):

        xloc = x - center_x
        yloc = y - center_y

        phi_G *= -1
        phi_G += 0.5*np.pi

        if q==1:

            r = np.sqrt(xloc ** 2 + yloc ** 2)

            magdef = theta_E

            return magdef * xloc * r ** -1 , magdef * yloc * r ** -1

        else:

            #q2 = q*q
            qfac = np.sqrt(1-q**2)

            #normFac = q*np.sqrt(2*(1+q2)**-1)
            normFac = q**-.5

            #theta_E *= normFac ** -1
            theta_E *= normFac
            xrot, yrot = rotate(xloc, yloc, -phi_G)
            psi = np.sqrt(q**2*(xrot**2)+yrot**2)
            psis = psi

            xdef = theta_E * q * qfac ** -1 * np.arctan(qfac * xrot * psis ** -1)
            ydef = theta_E * q * qfac ** -1 * np.arctanh(qfac * yrot * (psi) ** -1)


            xdef,ydef = rotate(xdef,ydef,phi_G)

            return xdef,ydef

    def convergence(self, rcore, rtrunc):
        return None

    def params(self,R_ein = None, ellip = None, ellip_theta = None, x=None,
               y = None, shear=0, shear_theta=0,trunc=None):

        subparams = {}
        otherkwargs = {}

        otherkwargs['name']='SPEMD'
        q = 1-ellip
        subparams['q'] = q
        subparams['phi_G'] = (ellip_theta)*np.pi*180**-1
        subparams['gamma'] = 2
        subparams['center_x'] = x
        subparams['center_y'] = y
        subparams['theta_E'] = R_ein*((1+q**2)*(2*q)**-1)**.5
        #q = subparams['q']
        #subparams['theta_E_fastell'] = R_ein*((1+q**2)*(2*q)**-1)**.5

        return subparams,otherkwargs

    def R_ein(self,vdis,z1,z2):
        Cosmo.__init__(self, zd=z1, zsrc=z2)
        return 4 * np.pi * (vdis * (0.001 * self.c * self.Mpc) ** -1) ** 2 * \
               self.D_ds * self.D_s ** -1 * self.arcsec ** -1


