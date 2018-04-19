import numpy as np
from MagniPy.util import *
from MagniPy.MassModels.ExternalShear import Shear
import matplotlib.pyplot as plt

class SIE:

    def __init__(self):
        """
        adopting a standard cosmology, other cosmologies not yet implemented
        :param z1: lens redshift
        :param z2: source redshift
        :param h: little h
        """
        self.Shear = Shear()

    def def_angle(self, x, y, theta_E, q, phi_G, center_x=0, center_y=0, gamma=2,shear=None,shear_theta=None):

        if gamma!=2:
            raise Exception('only isothermal (gamma=2) models allowed')
            return

        xloc = x - center_x
        yloc = y - center_y

        phi_G *= -1
        phi_G += -0.5*np.pi

        shearx,sheary = 0,0

        #if shear is not None:
        #    assert shear_theta is not None
        #    raise Exception('inner SIE shear not implemented.')
        #    shearx,sheary = self.Shear.def_angle(xloc,yloc,shear,shear_theta)

        if q==1:

            r = np.sqrt(xloc ** 2 + yloc ** 2)

            magdef = theta_E

            return shearx+magdef * xloc * r ** -1 , sheary+magdef * yloc * r ** -1

        else:


            q2 = q * q
            qfac = np.sqrt(1 - q2)

            normFac = q * np.sqrt(2 * (1 + q2) ** -1)

            theta_E *= normFac ** -1

            xrot, yrot = rotate(xloc, yloc, -phi_G)
            psi = np.sqrt(q**2*xrot**2+yrot**2)
            psis = psi

            xdef = theta_E * q * qfac ** -1 * np.arctan(qfac * xrot * psis ** -1)
            ydef = theta_E * q * qfac ** -1 * np.arctanh(qfac * yrot * (psi) ** -1)

            xdef,ydef = rotate(xdef,ydef,phi_G)

            return xdef,ydef

    def kappa(self, x, y, theta_E, q, phi_G, center_x=0, center_y=0, gamma=2):

        alpha = 3-gamma

        r_ellip_square = ((x-center_x)**2 + (y-center_y)**2*q**-2)
        rmin = 1e-9
        try:
            r_ellip_square[np.where(r_ellip_square<rmin)] = rmin
        except:
            r_ellip_square = np.max(r_ellip_square,rmin)

        kappa = 0.5*theta_E**(2-alpha)*r_ellip_square**-(1-alpha*0.5)

        return kappa

    def params(self,R_ein = None, ellip = None, ellip_theta = None, x=None,
               y = None, gamma=2,trunc=None,**kwargs):

        subparams = {}
        otherkwargs = {}

        otherkwargs['name']='SPEMD'
        q = 1-ellip
        subparams['q'] = q
        subparams['phi_G'] = (ellip_theta)*np.pi*180**-1
        subparams['gamma'] = gamma
        subparams['center_x'] = x
        subparams['center_y'] = y
        subparams['theta_E'] = R_ein
        #q = subparams['q']
        #subparams['theta_E_fastell'] = R_ein*((1+q**2)*(2*q)**-1)**.5

        return subparams,otherkwargs

    #def R_ein(self,vdis,D_ds,D_s,arcsec):

    #    return 4 * np.pi * (vdis * (0.001 * self.c * self.Mpc) ** -1) ** 2 * \
    #           self.D_ds * self.D_s ** -1 * self.arcsec ** -1
