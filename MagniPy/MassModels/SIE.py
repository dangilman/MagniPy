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

    def def_angle(self, xgrid, ygrid, x=None, y=None, rcore=0, R_ein=None, ellip=None, ellip_theta=None,
                  shear=0, shear_theta=0, **kwargs):

        ellip_theta*=-1
        ellip_theta+=-90 # to gravlens standard

        xloc = xgrid - x
        yloc = ygrid - x

        if shear!=0:
            s = Shear()

            shearx,sheary = s.def_angle(xgrid,ygrid,shear,shear_theta-90)

        else:
            shearx,sheary = 0,0

        if ellip==0:

            r = np.sqrt(xloc ** 2 + yloc ** 2)

            magdef = R_ein * r ** -1 * (np.sqrt(rcore ** 2 + r ** 2) - rcore)


            return magdef * xloc * r ** -1 + shearx, magdef * yloc * r ** -1 + sheary

        else:

            q=1-ellip
            q2 = q*q
            qfac = np.sqrt(1-q2)

            normFac = q*np.sqrt(2*(1+q2)**-1)

            R_ein*= normFac ** -1
            rcore*=normFac**-1

            xrot, yrot = rotate(xloc, yloc,-ellip_theta)
            psi = np.sqrt(q2*(xrot**2+rcore**2)+yrot**2)
            psis = psi + rcore

            xdef = R_ein * q * qfac ** -1 * np.arctan(qfac * xrot * psis ** -1)
            ydef = R_ein * q * qfac ** -1 * np.arctanh(qfac * yrot * (psi + q2 * rcore) ** -1)


            xdef,ydef = rotate(xdef,ydef,ellip_theta)

            return xdef+shearx,ydef+sheary

    def convergence(self, rcore, rtrunc):
        return None

    def params(self,R_ein = None, ellip = None, ellip_theta = None, x=None,
               y = None, shear=0, shear_theta=0,trunc=None):

        subparams = {}
        subparams['name']='SIE'
        subparams['x'] = x
        subparams['y'] = y
        subparams['R_ein'] = R_ein
        subparams['ellip'] = ellip
        subparams['ellip_theta'] = ellip_theta
        subparams['shear'] = shear
        subparams['shear_theta'] = shear_theta
        subparams['trunc'] = trunc
        subparams['lenstronomy_name'] = 'SPEMD'

        lenstronomy_params = {}
        lenstronomy_params['theta_E'] = R_ein
        lenstronomy_params['q'] = 1 - ellip
        lenstronomy_params['phi_G'] = ellip_theta*-1 - 90
        lenstronomy_params['gamma'] = 2

        return subparams,lenstronomy_params

    def R_ein(self,vdis,z1,z2):
        Cosmo.__init__(self, zd=z1, zsrc=z2)
        return 4 * np.pi * (vdis * (0.001 * self.c * self.Mpc) ** -1) ** 2 * \
               self.D_ds * self.D_s ** -1 * self.arcsec ** -1


