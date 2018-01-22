import numpy as np
from MagniPy.LensBuild.cosmology import Cosmo
from MagniPy.util import *
from ExternalShear import Shear

class SIE_lens:

    def def_angle(self,xgrid,ygrid,x0=None,y0=None,rcore=0,rtrunc=0,b=None,ellip=None,ellip_theta=None,
                  shear=0,shear_theta=0):

        ellip_theta*=-1
        ellip_theta+=-90 # to gravlens standard

        x = xgrid - x0
        y = ygrid - y0

        if shear!=0:
            shearx,sheary = Shear().def_angle(xgrid,ygrid,shear,shear_theta)
        else:
            shearx,sheary = 0,0

        if ellip==False or ellip==0:
            # shear_theta*=-1

            r = np.sqrt(x ** 2 + y ** 2)

            magdef = b * r ** -1 * (np.sqrt(rcore ** 2 + r ** 2) - rcore)

            return magdef * x * r ** -1 + shearx, magdef * y * r ** -1 + sheary

        else:

            q=1-ellip
            q2 = q*q
            qfac = np.sqrt(1-q2)

            normFac = q*np.sqrt(2*(1+q2)**-1)

            b*=normFac**-1
            rcore*=normFac**-1

            xrot, yrot = rotate(x, y,-ellip_theta)
            psi = np.sqrt(q2*(xrot**2+rcore**2)+yrot**2)
            psis = psi + rcore

            xdef = b * q * qfac ** -1 * np.arctan(qfac * xrot * psis**-1)
            ydef = b * q * qfac ** -1 * np.arctanh(qfac * yrot * (psi+q2*rcore)**-1)


            xdef,ydef = rotate(xdef,ydef,ellip_theta)

            return xdef+shearx,ydef+sheary

    def convergence(self, rcore, rtrunc):
        return None

class SIE(Cosmo):

    def __init__(self,z1=0.5,z2=1.5):
        """
        adopting a standard cosmology, other cosmologies not yet implemented
        :param z1: lens redshift
        :param z2: source redshift
        :param h: little h
        """
        pass

    def params(self,R_ein = None, vdis = None, ellip = None, ellip_theta = None, x0=None,
               y0 = None, shear=0, shear_theta=0):

        if R_ein is None:
            R_ein = self.R_ein(vdis)
        subparams = {}
        subparams['name']='SIE'
        subparams['b'] = R_ein
        subparams['ellip'] = ellip
        subparams['ellip_theta'] = ellip_theta
        subparams['shear'] = shear
        subparams['shear_theta'] = shear_theta

        return subparams

    def R_ein(self,vdis,z1,z2):
        Cosmo.__init__(self, zd=z1, zsrc=z2)
        return 4 * np.pi * (vdis * (0.001 * self.c * self.Mpc) ** -1) ** 2 * \
               self.D_ds * self.D_s ** -1 * self.arcsec ** -1


