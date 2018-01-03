import numpy as np
from magnipy.util import *

class SIE:
    def __init__(self,xgrid,ygrid):

        self.x, self.y = xgrid, ygrid

    def convergence(self, rcore, rtrunc):
        return 1

    def def_angle(self,x0=None,y0=None,rcore=None,rtrunc=None,b=None,ellip=None,ellip_theta=None,shear=None,shear_theta=None):

        ellip_theta*=-1
        ellip_theta+=-90 # to gravlens standard
        shear_theta += -90

        x = self.x - x0
        y = self.y - y0

        phi = np.arctan2(y, x)
        e1, e2 = shear * np.cos(2 * (phi - shear_theta * np.pi / 180)), shear * np.sin(
            2 * (phi - shear_theta * np.pi / 180))
        shearx = -e1 * x - e2 * y
        sheary = e2 * x - e1 * y

        if ellip==False or ellip==0:
            # shear_theta*=-1

            r = np.sqrt(x ** 2 + y ** 2)

            magdef = b * r ** -1 * (np.sqrt(rcore ** 2 + r ** 2) - rcore)

            return magdef * x * r ** -1+shearx, magdef * y * r ** -1+sheary

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
