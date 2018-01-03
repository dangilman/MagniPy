import numpy as np

class Sersic:

    def __init__(self,xgrid,ygrid):

        self.x,self.y = xgrid,ygrid

    def b(self,n):
        return 1.9992*n - 0.3271 + 4*(405*n)**-1

    def convergence(self, r=None, b=None, re=None, n=None, xy=None, ellip=None,ellip_PA=None):

        if xy is not None:
            r = np.sqrt(xy[0]**2+xy[1]**2+1e-9)
        if r is None:
            r = np.sqrt(self.x**2+self.y**2+1e-9)

        bn = self.b(n)

        if ellip==0 or ellip is None:

            return b*np.exp(-bn*((r*re**-1)**(1*n**-1)-1))
        else:

            return b*np.exp(-bn*((r*re**-1)**(1*n**-1)-1))


    def def_angle(self,x0=None,y0=None,n=None,re=None,ke=None,ellip=None,ellip_PA=None):

        x = self.x - x0
        y = self.y - y0

        x = (np.sqrt(x**2+y**2+1e-9)*re**-1)**(-n)
        b = self.b(n)

        if ellip==0 or ellip is None:

            alpha_e = n*re*ke*b**(-2*n)*np.exp(b)*gamma(2*n)
            return 2*alpha_e*x**(-n)*(1-gammainc(2*n,b*x)*gamma(2*n)**-1)

        else:
            q = 1-ellip
            ellip_PA *= -1
            ellip_PA += -90

            xrot, yrot = rotate(x, y, -ellip_PA)

            xdef,ydef = deflection_integrals.xydef(xrot,yrot,q,self.convergence,b=ke,re=re,n=n,ellip=ellip,ellip_PA=ellip_PA)

            xdef, ydef = rotate(xdef, ydef, ellip_PA)
            return xdef,ydef
