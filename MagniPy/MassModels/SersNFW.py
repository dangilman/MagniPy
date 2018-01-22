import numpy as np

class SersNFW:

    def __init__(self, xgrid=None, ygrid=None):
        if xgrid is None and ygrid is None:
            x, y = np.linspace(-500, 500, 1000), np.linspace(-500, 500, 1000)
            xgrid,ygrid = np.meshgrid(x,y)
            print Warning('Did not specify grid position, defaulting to (0,0), (1000,1000) pixels')

        self.x, self.y = xgrid, ygrid
        self.f, self.r0fac, self.rsfac = 1*3**-1,0.5,5
        self.nfw = NFW(xgrid=self.x, ygrid=self.y)
        self.sers = Sersic(xgrid=self.x, ygrid=self.y)

    def b(self,n):
        return 1.9992*n - 0.3271 + 4*(405*n)**-1
    def get_rs(self,re):
        return self.rsfac*re

    def normalizations(self,Rein=None,re=None,n=None,f=1./3,r0fac=0.5,rsfac=5):

        def G(x):

            return np.log(0.25*x**2)+2*np.arctanh(np.sqrt(1-x**2))*np.sqrt(1-x**2)**-1
        def C(r,R_e,n):

            b = self.b(n)
            return (n*np.exp(b)*b**(-2*n))*(gamma(2*n) - gammainc(2*n,b*(r*R_e**-1)**(1*n**-1)))

        def norm_sersic(ks,R_ein,re,n,f=1*3**-1,r0fac=0.5,rsfac=5):
            r0 = r0fac*re
            rs = rsfac*re
            n2 = r0*rs**-1
            f_fac = (1-f)*f**-1
            ratio = (rs*re**-1)**2
            return f_fac*ks*G(n2)*ratio**2*C(r0,re,n)**-1


        def norm_nfw(R_ein,re,n,f=1*3**-1,r0fac=0.5,rsfac=5):
            r0 = r0fac * re
            rs = rsfac * re
            n1 = R_ein * rs ** -1
            n2 = r0 * rs ** -1
            f_fac = (1 - f) * f ** -1
            ratio1 = (R_ein * rs ** -1) ** 2

            return 0.5*ratio1*(G(n1)+f_fac*G(n2)*C(R_ein,re,n)*C(r0,re,n)**-1)**-1

        nfwnorm = norm_nfw(Rein, re, n, f, r0fac, rsfac)
        snorm = norm_sersic(nfwnorm,Rein,re,n,f,r0fac,rsfac)

        return snorm,nfwnorm

    def convergence(self,Rein=None,re=None,n=None,ellip=None,ellip_PA=None,shear=None,shear_PA=None):

        f,r0fac,rsfac = self.f,self.r0fac,self.rsfac

        rs = rsfac*re

        ks,sersnorm = self.normalizations(Rein,re,n,f,r0fac,rsfac)

        if ellip is None or ellip==0:

            return self.nfw.convergence(ks=ks,rs=rs)+self.sers.convergence(b=sersnorm,re=re,n=n,ellip=ellip)

    def conv_insersic(self,Rein=None,re=None,n=None,ellip=None,ellip_PA=None,shear=None,shear_PA=None):

        f,r0fac,rsfac = self.f,self.r0fac,self.rsfac

        rs = rsfac*re

        ks,sersnorm = self.normalizations(Rein,re,n,f,r0fac,rsfac)

        if ellip is None or ellip==0:

            return self.sers.convergence(b=sersnorm,re=re,n=n,ellip=ellip)

    def conv_innfw(self,Rein=None,re=None,n=None,ellip=None,ellip_PA=None,shear=None,shear_PA=None):

        f,r0fac,rsfac = self.f,self.r0fac,self.rsfac

        rs = rsfac*re

        ks,sersnorm = self.normalizations(Rein,re,n,f,r0fac,rsfac)

        if ellip is None or ellip==0:

            return self.nfw.convergence(ks=ks,rs=rs)

    def def_angle(self,x0=None,y0=None,Rein=None,re=None,n=None,ellip=None,ellip_PA=None,shear=None,shear_PA=None):

        f, r0fac, rsfac = self.f, self.r0fac, self.rsfac

        rs = rsfac * re

        ks, sersnorm = self.normalizations(Rein, re, n, f, r0fac, rsfac)

        return self.nfw.def_angle(x0=x0,y0=y0,rs=rs,ks=ks)+self.sers.def_angle(x0=x0,y0=y0,n=n,re=re,ke=sersnorm,
                                                                               ellip=ellip,ellip_PA=ellip_PA)
