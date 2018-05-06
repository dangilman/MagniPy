import numpy as np
from Sersic import Sersic
from NFW import NFW
from TNFW import TNFW
from scipy.special import gammainc,gamma

class SersNFW:

    def __init__(self):

        self.sersic = Sersic()
        self.nfw = NFW()
        self.R0_fac = 0.5
        self.Rs_fac = 5

    def params(self,R_ein = None, ellip = None, ellip_theta = None, x=None,
               y = None, r_eff=None, n_sersic=None):

        subparams = {}
        otherkwargs = {}

        otherkwargs['name']='SERSIC_NFW'
        q = 1-ellip
        subparams['q'] = q
        subparams['phi_G'] = (ellip_theta)*np.pi*180**-1

        subparams['Rs'] = r_eff*self.Rs_fac
        subparams['center_x'] = x
        subparams['center_y'] = y

        subparams['r_eff'] = r_eff
        subparams['n_sersic'] = n_sersic

        k_eff,ks_nfw = self.normalizations(Rein=R_ein,re=r_eff,Rs=self.Rs_fac*r_eff,n=n_sersic,R0=self.R0_fac*r_eff,f=self.f)

        subparams['k_eff'] = k_eff
        subparams['theta_Rs'] = 4*ks_nfw*r_eff*self.Rs_fac*(1+np.log(0.5))

        return subparams,otherkwargs

    def kappa(self,x,y,theta_E=None,r_eff=None,n_sersic=None,q=None,separate=False,f=0.333,r0fac=0.5,rsfac=5):

        Rs = rsfac*r_eff
        R0 = r0fac*r_eff

        sersnorm,nfwnorm = self.normalizations(Rein=theta_E,re=r_eff,Rs=Rs,R0=R0,n=n_sersic,f=f)

        nfw_kappa = self.nfw.kappa(x,y,theta_Rs=4*nfwnorm*Rs*(np.log(0.5)+1),Rs=Rs)
        sersic_kappa = self.sersic.kappa(x, y, n_sersic, r_eff, sersnorm, q, center_x=0, center_y=0)

        if separate:
            return nfw_kappa,sersic_kappa
        else:
            return nfw_kappa+sersic_kappa

    def b(self,n):
        return 1.9992*n - 0.3271 + 4*(405*n)**-1

    def normalizations(self,Rein=None,re=None,Rs=None,R0=None,n=None,f=None):

        def F(x):
            return np.log(0.25*x**2)+2*np.arctanh(np.sqrt(1-x**2))*np.sqrt(1-x**2)**-1
        def G(x,n):
            b = self.b(n)
            X = b*x**(n**-1)
            return n*np.exp(b)*b**(-2*n)*gammainc(2*n,X)
        def denom(R_ein,R_eff,Rs,R0,f,n):
            return f*F(R_ein*Rs**-1)*G(R0*R_eff**-1,n)+(1-f)*F(R0*Rs**-1)*G(R_ein*R_eff,n)
        def norm_sersic(R_ein,R_eff,Rs,R0,f,n):
            return 0.5*R_ein**2*Rs**-2*((1-f)*F(R0*Rs**-1))*denom(R_ein,R_eff,Rs,R0,f,n)**-1
        def norm_nfw(R_ein,R_eff,Rs,R0,f,n):
            return 0.5*R_ein**2*R_eff**-2*(f*G(R0*R_eff**-1,n))*denom(R_ein,R_eff,Rs,R0,f,n)**-1


        nfwnorm = norm_nfw(Rein, re, Rs, R0,f,n)
        snorm = norm_sersic(Rein, re, Rs, R0,f,n)

        return snorm,nfwnorm
