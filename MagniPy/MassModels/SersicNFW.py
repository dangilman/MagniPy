import numpy as np
from Sersic import Sersic
from NFW import NFW
from scipy.special import gammainc,gamma

class SersicNFW:

    def __init__(self,R0_fac=0.5):

        self.sersic = Sersic()
        self.nfw = NFW()
        self.R0_fac = R0_fac

    def params(self,R_ein = None, ellip = None, ellip_theta = None, x=None,
               y = None, Rs=None, n_sersic=None,reff_thetaE_ratio=None,f=None,**kwargs):

        subparams = {}
        otherkwargs = {}

        otherkwargs['name']='SERSIC_NFW'
        q = 1-ellip
        subparams['q'] = q
        subparams['phi_G'] = (ellip_theta)*np.pi*180**-1

        subparams['Rs'] = Rs
        subparams['center_x'] = x
        subparams['center_y'] = y

        subparams['R_sersic'] = reff_thetaE_ratio*R_ein
        subparams['n_sersic'] = n_sersic

        k_eff,ks_nfw = self.normalizations(Rein=R_ein,re=subparams['R_sersic'],Rs=Rs,
                                           n=n_sersic,R0=self.R0_fac*subparams['R_sersic'],f=f)

        subparams['k_eff'] = k_eff

        subparams['theta_Rs'] = 4*ks_nfw*subparams['Rs']*(1+np.log(0.5))

        return subparams,otherkwargs

    def kappa(self,x,y,theta_E=None,Rs=None,reff_thetaE_ratio=1,n_sersic=None,q=None,f=None,separate=False):

        r_eff = reff_thetaE_ratio*theta_E

        sersnorm,nfwnorm = self.normalizations(Rein=theta_E,re=r_eff,Rs=Rs,R0=self.R0_fac*r_eff,n=n_sersic,f=f)

        nfw_kappa = self.nfw.kappa(x,y,theta_Rs=4*nfwnorm*Rs*(np.log(0.5)+1),Rs=Rs)
        sersic_kappa = self.sersic.kappa(x, y, n_sersic, r_eff, sersnorm, q, center_x=0, center_y=0)

        if separate:
            return nfw_kappa,sersic_kappa
        else:
            return nfw_kappa+sersic_kappa

    def b(self,n):
        return 1.9992*n - 0.3271 + 4*(405*n)**-1

    def G(self,x):
        return np.log(0.25 * x ** 2) + 2 * np.arctanh(np.sqrt(1 - x ** 2)) * np.sqrt(1 - x ** 2) ** -1

    def F(self,x,n):
        b = self.b(n)
        X = b * x ** (n ** -1)
        return n * np.exp(b) * b ** (-2 * n) * (gamma(2*n) - gammainc(2 * n, X))

    def denom(self,theta_E,re,Rs,R0,n,f):
        return self.F(theta_E * re ** -1,n) + (f * (1 - f) ** -1) * self.G(theta_E * Rs ** -1) * self.F(R0 * Rs ** -1,n) * self.G(
            R0 * Rs ** -1) ** -1

    def norm_nfw(self,theta_E,re,Rs,R0,n,f):

        return 0.5 * f * (1 - f) ** -1 * (theta_E * Rs ** -1) ** 2 * self.F(R0 * Rs ** -1,n) * self.G(
            R0 * Rs ** -1) ** -1 * self.denom(theta_E,re,Rs,R0,n,f) ** -1

    def norm_sersic(self,theta_E,re,Rs,R0,n,f):
        return 0.5 * (theta_E * re ** -1) ** 2 * self.denom(theta_E,re,Rs,R0,n,f) ** -1

    def normalizations(self,Rein=None,re=None,Rs=None,R0=None,n=None,f=None):

        return self.norm_nfw(Rein,re,Rs,R0,n,f),self.norm_sersic(Rein,re,Rs,R0,n,f)



