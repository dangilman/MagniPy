import numpy as np
from cosmology import Cosmo
from scipy.special import j1
from scipy.integrate import quad

class Plaw:

    def __init__(self,norm=float,logmL=float,logmH=float,area=classmethod,scrit=float,logmbreak=0,
                 alpha=1.9,gamma=1.3,**norm_kwargs):

        self.alpha,self.gamma = alpha,gamma

        self.mL,self.mH = 10**logmL,10**logmH

        if logmbreak == 0:

            self.mbreak = 0

        else:

            self.mbreak = 10**logmbreak

        self.A0 = self.get_A0(norm=norm,scrit=scrit,area=area,**norm_kwargs)

        self.Nsub,self.Nmean = self.get_Nsub(self.A0,scrit,area)

    def draw(self):
        return self.sample_CDF(self.Nsub)

    def get_Nsub(self,A0=float,scrit=None,area=None):

        Nwdm = A0*self.moment(0,self.mL,self.mH)

        return np.random.poisson(Nwdm),Nwdm

    def get_A0(self,norm,scrit,area,**kwargs):
        kwargs = kwargs['norm_kwargs']
        if isinstance(norm,float):

            A0 = norm*scrit*area*self.moment(1,self.mL,self.mH)**-1

        else:

            mass = norm.compute_mass(z=kwargs['z'],dz = kwargs['dz'], area= kwargs['area'])
            A0 = mass*self.moment(1,kwargs['mlow_norm'],kwargs['mhigh_norm'])**-1


        return A0

    def moment(self,n,m1,m2):
        return (n + 1 - self.alpha) ** -1 * (m2 ** (n + 1 - self.alpha) - m1 ** (n + 1 - self.alpha))

    def sample_CDF(self, Nsamples):

        if self.alpha == 2:
            raise ValueError('alpha cannot equal 2')

        x = np.random.rand(Nsamples)
        X = (x * (self.mH ** (1 - self.alpha) - self.mL ** (1 - self.alpha)) + self.mL ** (1 - self.alpha)) ** ((1 - self.alpha) ** -1)

        if self.mbreak == 0:
            return np.array(X)
        else:
            mass = []

            for i in range(0, Nsamples):

                u = np.random.rand()
                if u < (1 + self.mbreak * X[i] ** -1) ** (-self.gamma):
                    mass.append(X[i])

        return np.array(mass)


class ShethTorman:

    def __init__(self,cosmology=None):

        if cosmology is None:
            self.cosmology = Cosmo(zd=0.5,zsrc=1.5)
        else:
            self.cosmology = cosmology

    def linear_transfer(self,k,z,m_hm=10**8):

        # a simple fit to the power spectrum; neglects small baryon acoustic oscillations

        q = k*self.cosmology.h**-1*(self.cosmology.cosmo.Om(z)*self.cosmology.cosmo.h)**-1

        L = np.log(2*np.e +1.8*q)
        C = 14.2+(731*(1+62.5*q)**-1)

        T = L*(L+C*q**2)**-1

        return T

    def P_lin(self,k,z,n=1,P0=1):

        return P0*self.linear_transfer(k,z)**2*k**n

    def P_nl(self,k,z, k_sigma=1, omega_m=1,C=1, neff = 1):

        # Non-linear correction to the linear power spectrum computed in Takahashi1 et al 2012

        Plin = self.P_lin(k,z)

        def f(x):
            return 0.25 * x + x ** 2 * 8 ** -1

        def f1(x):
            return x ** (-0.0307)

        def f2(x):
            return x ** -0.0585

        def f3(x):
            return x ** 0.0743

        an = 10 ** (1.5222 + 2.8553 * neff + 2.3706 * neff ** 2 + 0.9903 * neff ** 3 + 0.225 * neff ** 4 - 0.6038 * C)
        bn = 10 ** (-.5642 + 0.5864 * neff + 0.5716 * neff ** 2 - 1.5474 * C)
        cn = 10 ** (0.3698 + 2.0404 * neff + 0.8161 * neff ** 2 + 0.5869 * C)
        gamman = 0.1971 - 0.0843 * neff + 0.8460 * C
        alphan = np.absolute(6.0835 + 1.3373 * neff - 0.1959 * neff ** 2 - 5.5274 * C)
        betan = 2.0379 - 0.7354 * neff + 0.3157 * neff ** 2 + 1.249 * neff ** 3 + 0.398 * neff ** 4 - 0.1682 * C
        mun = 0
        nun = 10 ** (5.2105 + 3.6902 * neff)

        y = k * k_sigma ** -1
        Plin = 1
        deltaPlin = k ** 3 * Plin * (2 * np.pi ** 2) ** -1

        halo2 = deltaPlin * ((1 + deltaPlin) ** betan * (1 + alphan * deltaPlin) ** -1) * np.exp(-f(y))

        halo1 = (1 + mun * y ** -1 + nun * y ** -2) ** -1 * (an * y ** (3 * f1(omega_m))) * (1 + bn * y ** (
        f2(omega_m)) + (cn * f3(omega_m) * y) ** (3 - gamman)) ** -1

        return 2 * np.pi ** 2 * k ** -3 * (halo1 + halo2)

    def growth_function(self,z):

        return (1+z)**-1

    def _normalize_simga_M(self,R,z,lambda_min=1e-3,lambda_max=10):

        kmin = 2 * np.pi * self.cosmology.h * lambda_max ** -1
        kmax = 2 * np.pi * self.cosmology.h * lambda_min ** -1

        def _integrand(k, R, z):
            return k ** -1 * self.P_lin(k, z) * k ** 3 * (2 * np.pi ** 2) ** -1 * j1(k * R) ** 2

        if isinstance(R, float) or isinstance(R, int):
            return self.growth_function(z) * quad(_integrand, kmin, kmax, args=(R, z))[0]
        else:
            vals = []
            for val in R:
                vals.append(self.growth_function(z) * quad(_integrand, kmin, kmax, args=(val, z))[0])
            return np.array(vals)

    def _integrand(self, k, M, z):
        R = (3 * M * (4 * np.pi * self.cosmology.rho_matter_crit(z)) ** -1) ** (1 * 3 ** -1)
        x = k * R
        return k ** -1 * self.P_lin(k, z) * k ** 3 * (2 * np.pi ** 2) ** -1 * (3 * j1(x) * x ** -1) ** 2

    def sigma_M(self,M,z,m_min=10**6,m_max=10**10):
        """

        :param M: mass in solar masses
        :param z: redshift
        :param lambda_min: distance in Mpc
        :param lambda_max: distance in Mpc
        :return:
        """

        lambda_min = (3*m_min*(4*np.pi)**-1)**0.3333
        lambda_max = (3*m_max*(4 * np.pi) ** -1) ** 0.3333

        kmin = 2*np.pi*self.cosmology.h*lambda_max**-1
        kmax = 2*np.pi*self.cosmology.h*lambda_min**-1

        if isinstance(M,float) or isinstance(M,int):
            return self.growth_function(z)*quad(self._integrand,kmin,kmax,args=(M,z))[0]
        else:
            vals = []
            for val in M:
                vals.append(self.growth_function(z)*quad(self._integrand,kmin,kmax,args=(val,z))[0])
            return np.array(vals)


    def _nu(self,M,z,delta_crit = 1.69):

        return delta_crit**2*self.sigma_M(M,z)**-2

    def f_nu(self,M,z,A=0.333,a=0.794,p=0.247):

        nu = self._nu(M,z)

        return A*nu**-1*(1+(a*nu)**-p)*(a*nu*(2*np.pi)**-1)**.5*np.exp(-0.5*a*nu)


if False:
    s = ShethTorman()
    x = np.linspace(10**6,10**10,200)
    y = s._nu(x,0.5)*s.f_nu(x,0.5)
    print s.f_nu(x,0.5)
    import matplotlib.pyplot as plt
    plt.plot(x,y)

    plt.show()












