import numpy as np
from cosmology import Cosmo

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


            a=input('continue')

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

class RedshiftNormalization(Cosmo):

    def __init__(self,zd,zsrc):

        Cosmo.__init__(self, zd=zd, zsrc=zsrc)

    def compute_mass(self,z=float,dz=float,area=float):

        mass = self.DMmass_inplane(angualr_area=area,zplane=z,dz=dz)

        return mass
