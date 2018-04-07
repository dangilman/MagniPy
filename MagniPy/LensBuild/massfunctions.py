import numpy as np
from MagniPy.LensBuild.Cosmology.cosmology import Cosmo
from MagniPy.MassModels.NFW import NFW
from spatial_distribution import Localized_uniform
from copy import deepcopy

class Plaw_secondary:
    """
    computes the subhalos of halos according to Giocoli 2008? equation 2
    dNs / dM = (N0 / m) * (m / (const*M))^const exp(-6 (m / (const*M))^3)

    implement the exponential cutoff by rendering up to some factor xi*M.

    """
    def __init__(self, M_parent=None, x_locations=None, y_locations=None, N0=0.1, alpha_secondary=0.8, xi=0.9,
                 log_mL = None,logmhm=None,cosmo_at_zlens=None):

        NFW_calculate = NFW(cosmology=cosmo_at_zlens)

        self.power_laws = []

        self.redshift = cosmo_at_zlens.zd

        self.locations = []

        for i,M in enumerate(M_parent):

            if M<10**log_mL:
                continue
            M *= xi

            normalization = N0*alpha_secondary**alpha_secondary*M**alpha_secondary

            c = NFW_calculate.nfw_concentration(M*xi**-1,logmhm=logmhm)

            # rmax2d in Mpc
            rmax2d = NFW_calculate.nfwParam_physical(M * xi ** -1, c)[2] * NFW_calculate.cosmology.D_A(0,self.redshift) ** -1
            rmax2d *= 1000*cosmo_at_zlens.kpc_per_asec(self.redshift)

            locations = Localized_uniform(rmax2d=rmax2d, xlocations=x_locations[i],ylocations=y_locations[i],cosmology=cosmo_at_zlens)

            self.locations.append(locations)

            self.power_laws.append(Plaw(normalization=normalization,log_mL=log_mL,log_mH=np.log10(M*xi**-1),logmhm=logmhm,plaw_index=-1.8))

    def draw(self):

        for i,plaw in enumerate(self.power_laws):

            newhalos = plaw.draw()

            newx,newy = self.locations[i].draw(N=int(len(newhalos)),z=self.redshift)

            if i==0:
                halos = np.array(newhalos)
                halox,haloy = np.array(newx),np.array(newy)
            else:
                halos = np.append(halos,newhalos)
                halox = np.append(halox,newx)
                haloy = np.append(haloy,newy)

        return halos,halox,haloy

class Plaw:

    def __init__(self, normalization=float, log_mL=None, log_mH=None, logmhm=0, plaw_index=-1.9, turnover_index=1.3,
                 **kwargs):

        self.plaw_index, self.turnover_index = plaw_index, turnover_index

        self.mL,self.mH = 10**log_mL,10**log_mH

        if logmhm == 0:

            self.mbreak = 0

        else:

            self.mbreak = 10**logmhm

        self.norm = normalization

        self.Nhalos,self.Nhalos_mean = self.get_Nsub(normalization)

    def draw(self):

        return self.sample_CDF(np.random.poisson(self.Nhalos_mean))

    def get_Nsub(self,norm=float):

        N = norm*self.moment(0,self.mL,self.mH)

        return np.random.poisson(N),N

    def moment(self,n,m1,m2):
        return (n + 1 + self.plaw_index) ** -1 * (m2 ** (n + 1 + self.plaw_index) - m1 ** (n + 1 + self.plaw_index))

    def sample_CDF(self, Nsamples):

        if self.plaw_index == 2:
            raise ValueError('index cannot equal 2')

        x = np.random.rand(Nsamples)
        X = (x * (self.mH ** (1 + self.plaw_index) - self.mL ** (1 + self.plaw_index)) + self.mL ** (1 + self.plaw_index)) ** ((1 + self.plaw_index) ** -1)

        if self.mbreak == 0:
            return np.array(X)
        else:
            mass = []

            for i in range(0, Nsamples):

                u = np.random.rand()
                if u < (1 + self.mbreak * X[i] ** -1) ** (-self.turnover_index):
                    mass.append(X[i])

        return np.array(mass)

class Delta:

    def __init__(self,N_per_image,logmass):

        self.norm = N_per_image
        self.mass = 10**logmass

    def draw(self):

        return np.ones(self.norm)*self.mass
