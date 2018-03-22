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
    def __init__(self, M_parent=None, parent_locations=None, N0=0.21, alpha_secondary=0.8, xi=0.9,
                 log_mL = None,logmhm=None,redshift=None,zsrc=None):

        NFW_calculate = NFW(redshift,zsrc)

        localized_2d = Localized_uniform(cosmology=NFW_calculate.cosmology)

        self.power_laws = []

        self.parent_locations = parent_locations

        self.redshift = redshift

        self.locations = []

        for i,M in enumerate(M_parent):

            M *= xi

            normalization = N0*alpha_secondary**alpha_secondary*M**alpha_secondary

            locations = deepcopy(localized_2d)

            locations.set_xy(parent_locations[0][i],parent_locations[1][i])

            c = NFW_calculate.nfw_concentration(M*xi**-1,logmhm=logmhm)

            rmax2d = NFW_calculate.nfwParam_physical(M * xi ** -1, c)[2] * NFW_calculate.cosmology.D_A(0,redshift) ** -1

            locations.set_rmax2d(rmax2d)

            self.locations.append(locations)

            self.power_laws.append(Plaw(normalization=normalization,log_mL=log_mL,log_mH=np.log10(M),logmhm=logmhm,plaw_index=-1.8))

    def draw(self):

        for i,plaw in enumerate(self.power_laws):
            newhalos = plaw.draw()

            new_halo_positions = self.locations[i].draw(N=len(newhalos),z=self.redshift)

            if i==0:
                halos = np.array(newhalos)
                halo_positions = np.array(new_halo_positions)
            else:
                halos = np.append(halos,newhalos)
                halo_positions = np.append(halo_positions,new_halo_positions)

        return (halos,halo_positions)

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

    def get_Nsub(self,norm=float,scrit=None,area=None):

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
