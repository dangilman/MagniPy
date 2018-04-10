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

    def __init__(self, M_parent=None, parent_r2d = None, x_locations=None, y_locations=None, N0=0.21, alpha_secondary=-0.8, xi=1,
                 log_mL = None,logmhm=None,cosmo_at_zlens=None):

        NFW_calculate = NFW(cosmology=cosmo_at_zlens)

        self.power_laws = []

        self.redshift = cosmo_at_zlens.zd

        self.locations = []

        self.alpha = alpha_secondary+1
        self.parent_masses = M_parent
        self.parent_r2d = parent_r2d

        for i,M in enumerate(M_parent):

            M *= xi

            if M<10**log_mL:
                normalization=0
            else:
                normalization = N0*np.absolute(alpha_secondary)*M

            c = NFW_calculate.nfw_concentration(M*xi**-1,logmhm=logmhm)

            # rmax2d in Mpc
            rmax2d = NFW_calculate.nfwParam_physical(M * xi ** -1, c)[2] * NFW_calculate.cosmology.D_A(0,self.redshift) ** -1
            rmax2d *= 1000*cosmo_at_zlens.kpc_per_asec(self.redshift)

            locations = Localized_uniform(rmax2d=rmax2d, xlocations=x_locations[i],ylocations=y_locations[i],cosmology=cosmo_at_zlens)

            self.locations.append(locations)

            self.power_laws.append(Plaw(normalization=normalization,log_mL=log_mL,log_mH=np.log10(M*xi**-1),logmhm=logmhm,plaw_index=alpha_secondary-1))

    def draw(self):

        for i,plaw in enumerate(self.power_laws):

            new_halos = plaw.draw()

            newhalos = []

            for halo in new_halos:

                if np.exp(-6.283*(halo*(0.8*self.parent_masses[i])**-1)**3) > np.random.random():
                    newhalos.append(halo)

            newx,newy,R2d = self.locations[i].draw(N=int(len(newhalos)),z=self.redshift)

            parent_mass = self.parent_masses[i] - np.sum(newhalos)

            parent_x, parent_y = self.locations[i].xlocations, self.locations[i].ylocations
            parent_r2d = self.parent_r2d[i]
            zvals = np.random.uniform(-self.locations[i].rmax2d,self.locations[i].rmax2d,len(newhalos))

            if i==0:
                halor2d = np.append(parent_r2d,np.sqrt(R2d**2+zvals**2))
                halos = np.append(parent_mass,newhalos)
                halox,haloy = np.append(parent_x,np.array(newx)),np.append(parent_y,np.array(newy))

            else:

                newR = np.append(parent_r2d,np.sqrt(R2d**2+zvals**2))
                halor2d = np.append(halor2d, newR)

                newobjects = np.append(parent_mass,newhalos)
                new_xloc = np.append(parent_x,newx)
                new_yloc = np.append(parent_y,newy)

                halos = np.append(halos,newobjects)
                halox = np.append(halox,new_xloc)
                haloy = np.append(haloy,new_yloc)

        return halos,halox,haloy,halor2d

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
