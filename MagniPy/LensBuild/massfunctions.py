import numpy as np
from MagniPy.LensBuild.Cosmology.cosmology import Cosmo
from MagniPy.MassModels.NFW import NFW
from MagniPy.LensBuild.spatial_distribution import NFW_3D
from scipy.integrate import quad

class Plaw_secondary:

    """
    computes the subhalos of halos according to Giocoli 2008? equation 2
    dNs / dM = (N0 / m) * (m / (const*M))^const exp(-6 (m / (const*M))^3)

    implement the exponential cutoff by rendering up to some factor xi*M.

    """

    def __init__(self, M_parent=None, parent_r2d = None, x_position=None, y_position=None, N0=0.21, alpha_secondary=0.8,
                 log_mL = None,logmhm=None,cosmo_at_zlens=None,parent_r3d=None,concentration_func=None,c_turnover=True):

        NFW_calculate = NFW(cosmology=cosmo_at_zlens)

        self.power_laws = []

        self.redshift = cosmo_at_zlens.zd

        self.locations = []

        self.alpha = alpha_secondary+1

        if isinstance(M_parent,float) or isinstance(M_parent,int):
            M_parent = [M_parent]
        if isinstance(x_position,float) or isinstance(x_position,int):
            x_position = [x_position]
        if isinstance(y_position,float) or isinstance(y_position,int):
            y_position = [y_position]
        if isinstance(parent_r2d,float) or isinstance(parent_r2d,int):
            parent_r2d = [parent_r2d]

        self.parent_masses = M_parent
        self.parent_r2d = parent_r2d
        self.parent_r3d = parent_r3d
        self.parent_rmax = []

        for i,M in enumerate(M_parent):

            if M<=10**log_mL:
                normalization=0
            else:
                normalization = N0

            c = concentration_func(M,logmhm=logmhm,z=self.redshift,concentration_turnover=c_turnover)

            # rmax2d in Mpc
            _,Rs_mpc,r200_mpc = NFW_calculate.nfwParam_physical(M, c)

            Rs_kpc,r200_kpc = Rs_mpc*1000,r200_mpc*1000

            kpc_per_asec = cosmo_at_zlens.kpc_per_asec(self.redshift)

            Rs_asec = Rs_kpc*kpc_per_asec**-1
            r200_asec = r200_kpc*kpc_per_asec**-1

            self.parent_rmax.append(r200_asec)

            locations = NFW_3D(rmax2d=3*Rs_asec,rmax3d=3*Rs_asec, Rs=Rs_asec, xoffset=x_position[i],
                               yoffset=y_position[i], tidal_core=False, cosmology=cosmo_at_zlens)

            self.locations.append(locations)

            self.power_laws.append(Plaw_subhalos(N0=normalization,alpha=alpha_secondary,
                                                 log_mL=log_mL,M_parent=M,logmhm=logmhm))

    def draw(self):

        for i,plaw in enumerate(self.power_laws):

            init = plaw.draw()

            newhalos = []

            for halo in init:

                if np.exp(-6.283*(halo*(0.8*self.parent_masses[i])**-1)**3) > np.random.random():
                    newhalos.append(halo)

            newx,newy,R2d,R3d = self.locations[i].draw(N=int(len(newhalos)))

            while self.parent_masses[i] - np.sum(newhalos) <= 0:
                newhalos = np.delete(newhalos,np.argmin(newhalos))

            parent_mass = self.parent_masses[i] - np.sum(newhalos)

            parent_x, parent_y = self.locations[i].xoffset, self.locations[i].yoffset

            parent_r2d = self.parent_r2d[i]
            parent_r3d = self.parent_r3d[i]

            if i==0:

                halor2d = np.append(parent_r2d,R2d)

                halor3d = np.append(parent_r3d,R3d)

                halos = np.append(parent_mass,newhalos)
                halox,haloy = np.append(parent_x,np.array(newx)),np.append(parent_y,np.array(newy))

            else:

                newR3d = np.append(parent_r3d,R3d)
                newR2d = np.append(parent_r2d,R2d)
                new_xloc = np.append(parent_x, newx)
                new_yloc = np.append(parent_y, newy)
                newobjects = np.append(parent_mass, newhalos)

                halor2d = np.append(halor2d, newR2d)
                halor3d = np.append(halor3d,newR3d)

                halos = np.append(halos,newobjects)
                halox = np.append(halox,new_xloc)
                haloy = np.append(haloy,new_yloc)

        return halos,halox,haloy,halor2d,halor3d

class Plaw_subhalos:

    def __init__(self, N0=0.21, alpha=0.8, log_mL=None, M_parent=None,logmhm=0):

        self.N0 = N0
        self.alpha = alpha

        self.N_halos_mean = self.N_mean(M_parent, 10 ** log_mL)
        self.plaw_index = -(alpha+1)
        self.mL = 10**log_mL
        self.mH = M_parent

        if logmhm == 0:

            self.mbreak = 0

        else:

            self.mbreak = 10**logmhm

    def integrand_N(self,x, M):

        return self.N0 * x ** -1 * (x * (self.alpha * M) ** -1) ** -self.alpha * \
               np.exp(-6.283 * (x * (self.alpha * M) ** -1) ** 3)

    def integrand_M(self, x, M):

        return x*self.integrand_N(x,M)

    def N_mean(self, M, mL):

        return quad(self.integrand_N, a=mL, b=M, args=(M))[0]

    def M_mean(self, M, mL):

        return quad(self.integrand_M, a=mL, b=M, args=(M))[0]

    def draw(self):

        return self.sample_CDF(np.random.poisson(self.N_halos_mean))

    def sample_CDF(self, Nsamples):

        if self.plaw_index == 2:
            raise ValueError('index cannot equal 2')

        x = np.random.rand(Nsamples)
        _X = (x * (self.mH ** (1 + self.plaw_index) - self.mL ** (1 + self.plaw_index)) + self.mL ** (1 + self.plaw_index)) ** ((1 + self.plaw_index) ** -1)
        X = []

        for halo in _X:
            if np.exp(-6.283 * (halo * (0.8 * self.mH) ** -1) ** 3) > np.random.rand():
                X.append(halo)

        if self.mbreak == 0:
            return np.array(X)
        else:
            mass = []

            for i in range(0, Nsamples):

                u = np.random.rand()
                if u < (1 + self.mbreak * X[i] ** -1) ** (-self.turnover_index):
                    mass.append(X[i])

        return np.array(mass)

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

    def __init__(self, N, logmass):

        self.norm = N
        self.mass = 10**logmass

    def draw(self):

        return np.ones(self.norm)*self.mass

def normalize_M200(fsub,M200,c,rmax2d,R200,mH,mL,rmin,plaw_index):

    _m = M200*mH**-1

    _r = rmax2d*R200**-1

    xmin = rmin*R200**-1

    f_xmin = np.log(2*(xmin*(1+xmin))**-1) + 0.5*(1-xmin)*(1+xmin)**-1

    beta = mL*mH**-1

    prefactor = (mH*mH**plaw_index)**-1

    fsub_volume = 1.5*_r**2*(1-xmin)

    fsub_effective = fsub_volume*fsub

    return prefactor*0.5*fsub_effective*_r*_m**2*c**2*f_xmin*(2+plaw_index)**-1*(1-beta**(2+plaw_index))
