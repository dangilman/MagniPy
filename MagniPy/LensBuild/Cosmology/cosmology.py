import numpy as np
from MagniPy.LensBuild.defaults import *
from scipy.integrate import quad
from colossus.halo.concentration import *

class Cosmo:

    M_sun = 1.9891 * 10 ** 30  # solar mass in [kg]

    Mpc = 3.08567758 * 10 ** 22  # Mpc in [m]

    arcsec = 2 * np.pi / 360 / 3600  # arc second in radian

    G = 6.67384 * 10 ** (-11) * Mpc**-3 * M_sun # Gravitational constant [Mpc^3 M_sun^-1 s^-2]

    c = 299792458*Mpc**-1 # speed of light [Mpc s^-1]

    density_to_MsunperMpc = 0.001 * M_sun**-1 * (100**3) * Mpc**3 # convert [g/cm^3] to [solarmasses / Mpc^3]

    def __init__(self,zd = None, zsrc = None, compute=True):

        self.cosmo = default_cosmology

        self.zd, self.zsrc = zd, zsrc

        self.cosmo_set = False

        if compute:

            self.h = self.cosmo.h

            self.epsilon_crit = self.get_epsiloncrit(zd,zsrc)

            self.sigmacrit = self.epsilon_crit*(0.001)**2*self.kpc_per_asec(zd)**2

            self.rhoc_physical = self.cosmo.critical_density0.value * self.density_to_MsunperMpc # [M_sun Mpc^-3]

            self.rhoc = self.rhoc_physical*self.h**-2

            self.D_d,self.D_s,self.D_ds = self.D_A(0,zd),self.D_A(0,zsrc),self.D_A(zd,zsrc)

            self.kpc_convert = self.kpc_per_asec(zd)

            self.d_hubble = self.c*self.Mpc*0.001*(self.h*100)

    def get_rhoc(self):

        return self.cosmo.critical_density0.value * self.density_to_MsunperMpc*self.cosmo.h**-2


    def get_epsiloncrit(self,z1,z2):

        D_ds = self.D_A(z1, z2)
        D_d = self.D_A(0, z1)
        D_s = self.D_A(0, z2)

        epsilon_crit = (self.c**2*(4*np.pi*self.G)**-1)*(D_s*D_ds**-1*D_d**-1)

        return epsilon_crit

    def get_sigmacrit(self):

        return self.get_epsiloncrit(self.zd,self.zsrc)*(0.001)**2*self.kpc_per_asec(self.zd)**2

    def get_sigmacrit_z1z2(self,zlens,zsrc):

        return self.get_epsiloncrit(zlens,zsrc)*(0.001)**2*self.kpc_per_asec(zlens)**2

    def D_A(self,z1,z2):

        return self.cosmo.angular_diameter_distance_z1z2(z1,z2).value

    def D_C(self,z):

        return self.cosmo.comoving_distance(z).value

    def E_z(self,z):

        return self.cosmo.E_z(z)

    def a_z(self, z):
        """
        returns scale factor (a_0 = 1) for given redshift
        """
        return 1. / (1 + z)

    def radian_to_asec(self,x):
        """

        :param x: angle in radians
        :return:
        """
        return x*self.arcsec**-1

    def kpc_per_asec(self,z):
        return self.cosmo.arcsec_per_kpc_proper(z).value**-1

    def D_ratio(self,z1,z2):

        return self.D_A(z1[0],z1[1])*self.D_A(z2[0],z2[1])**-1

    def f_z(self,z):

        I = quad(self.E_z,0,z)[0]
        I2 = quad(self.E_z,0,self.zd)[0]

        return (1+z)**-2*self.E_z(z)**-1*(I**2*I2**-2)

    def T_xy(self, z_observer, z_source):
        """
        transverse comoving distance in units of Mpc
        """
        T_xy = self.cosmo.comoving_transverse_distance(z_source).value - self.cosmo.comoving_transverse_distance(z_observer).value

        return T_xy

    def physical_distance_z1z2(self,z1,z2):

        a1 = (1+z1)**-1
        a2 = (1+z2)**-1

        d1 = self.T_xy(0,z1)
        d2 = self.T_xy(0,z2)

        return d2 - d1

    def D_xy(self, z_observer, z_source):
        """
        angular diamter distance in units of Mpc
        :param z_observer: observer
        :param z_source: source
        :return:
        """
        a_S = self.a_z(z_source)
        D_xy = (self.cosmo.comoving_transverse_distance(z_source) - self.cosmo.comoving_transverse_distance(z_observer))*a_S
        return D_xy.value

    def rho_crit(self,z):
        return self.cosmo.critical_density(z).value*self.density_to_MsunperMpc

    def rho_matter_crit(self,z):
        return self.rho_crit(z)*self.cosmo.Om(z)

    def vdis_to_Rein(self,zd,zsrc,vdis):

        return 4 * np.pi * (vdis * (0.001 * self.c * self.Mpc) ** -1) ** 2 * \
               self.D_A(zd, zsrc) * self.D_A(0,zsrc) ** -1 * self.arcsec ** -1

    def vdis_to_Reinkpc(self,zd,zsrc,vdis):

        return self.kpc_per_asec(zd)*self.vdis_to_Rein(zd,zsrc,vdis)

    def beta(self,z,zmain,zsrc):

        D_12 = self.D_A(zmain, z)
        D_os = self.D_A(0, zsrc)
        D_1s = self.D_A(zmain, zsrc)
        D_o2 = self.D_A(0, z)

        return D_12 * D_os * (D_o2 * D_1s) ** -1

    def NFW(self,M_200,c,z):

        h = self.h

        rho = self.get_rhoc()

        r200 = (3*M_200*h*(4*np.pi*rho*200)**-1)**(1./3.) * h * self.a_z(z)

        if c is None:
            c = self.NFW_concentration(M_200,z=z)

        rho0_c = 200./3*rho*c**3/(np.log(1+c)-c/(1+c))

        rho0 = rho0_c / h ** 2 / self.a_z(z) ** 3

        rho0_kpc = rho0*(1000)**-3
        r200_kpc = r200 * 1000

        Rs = r200_kpc / c

        return rho0_kpc, Rs, r200_kpc

    def NFW_concentration(self,M,model='bullock01',mdef='200c',logmhm=0,z=None,
                           g1=60,concentration_turnover=True):

        # WDM relation adopted from Ludlow et al
        # scatter adopted from Ludlow et al CDM

        g2 = concentration_power

        def beta(z_val):
            return 0.026*z_val - 0.04

        if self.cosmo_set is False:
            self._set_cosmo()

        if z is None:
            z = self.zd

        c_cdm = concentration(M*self.h,mdef=mdef,model=model,z=z)

        if concentration_turnover is False:
            # scatter from Dutton, maccio et al 2014
            return np.random.lognormal(np.log(c_cdm),0.13)

        if logmhm == 0:
            # scatter from Dutton, maccio et al 2014
            return np.random.lognormal(np.log(c_cdm),0.13)

        else:

            mhm = 10**logmhm
            factor = (1+g1*mhm*M**-1)**g2
            c_wdm = c_cdm*factor**-1
            c_wdm *= (1+z)**beta(z)
            # scatter from Dutton, maccio et al 2014
            return np.random.lognormal(np.log(c_wdm),0.13)

class ParticleMasses:

    def __init__(self,h=0.7):

        pass
        # from lyman alpha: 1,2,2.5,3.3 keV excluded at 9s, 4s, 3s, 2s from Viel et al

    def hm_mass(self,m,h=0.7):
        # half mode mass corresponding to thermal relic of mass m (kev)
        # calibrated to Schneider 2012

        mass = 1.07e+10 * h * m**-3.33
        return mass

    def thermal_to_sterile(self,m,w=0.1225):

        return 3.9*m**1.294

    def hm_to_thermal(self,m,as_string=False,h=0.7):

        # particle mass (keV) corresponding to half-mode mass m (solar masses)

        #masses = (np.array(m) * (1.07e+10 * h**-1) ** -1) ** (-1 * 3.33 ** -1)
        masses = (np.array(m) * (1.07e+10 * h) ** -1) ** (-1 * 3.33 ** -1)

        if as_string:
            massstring = []
            for m in masses:
                massstring.append(str(m))
            return massstring
        else:
            return masses

    def wave_alpha(self,m_kev,omega_WDM=0.25,h=0.7):

        return 0.049*(m_kev)**-1.11*(omega_WDM*4)**0.11*(h*0.7**-1)**1.22*h
