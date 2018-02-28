import numpy as np
from MagniPy.LensBuild.defaults import *
from scipy.integrate import quad


class Cosmo:

    M_sun = 1.9891 * 10 ** 30  # solar mass in [kg]

    Mpc = 3.08567758 * 10 ** 22  # Mpc in [m]

    arcsec = 2 * np.pi / 360 / 3600  # arc second in radian

    G = 6.67384 * 10 ** (-11) * Mpc**-3 * M_sun # Gravitational constant [Mpc^3 M_sun^-1 s^-2]

    c = 299792458*Mpc**-1 # speed of light [Mpc s^-1]

    density_to_MsunperMpc = 0.001 * M_sun**-1 * (100**3) * Mpc**3 # convert [g/cm^3] to [solarmasses / Mpc^3]

    def __init__(self,zd = None, zsrc = None, compute=True):

        self.cosmo = default_cosmology

        if compute:

            self.zd,self.zsrc = zd,zsrc

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

    def get_sigmacrit(self):

        return self.get_epsiloncrit(self.zd,self.zsrc)*(0.001)**2*self.kpc_per_asec(self.zd)**2

    def D_A(self,z1,z2):

        return self.cosmo.angular_diameter_distance_z1z2(z1,z2).value

    def D_C(self,z):

        return self.cosmo.comoving_transverse_distance(z).value
    def E_z(self,z):

        return np.sqrt(self.cosmo.Om(z) + self.cosmo.Ode(z))

    def a_z(self, z):
        """
        returns scale factor (a_0 = 1) for given redshift
        """
        return 1. / (1 + z)

    def kpc_per_asec(self,z):
        return self.cosmo.arcsec_per_kpc_proper(z).value**-1

    def D_ratio(self,z1,z2):

        return self.D_A(z1[0],z1[1])*self.D_A(z2[0],z2[1])**-1

    def get_epsiloncrit(self,z1,z2):

        D_ds = self.D_A(z1, z2)
        D_d = self.D_A(0, z1)
        D_s = self.D_A(0, z2)

        epsilon_crit = (self.c**2*(4*np.pi*self.G)**-1)*(D_s*D_ds**-1*D_d**-1)

        return epsilon_crit

    def E_z(self,z):
        return (self.cosmo.Om(z)*(1+z)**-3 + self.cosmo.Ode(z))**.5

    def f_z(self,z):

        I = quad(self.E_z,0,z)[0]
        I2 = quad(self.E_z,0,self.zd)[0]

        return (1+z)**-2*self.E_z(z)**-1*(I**2*I2**-2)

    def T_xy(self, z_observer, z_source):
        """
        transverse comoving distance in units of Mpc
        """
        T_xy = self.cosmo.comoving_transverse_distance(z_source).value - \
               self.cosmo.comoving_transverse_distance(z_observer).value
        return T_xy

    def D_co(self,z_observer, z):
        """

        :param z_observer: initial z
        :param z: target z
        :return: comoving distance between redshift z_observer and z
        """
        return self.cosmo._comoving_distance_z1z2(z_observer,z).value

    def D_xy(self, z_observer, z_source):
        """

        :param z_observer: observer
        :param z_source: source
        :return: angular diamter distance in units of Mpc
        """
        a_S = self.a_z(z_source)
        D_xy = (self.cosmo.comoving_transverse_distance(z_source) - self.cosmo.comoving_transverse_distance(z_observer))*a_S
        return D_xy.value

    def rho_crit(self,z):
        return self.cosmo.critical_density(z).value*self.density_to_MsunperMpc

    def rho_matter_crit(self,z):
        return self.rho_crit(z)*self.cosmo.Om(z)

    def _physical2angle(self,phys,z):

        return phys*self.D_A(0,z)**-1

    def _physical2comoving(self,phys,z):

        return phys*(1+z)

    def _comoving2physical(self,co,z):

        return co*(1+z)**-1


class ParticleMasses:

    def __init__(self,h=0.7):

        pass
        # from lyman alpha: 1,2,2.5,3.3 keV excluded at 9s, 4s, 3s, 2s from Viel et al

    def hm_mass(self,m,h=0.7):
        # half mode mass corresponding to thermal relic of mass m (kev)
        # calibrated to Schneider 2012

        mass = 1.07e+10 * h**-1 * m**-3.33
        return mass

    def thermal_to_sterile(self,m,w=0.1225):

        return 4.43*(m)**1.333*(w*0.1225**-1)**-0.333

    def sterile_to_thermal(self,m,w=0.1225):

        return (4.43**-1*(w*0.1225**-1)**0.333)**0.75

    def hm_to_thermal(self,m,as_string=False,h=0.7):

        # particle mass (keV) corresponding to half-mode mass m (solar masses)

        masses = (np.array(m) * (1.07e+10 * h**-1) ** -1) ** (-1 * 3.33 ** -1)

        if as_string:
            massstring = []
            for m in masses:
                massstring.append(str(m))
            return massstring
        else:
            return masses

    def wave_alpha(self,m_kev,omega_WDM=0.25,h=0.7):

        return 0.049*(m_kev)**-1.11*(omega_WDM*4)**0.11*(h*0.7**-1)**1.22*h**-1
