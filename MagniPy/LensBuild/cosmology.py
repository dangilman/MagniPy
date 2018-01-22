import numpy as np
import astropy.cosmology as C

class Cosmo:

    M_sun = 1.9891 * 10 ** 30  # solar mass in [kg]

    Mpc = 3.08567758 * 10 ** 22  # Mpc in [m]

    arcsec = 2 * np.pi / 360 / 3600  # arc second in radian

    G = 6.67384 * 10 ** (-11) * Mpc**-3 * M_sun # Gravitational constant [Mpc^3 M_sun^-1 s^-2]
    c = 299792458*Mpc**-1 # speed of light [Mpc s^-1]

    def __init__(self,cosmology = 'FlatLambdaCDM',zd=0.5,zsrc = 1.5):

        if cosmology == 'FlatLambdaCDM':

            self.zd,self.zsrc = zd,zsrc
            self.cosmo = C.FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.045)
            self.h = self.cosmo.h
            self.epsilon_crit = self.get_epsiloncrit(zd,zsrc)
            self.sigmacrit = self.epsilon_crit*(0.001)**2*self.kpc_per_asec(zd)**2
            self.rhoc_physical = self.cosmo.critical_density0.value * 0.001 * self.M_sun**-1 * (100**3) * self.Mpc**3 # [M_sun Mpc^-3]
            self.rhoc = self.rhoc_physical*self.h**-2
            self.D_d,self.D_s,self.D_ds = self.D_A(0,zd),self.D_A(0,zsrc),self.D_A(zd,zsrc)

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
        if not hasattr(self,'epsilon_crit'):
            D_ds = self.D_A(z1, z2)
            D_d = self.D_A(0, z1)
            D_s = self.D_A(0, z2)
            # Units [M_sun arcsec^-2]
            epsilon_crit = (self.c**2*(4*np.pi*self.G)**-1)*(D_s*D_ds**-1*D_d**-1)
        return epsilon_crit

    def DMmass_inplane(self,angualr_area,zplane,dz):

        v_p = self.cosmo.differential_comoving_volume(zplane+dz)
        density = self.cosmo.critical_density(zplane).value*0.001 * self.M_sun**-1 * (100**3) * self.Mpc**3
        return v_p.value*angualr_area*dz * density * (1 - self.cosmo.Ob(zplane) - self.cosmo.Ode(zplane))
        #print v_p.value*dz*angualr_area

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



