from cosmolopy.perturbation import *
from cosmology import Cosmo
import numpy as np
import matplotlib.pyplot as plt
from cosmology import ParticleMasses
from scipy.integrate import quad
from scipy.special import j1
from scipy.special import hyp2f1
from copy import deepcopy

class CosmoExtension(Cosmo):

    def __init__(self,zd=0.5,zsrc=1.5,sigma_8=0.83,rescale_sigma8 = None, omega_M_void=0.035):

        Cosmo.__init__(self,zd=zd,zsrc=zsrc)

        if rescale_sigma8 is True:
            self.sigma_8 = self.rescale_sigma8(sigma_8,void_omega_M=omega_M_void)
            self.delta_c = 1.62 # appropriate for an "open" universe in the void
        else:
            self.sigma_8 = sigma_8
            self.delta_c = 1.68647 # appropriate for a flat universe

        self.cosmology = {'omega_M_0':self.cosmo.Om0,'omega_b_0':self.cosmo.Ob0,'omega_lambda_0':1-self.cosmo.Om0,
                          'omega_n_0':0,'N_nu':1,'h':self.h,'sigma_8':self.sigma_8,'n':1}

        self.dm_particles = ParticleMasses(h=self.h)

    def D_growth(self,z,omega_M,omega_L):

        def f(x,OmO,OmL):
            return (1+OmO*(x**-1 - 1)+OmL*(x**2-1))**-.5

        a = (1+z)**-1

        if omega_M+omega_L != 1:
            return a * hyp2f1(3 ** -1, 1, 11 * 6 ** -1, a ** 3 * (1 - omega_M ** -1))
        else:
            prefactor = 5*omega_M*(2*a*f(a,omega_M,omega_L))**-1
            return prefactor*quad(f,0,a,args=(omega_M,omega_L))[0]

    def rescale_sigma8(self,sigma_8_init,void_omega_M):

        """
        :param sigma_8_init: initial cosmological sigma8 in the field
        :param void_omega_M: the matter density in the void
        :return: a rescaled sigma8 appropriate for an under dense region
        Gottlober et al 2003
        """

        zi = 1000

        D_ai = self.D_growth(zi,self.cosmo.Om0,self.cosmo.Ode0)
        D_a1 = self.D_growth(0,self.cosmo.Om0,self.cosmo.Ode0)

        D_void_ai = self.D_growth(zi, void_omega_M,self.cosmo.Ode0)
        D_void_a1 = self.D_growth(0, void_omega_M, self.cosmo.Ode0)

        return sigma_8_init*(D_ai*D_void_a1)*(D_a1*D_void_ai)**-1

    def delta_lin(self,z):

        fgrow = fgrowth(z,self.cosmology['omega_M_0'])

        return self.delta_c*fgrow**-1

    def sigma(self,r,z):

        """

        :param r: length scale in Mpc
        :param z: redshift
        :return:
        """

        return sigma_r(r,z,**self.cosmology)[0]

    def sigma_inv_log(self, sigma):

        return np.log(sigma ** -1)


    def transfer_WDM(self,k,z,m_hm,n=1.12):

        alpha = self.dm_particles.wave_alpha(m_kev=self.dm_particles.hm_to_thermal(m_hm,h=self.h),
                                             omega_WDM=self.cosmo.Om(0),h=self.h)

        return (1+(alpha*k)**(2*n))**(-5*n**-1)

    def power_spectrum(self,k,z,m_hm=0):
        """

        :param k: wave number in Mpc^-1
        :param z: redshift
        :return:
        """
        if m_hm == 0:
            transfer_WDM = 1
        else:
            transfer_WDM = self.transfer_WDM(k,z,m_hm)

        return power_spectrum(k,z,**self.cosmology)*transfer_WDM**2

    def mass2size_comoving(self, m, z):
        """

        :param m: mass in solar masses
        :param z: redshift
        :return: comoving distance corresponding to a sphere of mass M computed w.r.t. background
        """

        return (3*m*(4*np.pi*self.rho_matter_crit(z))**-1)**(1*3**-1)

    def mass2wavenumber_comoving(self, m, z):
        """

        :param m: mass in solar masses
        :param z: redshift
        :return: physical distance corresponding to a sphere of mass M computed w.r.t. background
        """
        return 2*np.pi* self.mass2size_comoving(m, z) ** -1

    def DsigmaInv_DlnM(self, M, z):

        sigma = self.sigma(self.mass2size_comoving(M, z), z)
        sigma_inv_log = self.sigma_inv_log(sigma)

        return np.polyval(np.polyder(np.polyfit(np.log10(M), sigma_inv_log, 2)), np.log10(M))

    def differential_comoving_volume_cone(self, z, angle):
        """
        :param z: redshift
        :param angle: in arcseconds
        :param dz: redshift spacing
        :return:
        """

        if z <= self.zd:

            angle *= self.arcsec

            base = angle*self.cosmo.comoving_distance(z).value

            return np.pi*base**2*self.cosmo.hubble_distance.value*self.cosmo.efunc(z)**-1

        else:

            reduced_Rein_deflection = 1 # arcsec

            physical_Rein_deflection = reduced_Rein_deflection*self.D_s * self.D_ds ** -1

            base = (self.D_d*angle - physical_Rein_deflection*self.D_A(self.zd,z))*self.arcsec

            return np.pi*base**2*self.cosmo.hubble_distance.value*self.cosmo.efunc(z)**-1


    def differential_comoving_volume_cylinder(self, z, angle):
        """

        :param z: redshift
        :param angle: in arcseconds
        :param dz: redshift spacing
        :return:
        """
        angle *= self.arcsec

        return self.cosmo.hubble_distance.value*np.pi*angle**2*self.D_d**2*self.cosmo.efunc(z)**-1

    def comoving_volume_cone(self, z1, z2, angle):
        """
        computes the comoving volume in a surface specified by angle and z1,z2, is an expanding cylinder
        same as differential_comoving_volume_disk for z2-z1 ~ 0
        :param z1: start redshift
        :param z2: end redshift
        :return:
        """
        def integrand(z,angle):

            return self.differential_comoving_volume_cone(z, angle)

        if isinstance(z2,float) or isinstance(z2,int):
            return quad(integrand,z1,z2,args=(angle))[0]
        else:
            integral = []
            for value in z2:

                integral.append(quad(integrand,z1,value,args=(angle))[0])
            return np.array(integral)

    def comoving_volume_cylinder(self,z1,z2,angle):
        """
        computes the comoving volume in a surface specified by angle, and z1 z2. is a cylinder
        same as differential_comoving_volume_disk for z2-z1 ~ 0
        :param z1: start redshift
        :param z2: end redshift
        :return:
        """

        def integrand(z,angle):
            return self.differential_comoving_volume_cylinder(z, angle)

        if isinstance(z2,float) or isinstance(z2,int):
            return quad(integrand,z1,z2,args=(angle))[0]
        else:
            integral = []
            for value in z2:
                integral.append(quad(integrand,z1,value,args=(angle))[0])
            return np.array(integral)
