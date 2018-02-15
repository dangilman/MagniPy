from cosmolopy.perturbation import *
from cosmology import Cosmo
import numpy as np
import matplotlib.pyplot as plt
from cosmology import ParticleMasses
from scipy.integrate import quad
from scipy.special import j1

class CosmoExtension(Cosmo):

    """
    the (A,a,p) parameters from Despali 2016 used to normalized the mass function
    """
    A = 0.333
    a = 0.794
    p = 0.247

    sigma8 = 0.8

    def __init__(self,zd=0.5,zsrc=1.5):

        Cosmo.__init__(self,zd=zd,zsrc=zsrc)

        self.cosmology = {'omega_M_0':self.cosmo.Om0,'omega_b_0':self.cosmo.Ob0,'omega_lambda_0':1-self.cosmo.Om0,
                          'omega_n_0':0,'N_nu':1,'h':self.h,'sigma_8':self.sigma8,'n':1}

        self.dm_particles = ParticleMasses(h=self.h)

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
    def sigma_numerical(self,r,z,m_hm):

        def integrand(k,r,z,m_hm):
            x = k*r

            return (2*np.pi**2)**-1*k**2*self.power_spectrum(k,z,m_hm=m_hm)*(3*j1(x)*x**-1)**2
        return 4.02885421452*quad(integrand,0,np.inf,args=(r,z,m_hm),limit=500)[0]

    def sigma(self,r,z):

        """

        :param r: length scale in Mpc
        :param z: redshift
        :return:
        """

        return sigma_r(r,z,**self.cosmology)[0]

    def delta_lin(self,z):

        return 1.68647*fgrowth(z,self.cosmology['omega_M_0'])**-1

    def Nu(self,r,z,m_hm=0):

        """

        :return: the parameter "nu" defined as the critical overdensity squared divided by sigma
        """
        if m_hm==0:
            return (self.delta_lin(z)*self.sigma(r,z)**-1)**2
        else:
            return (self.delta_lin(z)*self.sigma_numerical(r,z,m_hm=m_hm)**-1)**2

    def f_nu(self,nu):

        nu_p = self.a*nu

        return (self.A*nu**-1)*(1+nu_p**-self.p)*(nu_p*(2*np.pi)**-1)**.5*np.exp(-0.5*nu_p)

    def dnu_dM(self,M,z):

        """
        calibrated for 5<M<10 solar masses
        and
        0<z<0.6

        :param M: mass in solar masses
        :return: d(nu) / dM
        """

        def a(z):
            return 0.01667204 * z ** 2 - 0.03291005 * z + 0.02367598

        def b(z):
            return -0.3941185 * z ** 2 + 0.87143976 * z - 0.71429476

        def c(z):
            return 2.25605581 * z ** 2 - 5.86192512 * z + 5.45501308

        def fnu_fit(logm, z):
            return a(z) * logm ** 2 + b(z) * logm + c(z)

        logM = np.log10(M)

        return np.absolute(M**-1*(2*a(z)*logM + b(z)))

    def mass2size_physical(self,m,z):
        """

        :param m: mass in solar masses
        :param z: redshift
        :return: physical distance corresponding to a sphere of mass M computed w.r.t. background
        """
        comoving = (1+z)**-1
        #comoving=1
        return comoving*(3*m*(4*np.pi*self.rho_matter_crit(z))**-1)**(1*3**-1)

    def mass2wavenumber_physical(self,m,z):
        """

        :param m: mass in solar masses
        :param z: redshift
        :return: physical distance corresponding to a sphere of mass M computed w.r.t. background
        """
        return 2*np.pi*self.mass2size_physical(m,z)**-1

    def f_nu_mass(self,m,z,m_hm=0):
        nu = self.Nu(self.mass2size_physical(m,z),z,m_hm=m_hm)
        return self.f_nu(nu)

    def comoving_ShethTormen_density(self,M,z,m_hm=0):

        return self.rho_matter_crit(z)*M**-1*self.dnu_dM(M,z)*self.f_nu_mass(M,z,m_hm=m_hm)

    def physical_ShethTormen(self,M,z1,z2,angle):

        zmin = 0.01
        delta_z = z2-z1


        if delta_z <= zmin:

            return self.comoving_ShethTormen_density(M,z1)*self.comoving_volume_disk(z1,z2,angle)
        else:

            dz = 0.01
            z = np.linspace(z1,z2,1+np.ceil((z2-z1)*dz**-1))

            value = 0
            for zval in z:
                value+=self.comoving_ShethTormen_density(M,zval)*self.comoving_volume_disk(zval,zval+0.01,angle)
            return value


    def differential_comoving_volume_disk(self,z,angle):
        """

        :param z: redshift
        :param angle: in arcseconds
        :param dz: redshift spacing
        :return:
        """
        angle *= self.arcsec

        return self.cosmo.hubble_distance.value*np.pi*angle**2*self.cosmo.comoving_distance(z).value**2*self.cosmo.efunc(z)**-1

    def comoving_volume_disk(self,z1,z2,angle):
        """
        computes the comoving volume in a surface specified by angle and z1,z2
        same as differential_comoving_volume_disk for z2-z1 ~ 0
        :param z1: start redshift
        :param z2: end redshift
        :return:
        """
        def integrand(z,angle):
            return self.differential_comoving_volume_disk(z,angle)
        if isinstance(z2,float) or isinstance(z2,int):
            return quad(integrand,z1,z2,args=(angle))[0]
        else:
            integral = []
            for value in z2:
                integral.append(quad(integrand,z1,value,args=(angle))[0])
            return np.array(integral)


C = CosmoExtension()

M = np.logspace(6,11,100)
y1 = C.comoving_ShethTormen_density(M,0.6)

plt.loglog(M,np.array(y1))

vals = np.polyfit(np.log10(M),np.log10(y1),1)

def fun(vals,M):
    return 10**vals[1]*M**vals[0]
plt.loglog(M,fun(vals,M),color='r')
plt.show()
print vals