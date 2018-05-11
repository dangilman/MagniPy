from cosmology import Cosmo
from cosmology import ParticleMasses
from scipy.integrate import quad
from MagniPy.LensBuild.defaults import *
from scipy.special import hyp2f1
from colossus.cosmology import cosmology
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

        self.cosmology_params = {'omega_M_0':self.cosmo.Om0, 'omega_b_0':self.cosmo.Ob0, 'omega_lambda_0': 1 - self.cosmo.Om0,
                          'omega_n_0':0,'N_nu':1,'h':self.h,'sigma_8':self.sigma_8,'n':1}

        self.dm_particles = ParticleMasses(h=self.h)

        self.cosmo_set = False

    def _set_cosmo(self):

        if self.cosmo_set is True:
            return self.colossus_cosmo

        self.cosmo_set = True

        params = {'flat': True, 'H0': self.cosmology_params['h']*100, 'Om0':self.cosmology_params['omega_M_0'],
                  'Ob0':self.cosmology_params['omega_b_0'], 'sigma8':self.cosmology_params['sigma_8'], 'ns': 0.9608}

        self.colossus_cosmo = cosmology.setCosmology('custom',params)

        return cosmology.setCosmology('custom',params)

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

    def delta_lin(self):

        return self.delta_c

    def transfer_WDM(self,k,z,m_hm,n=1.12):

        alpha = self.dm_particles.wave_alpha(m_kev=self.dm_particles.hm_to_thermal(m_hm,h=self.h),
                                             omega_WDM=self.cosmo.Om(0),h=self.h)

        return (1+(alpha*k)**(2*n))**(-5*n**-1)

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

    def _angle_to_physicalradius(self, angle, z, z_base, Rein_def=None):

        angle_radian = angle*self.arcsec

        if Rein_def is None:
            Rein_def = default_Rein_deflection(angle)

        angle_deflection_reduced = Rein_def * self.arcsec

        angle_deflection = angle_deflection_reduced*self.D_s*self.D_ds**-1

        if z<=z_base:
            R = angle_radian*self.D_A(0,z)
        else:
            R = angle_radian * self.D_A(0, z) - angle_deflection * self.D_A(z_base, z)

        if R<0:
            return 0

        return R

    def angle_to_physical_area(self, angle, z, z_base, Rein_def=None):
        """
        computes the area corresponding to the angular radius of a plane at redshift z for a double cone with base at z_base
        :param angle: angle in arcsec
        :param z: redshift of plane
        :param z_base: redshift of cone base
        :return: comoving area
        """

        if Rein_def is None:
            Rein_def = default_Rein_deflection(angle)

        R = self._angle_to_physicalradius(angle, z, z_base, Rein_def=Rein_def)

        return np.pi*R**2

    def differential_physical_volume_cone(self, z, angle, z_base=None, Rein_def=None):
        """
        :param z: redshift
        :param angle: in arcseconds
        :param dz: redshift spacing
        :return:
        """

        scale_factor = (1+z)**-1

        if Rein_def is None:
            Rein_def = default_Rein_deflection(angle)

        return scale_factor*self.angle_to_physical_area(angle, z, z_base, Rein_def) * \
               self.cosmo.hubble_distance.value * self.cosmo.efunc(z) ** -1

    def comoving_volume_cone(self, z1, z2, angle, z_base=None, Rein_def=None):
        """
        computes the comoving volume in a surface specified by angle and z1,z2
        :param z1: start redshift
        :param z2: end redshift
        :return:
        """

        if z_base is None:
            z_base = self.zd

        if Rein_def is None:
            Rein_def = default_Rein_deflection(angle)

        def integrand(z, angle, z_base, Rein_def):

            return self.differential_physical_volume_cone(z, angle, z_base, Rein_def) * (1 + z) ** 3

        if z2-z1 < zstep:
            return integrand(z1, angle, z_base, Rein_def) * (z2 - z1)


        if isinstance(z2,float) or isinstance(z2,int):
            return quad(integrand, z1, z2, args=(angle, z_base, Rein_def))[0]
        else:
            integral = []
            for value in z2:

                integral.append(quad(integrand, z1, value, args=(angle, z_base, Rein_def))[0])
            return np.array(integral)

    def physical_volume_cone(self, z1, z2, angle, z_base=None, Rein_def=None):
        """
        computes the comoving volume in a surface specified by angle and z1,z2
        :param z1: start redshift
        :param z2: end redshift
        :return:
        """

        if z_base is None:
            z_base = self.zd

        if Rein_def is None:
            Rein_def = default_Rein_deflection(angle)

        def integrand(z,angle,z_base,Rein_def):

            return self.differential_physical_volume_cone(z, angle,z_base,Rein_def)

        if isinstance(z2,float) or isinstance(z2,int):
            return quad(integrand, z1, z2, args=(angle, z_base, Rein_def))[0]
        else:
            integral = []
            for value in z2:

                integral.append(quad(integrand, z1, value, args=(angle, z_base, Rein_def))[0])
            return np.array(integral)
