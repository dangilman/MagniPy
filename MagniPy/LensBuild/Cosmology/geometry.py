from MagniPy.LensBuild.Cosmology.cosmology import Cosmo,ParticleMasses
from scipy.integrate import quad
from MagniPy.LensBuild.defaults import *
from scipy.special import hyp2f1
from colossus.cosmology import cosmology
from copy import deepcopy

class Geometry(Cosmo):

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

    def angle_to_physicalradius(self, angle, z, z_base, delta_angle):

        angle_radian = angle * self.arcsec

        angle_deflection = delta_angle * self.arcsec * self.reduced_to_phys(self.zd,self.zsrc)

        R = angle_radian * self.D_A(0, z)
        #print(z,z_base)
        if z <= z_base:
            return R
        else:
            dR = - angle_deflection * self.D_A(z_base, z)
            #print(R,dR)
            #a=input('continue')
            return R + dR

    def angle_to_comovingradius(self, angle, z, z_base, delta_angle):

        physical_radius = self.angle_to_physicalradius(angle, z, z_base, delta_angle)

        return physical_radius*(1+z)

    def angle_to_physical_area(self, angle, z, z_base, delta_angle):
        """
        computes the area corresponding to the angular radius of a plane at redshift z for a double cone with base at z_base
        :param angle: angle in arcsec
        :param z: redshift of plane
        :param z_base: redshift of cone base
        :return: comoving area
        """

        R = self.angle_to_physicalradius(angle, z, z_base, delta_angle)

        return np.pi * R ** 2

    def angle_to_comoving_area(self, angle, z, z_base, delta_angle):
        """
        computes the area corresponding to the angular radius of a plane at redshift z for a double cone with base at z_base
        :param angle: angle in arcsec
        :param z: redshift of plane
        :param z_base: redshift of cone base
        :return: comoving area
        """

        R = self.angle_to_physicalradius(angle, z, z_base, delta_angle)

        R *= (1+z)

        return np.pi * R ** 2

    def comoving_volume_cone(self, z1, z2, angle, z_base, delta_angle):
        """
        computes the comoving volume in a surface specified by angle and z1,z2
        :param z1: start redshift
        :param z2: end redshift
        :return:
        """
        dz = 0.01
        nsteps = np.round((z2 - z1) * dz ** -1) + 1
        zvals = np.linspace(z1, z2, nsteps)

        volume = 0
        for z in zvals:
            dV = self._dV_cone_comoving(angle, z, z_base, delta_angle)
            volume += dV

        return volume


    def physical_volume_cone(self, z1, z2, angle, z_base, delta_angle):
        """
        computes the comoving volume in a surface specified by angle and z1,z2
        :param z1: start redshift
        :param z2: end redshift
        :return:
        """

        dz = 0.01
        nsteps = np.round((z2 - z1)*dz**-1)+1
        zvals = np.linspace(z1,z2,nsteps)

        volume = 0
        for z in zvals:

            dV = self._dV_cone_physical(angle,z,z_base,delta_angle)
            volume += dV

        return volume

    def _dV_cone_comoving(self, angle, z, z_base, delta_angle):

        dV_physical = self._dV_cone_physical(angle, z, z_base, delta_angle)

        return (1+z)**3 * dV_physical

    def _dV_cone_physical(self, angle, z, z_base, delta_angle):

        delta_comoving = self.cosmo.hubble_distance.value * self.cosmo.efunc(z) ** -1

        delta_phys = (1 + z) ** -1 * delta_comoving

        dV = self.angle_to_physical_area(angle, z, z_base, delta_angle) * delta_phys

        return dV
