import numpy as np
from colossus.lss.mass_function import *
import matplotlib.pyplot as plt
from MagniPy.LensBuild.Cosmology.extension import CosmoExtension

class HaloMassFunction:

    def __init__(self,model='reed07',sigma_8=0.82,rescale_sigma8=False,omega_M_void=None,**modelkwargs):

        self.extension = CosmoExtension(sigma_8=sigma_8,rescale_sigma8=rescale_sigma8,omega_M_void=omega_M_void)

        self.cosmology_params = self.extension.cosmology_params

        self.cosmology = self._set_cosmo()

        self.model = model

        self.modelkwargs = modelkwargs

    def _set_cosmo(self):

        params = {'flat': True, 'H0': self.cosmology_params['h']*100, 'Om0':self.cosmology_params['omega_M_0'],
                  'Ob0':self.cosmology_params['omega_b_0'], 'sigma8':self.cosmology_params['sigma_8'], 'ns': 0.9608}

        return cosmology.setCosmology('custom',params)

    def sigma(self,M,z):

        R = peaks.lagrangianR(M)

        return self.cosmology.sigma(R, z)

    def dN_dM_physical(self,M,z):
        """

        :param M: m200 in comoving units
        :param z: redshift
        :return: differential number per unit mass per cubic Mpc (physical)
        [h^3 N / M_odot / Mpc^3] where Mpc is phsical
        """
        q_out = 'dndlnM'
        #q_out = 'f'
        return massFunction(M,z,q_out=q_out,model=self.model,**self.modelkwargs)*M**-1

    def dN_dM_comoving(self, M, z):
        """

        :param M: m200 in comoving units
        :param z: redshift
        :return: differential number per unit mass per cubic Mpc (physical)
        [h^3 N / M_odot / Mpc^3] where Mpc is comoving
        """
        return (1+z)**3*self.dN_dM_physical(M,z)

    def fit_norm_index(self,M,dNdM):

        coeffs = np.polyfit(np.log10(M), np.log10(dNdM), 1)
        plaw_index = coeffs[0]
        norm = 10 ** coeffs[1]

        return norm,plaw_index

    def mass_function_moment(self, M, dNdM, N, m_low, m_high):

        """
        :param normalization: dimensions M_sun^-1 Mpc^-3 or M_sun
        :param plaw_index: power law index
        :param N: moment
        :return: Nth moment of the mass funciton
        """
        norm,plaw_index = self.fit_norm_index(M,dNdM)

        if plaw_index == 2:
            return norm * np.log(m_high * m_low ** -1)
        else:
            newindex = 1 + N + plaw_index
            return norm * newindex ** -1 * (m_high ** newindex - m_low ** newindex)

    def dndM_integrated_z1z2(self,M,z1,z2,delta_z_min=0.05,cone_base=3, Rein_def = 1):
        """
        From Reed et al 2006
        :param M:
        :param z1:
        :param z2:
        :param delta_z_min:
        :param cone_base:
        :param Rein_def:
        :return:
        """

        if z1<1e-4:
            z1 = 1e-4

        if z2 - z1 < delta_z_min:
            return self.get_dN_dM_physical(M, z1) * self.extension.physical_volume_cone(z1, z2 - z1, cone_base, base_deflection = Rein_def)

        N = int((z2 - z1)*delta_z_min**-1 + 1)

        zvals = np.linspace(z1,z2,N)
        dz = zvals[1] - zvals[0]
        integral = 0

        for z in zvals:

            integral += self.get_dN_dM_physical(M, z) * self.extension.physical_volume_cone(z, z + dz, cone_base, base_deflection = Rein_def)
        return integral

M = 10**np.arange(6,10,.1)
ax=plt.subplot(111)
z = 10
h = HaloMassFunction()
dndm = h.dN_dM_comoving(M,z)
h3 = 0.7**3

plt.plot(np.log10(M),np.log10(dndm),color='r')

h = HaloMassFunction(rescale_sigma8=True,omega_M_void=0.05)
dndm = h.dN_dM_comoving(M,z)
plt.plot(np.log10(M),np.log10(dndm),color='k')

plt.show()

