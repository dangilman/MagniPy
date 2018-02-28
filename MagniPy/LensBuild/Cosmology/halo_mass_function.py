import numpy as np
from colossus.lss.mass_function import *
import matplotlib.pyplot as plt
from MagniPy.LensBuild.Cosmology.extension import CosmoExtension

class HaloMassFunction:

    def __init__(self,model='reed07',sigma_8=0.82,rescale_sigma8=False,omega_M_void=None,zd=None,zsrc=None,**modelkwargs):

        self.extension = CosmoExtension(zd=zd,zsrc=zsrc,sigma_8=sigma_8,rescale_sigma8=rescale_sigma8,omega_M_void=omega_M_void)

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

    def dN_dM_comoving_deltaFunc(self,M,z,omega):

        """

        :param z: redshift
        :param omega: density parameter; fraction of the matter density (not fraction of critical density)
        :return: the number of objects of mass M * Mpc^-3
        """
        return self.cosmo.rho_matter_crit(z)*omega*M**-1

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
        return (1+z)**-3*self.dN_dM_physical(M,z)

    def fit_norm_index(self,M,dNdM,order=1):

        coeffs = np.polyfit(np.log10(M), np.log10(dNdM), order)
        plaw_index = coeffs[0]
        norm = 10 ** coeffs[1]

        return norm,plaw_index

    def mass_function_moment(self, M, dNdM, N, m_low=None, m_high=None,order=1):

        """
        :param normalization: dimensions M_sun^-1 Mpc^-3 or M_sun
        :param plaw_index: power law index
        :param N: moment
        :return: Nth moment of the mass funciton
        """

        if m_low is None or m_high is None:
            m_low,m_high = np.min(M),np.max(M)

        norm,plaw_index = self.fit_norm_index(M,dNdM,order=order)

        if plaw_index == 2 and N==1:
            Nsub = norm * np.log(m_high * m_low ** -1)
            return Nsub,norm,plaw_index
        else:
            newindex = 1 + N + plaw_index
            Nsub = norm * newindex ** -1 * (m_high ** newindex - m_low ** newindex)
            return Nsub,norm,plaw_index

    def dndM_integrated_z1z2_old(self,M,z1,z2,delta_z_min=0.02,cone_base=3, Rein_def = 1,physical=False,functype='plaw',
                             omega=None):

        if z1<1e-4:
            z1 = 1e-4

        if z2 - z1 < delta_z_min:

            if functype=='delta':
                return self.dN_dM_comoving_deltaFunc(M,z1,omega)*\
                       self.extension.comoving_volume_cone(z1, z2, cone_base, base_deflection = Rein_def)

            else:

                if physical:
                    return self.dN_dM_physical(M, z1) * self.extension.physical_volume_cone(z1, z2, cone_base,
                                                                base_deflection=Rein_def)
                else:
                    return self.dN_dM_comoving(M, z1) * self.extension.comoving_volume_cone(z1, z2, cone_base, base_deflection = Rein_def)

        N = int((z2 - z1)*delta_z_min**-1 + 1)

        zvals = np.linspace(z1,z2,N)
        dz = zvals[1] - zvals[0]
        integral = 0

        for z in zvals:

            if functype=='delta':
                integral += self.dN_dM_comoving_deltaFunc(M,z1,omega)*\
                       self.extension.comoving_volume_cone(z1, z+dz, cone_base, base_deflection = Rein_def)
            else:
                integral += dz*self.dN_dM_comoving(M, z) * self.extension.differential_physical_volume_cone(z,cone_base,
                                                                                                z_base=self.extension.zd,base_deflection=1)
        return integral

    def dndM_integrated_z1z2(self,M,z1,z2,delta_z_min=0.04,cone_base=3, Rein_def = 1,functype='plaw',
                             omega=None):

        if z1<1e-4:
            z1 = 1e-4

        if z2 - z1 < delta_z_min:

            if functype=='delta':
                return self.dN_dM_comoving_deltaFunc(M,z1,omega)*\
                       self.extension.comoving_volume_cone(z1, z2, cone_base, base_deflection = Rein_def)

            else:
                return self.dN_dM_comoving(M, z1) * self.extension.comoving_volume_cone(z1, z2, cone_base, base_deflection = Rein_def)

        N = int((z2 - z1)*delta_z_min**-1 + 1)

        zvals = np.linspace(z1,z2,N)
        dz = zvals[1] - zvals[0]
        integral = 0

        for z in zvals:

            if functype=='delta':
                integral += self.dN_dM_comoving_deltaFunc(M,z1,omega)*\
                       self.extension.comoving_volume_cone(z1, z+dz, cone_base, base_deflection = Rein_def)
            else:
                integral += self.dN_dM_comoving(M, z) * self.extension.comoving_volume_cone(z, z + dz,
                                                                                                cone_base, base_deflection = Rein_def)
        return integral

