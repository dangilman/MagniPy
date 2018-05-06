import numpy as np
from colossus.lss.mass_function import *
from MagniPy.LensBuild.defaults import default_Rein_deflection,spatial_defaults,default_sigma8,default_halo_mass_function
import matplotlib.pyplot as plt
from MagniPy.LensBuild.Cosmology.extension import CosmoExtension

class HaloMassFunction:

    def __init__(self,model=None,sigma_8=None,rescale_sigma8=False,omega_M_void=None,zd=None,zsrc=None,**modelkwargs):

        if sigma_8 is None:
            sigma8 = default_sigma8
        else:
            sigma8 = sigma_8
        if model is None:
            model = default_halo_mass_function

        self.extension = CosmoExtension(zd=zd,zsrc=zsrc,sigma_8=sigma8,rescale_sigma8=rescale_sigma8,omega_M_void=omega_M_void)

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
        return self.extension.rho_matter_crit(z)*omega*M**-1

    def dN_dM_comoving(self,M,z):
        """

        :param M: m200 in comoving units
        :param z: redshift
        :return: differential number per unit mass per cubic Mpc (physical)
        [h^3 N / M_odot / Mpc^3] where Mpc is physical
        """
        q_out = 'dndlnM'

        h = self.extension.h

        M_h = M*h

        return h**3*massFunction(M_h,z,q_out=q_out,model=self.model,**self.modelkwargs)*M_h**-1

    def dN_dM_physical(self, M, z):
        """

        :param M: m200 in comoving units
        :param z: redshift
        :return: differential number per unit mass per cubic Mpc (physical)
        [h^3 N / M_odot / Mpc^3] where Mpc is comoving
        """
        return (1+z)**3*self.dN_dM_comoving(M,z)

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

    def dndM_integrated_z1z2(self,M,z1,z2,delta_z_min=0.01,cone_base=None, Rein_def = None, functype='plaw',
                             omega=None):

        if z1<1e-4:
            z1 = 1e-4

        assert cone_base is not None,'Specify cone_base.'

        if Rein_def is None:
            Rein_def = default_Rein_deflection(cone_base)

        if z2 - z1 < delta_z_min:

            if functype=='delta':
                return self.dN_dM_comoving_deltaFunc(M,z1,omega)*\
                       self.extension.comoving_volume_cone(z1, z2, cone_base, Rein_def= Rein_def)

            else:
                return self.dN_dM_comoving(M, z1) * self.extension.comoving_volume_cone(z1, z2, cone_base, Rein_def= Rein_def)

        N = int((z2 - z1)*delta_z_min**-1 + 1)

        zvals = np.linspace(z1,z2,N)
        dz = zvals[1] - zvals[0]
        integral = 0

        for z in zvals:

            if functype=='delta':
                integral += self.dN_dM_comoving_deltaFunc(M,z1,omega)*\
                       self.extension.comoving_volume_cone(z1, z + dz, cone_base, Rein_def= Rein_def)
            else:
                integral += self.dN_dM_comoving(M, z) * self.extension.comoving_volume_cone(z, z + dz,
                                                                                            cone_base, Rein_def= Rein_def)
        return integral
