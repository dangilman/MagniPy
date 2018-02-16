from extension import CosmoExtension
import numpy as np
import matplotlib.pyplot as plt

def mass_function_moment(norm,plaw_index,N,m_low,m_high):

    """
    :param normalization: dimensions M_sun^-1 Mpc^-3 or M_sun
    :param plaw_index: power law index
    :param N: moment
    :return: Nth moment of the mass funciton
    """

    if plaw_index==2:
        return norm*np.log(m_high*m_low**-1)
    else:
        newindex = 1+N+plaw_index
        return norm*newindex**-1*(m_high**newindex - m_low**newindex)

class ShethTormen(CosmoExtension):

    def f_sigma(self,sigma,z,neff,A_prime=0.31,p=0.3,ca=0.764):

        w = (ca*self.delta_lin(z)**2*sigma**-2)**.5

        G1 = np.exp(-0.5*(np.log(w)-0.788)**2*0.6**-2)

        G2 = np.exp(-0.5*(np.log(w)-1.138)**2*0.2**-2)

        return A_prime*w*(2*np.pi**-1)**.5*(1 + 1.02*w**(2*p) + 0.6*G1 + 0.4*G2)*\
               np.exp(-0.5*w - (0.0325*w**p)*(neff + 3)**-2)

    def neff_approx(self,sigma,z):

        def mz(z):
            return 0.55-0.32*(1-(1+z)**-1)**5
        def rz(z):
            return -1.74-0.8*(np.log((1+z)**-1))**0.8

        return mz(z)*np.log(sigma**-1) + rz(z)

    def Dn_DM_density(self, M, z):
        """

        :param M: comoving m200
        :param z: redshift
        :return: dN/dm (d Mpc)^-3 comoving
        """

        neff = 6*self.DsigmaInv_DlnM(M,z) - 3

        sigma = self.sigma(self.mass2size_comoving(M, z), z)
        #neff = self.neff_approx(sigma, z)


        return self.rho_matter_crit(z) * M ** -2 * self.f_sigma(sigma,z,neff) * self.DsigmaInv_DlnM(sigma,z)

    def Dn_DM(self,M,z,dz=0.05,delta_z_min=0.05):

        assert isinstance(z,float) or isinstance(z,int)
        assert dz <= delta_z_min

        return self.Dn_DM_density(M, z) * self.comoving_volume_cone(z, z + dz, 3)

if False:
    M = np.logspace(6,10)
    m = ShethTormen()
    z2 = 0.4
    dndm = m.Dn_DM_density(M,0)
    dndm2 = m.Dn_DM_density(M,z2)
    plt.loglog(M,dndm,color='k')
    plt.loglog(M,dndm2,color='r')
    plt.show()