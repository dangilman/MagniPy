from MagniPy.LensBuild.Cosmology.cosmology import Cosmo
import numpy as np

def reff_sigma(vdis):

    return (4.1*vdis*250**-1)**1.5

def parameter_generate(zlens=None,zsrc=None,mean_vdis=None, vdis_sigma = 15,mean_sersic_index=4.8,sersic_index_sigma=1,
                       mean_DM_fraction=0.4,DM_fraction_sigma=0.05,halo_mass = 10**13,RS_sigma=10):

    c = Cosmo(zlens, zsrc)

    rho0_kpc, mean_Rs, r200_kpc = c.NFW(halo_mass,c=3,z=zlens)

    vdis = np.random.normal(mean_vdis, vdis_sigma)
    theta_E = c.vdis_to_Rein(zlens, zsrc, vdis)

    reff = reff_sigma(vdis)*c.kpc_per_asec(zlens)**-1

    ratio = theta_E * reff**-1

    Rs_values = np.array([mean_Rs, mean_Rs*0.1]) * c.kpc_per_asec(zlens)**-1
    Rs = np.random.normal(Rs_values[0], Rs_values[1])

    n_values = [mean_sersic_index, sersic_index_sigma]
    n_sersic = np.random.normal(n_values[0], n_values[1])

    f_values = [mean_DM_fraction, DM_fraction_sigma]
    f = np.random.normal(f_values[0], f_values[1])

    return {'theta_E':theta_E,'Rs':Rs,'ratio':ratio,'n_sersic':n_sersic,'f':f}
