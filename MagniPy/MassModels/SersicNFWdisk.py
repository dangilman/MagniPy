from SersicNFW import SersicNFW
import numpy as np

class SersicNFWdisk(object):

    def __init__(self,R0_fac=0.5):

        self.SersicNFW = SersicNFW(R0_fac=R0_fac)

    def params(self,R_ein=None, ellip=None, ellip_theta=None, x=None,
               y=None, Rs=None, n_sersic=None, reff_thetaE_ratio=None, f=None, fdisk=None, n_disk = 1, q_disk=None,
               reff_Rdisk_ratio=None,**kwargs):

        # as per Atlas 3d XVII I'm putting 20% of the total mass in the disk

        subparams = {}
        otherkwargs = {}

        otherkwargs['name'] = 'SERSIC_NFW_DISK'
        q = 1 - ellip
        subparams['q'] = q
        subparams['phi_G'] = (ellip_theta) * np.pi * 180 ** -1
        subparams['q_disk'] = q_disk

        subparams['Rs'] = Rs
        subparams['center_x'] = x
        subparams['center_y'] = y

        subparams['R_sersic'] = reff_thetaE_ratio * R_ein
        subparams['n_sersic'] = n_sersic
        subparams['R_disk'] = reff_Rdisk_ratio * subparams['R_sersic']
        subparams['n_sersic_disk'] = n_disk

        k_eff, ks_nfw = self.SersicNFW.normalizations(Rein=R_ein, re=subparams['R_sersic'], Rs=Rs,
                                            n=n_sersic, R0=self.SersicNFW.R0_fac * subparams['R_sersic'], f=f)

        subparams['k_eff'] = k_eff*(1-fdisk)
        subparams['k_eff_disk'] = k_eff * fdisk

        subparams['theta_Rs'] = 4 * ks_nfw * subparams['Rs'] * (1 + np.log(0.5))

        return subparams, otherkwargs
