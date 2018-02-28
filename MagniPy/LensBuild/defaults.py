powerlaw_defaults = {'log_ML':6,'log_MH':10,'plaw_index':-1.9,'turnover_index':1.3}

spatial_defaults = {'Rmax_z_kpc':500,'theta_max':3,'nfw_core_kpc':100} # theta_max in arcseconds

profile_defaults = {'c_turnover':True,'core':1e-6}

zstep = 0.02

concentration_turnover = True

sigma_8 = 0.83

kappa_Rein = 0.5

from astropy.cosmology import WMAP9 as cosmology
default_cosmology = cosmology

from MagniPy.LensBuild.lens_assemble import Deflector
from MagniPy.MassModels.SIE import SIE
startkwargs = {'R_ein':1,'ellip':0.2,'ellip_theta':-80,'x':0,'y':0,'shear':0.02,'shear_theta':0}
default_SIE = Deflector(subclass=SIE(),tovary=True,varyflags=['1','1','1','1','1','1','1','0','0','0'],redshift=None,**startkwargs)

sigma_pos,sigma_flux,sigma_time = [[0.003]*4]*2,[0.3]*4,[0,2,2,2]
default_sigmas = [sigma_pos,sigma_flux,sigma_time]

def default_gridrmax(srcsize):

    if srcsize <= 0.0013*2.355**-1:
        return 0.45
    elif srcsize <= 0.0018*2.355**-1:
        return 0.8
    else:
        raise Exception('manually enter grid size')

def default_res(srcsize):

    if srcsize <= 0.0013*2.355**-1:
        return 0.0008
    elif srcsize <= 0.0018*2.355**-1:
        return 0.001
    else:
        raise Exception('manually enter grid size')