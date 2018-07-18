powerlaw_defaults = {'log_ML':6,'log_MH':10,'plaw_index':-1.9,'turnover_index':1.3,'subhalo_log_mL_low':-0.5}

spatial_defaults = {'Rmax_z_kpc':250,'theta_max':3,'nfw_core_kpc':6.1*25} # theta_max in arcseconds; radius in image plane
spatial_defaults['default_cone_base_factor'] = 1

profile_defaults = {'c_turnover':True,'core':1e-6}

filter_args = {'mindis':0.5,'log_masscut_low':7}

zstep = 0.02

default_source_shape = 'GAUSSIAN'
default_source_size = 0.0012*2.355**-1 #FWHM 10pc

concentration_turnover = True
concentration_power = 0.17

default_sigma8 = 0.82

default_halo_mass = 10**13

default_halo_mass_function = 'reed07'

kappa_Rein_default = 0.5

from astropy.cosmology import WMAP9
default_cosmology = WMAP9

from MagniPy.LensBuild.lens_assemble import Deflector
from MagniPy.MassModels.SIE import SIE
import numpy as np


def get_default_SIE_random(z,varyflags=['1','1','1','1','1','1','1','0','0','0']):
    default_startkwargs_random = {'R_ein': np.absolute(np.random.normal(1, 0.1)),
                                  'ellip': np.absolute(np.random.normal(0.3, 0.05)),
                                  'ellip_theta': np.random.uniform(-90, 90),
                                  'x': 0, 'y': 0, 'shear': np.random.uniform(0.03, 0.06),
                                  'shear_theta': np.random.uniform(-90, 90), 'gamma': 2}
    return Deflector(subclass=SIE(),tovary=True,varyflags=varyflags,redshift=z,lens_params=None,**default_startkwargs_random)

default_startkwargs = {'R_ein':1,'x':0,'y':0,'ellip':0.12,'ellip_theta':14,'shear':0.05,'shear_theta':-20,'gamma':2}
def get_default_SIE(z,varyflags = ['1','1','1','1','1','1','1','0','0','0']):
    return Deflector(subclass=SIE(),tovary=True,varyflags=varyflags,redshift=z,lens_params=None,**default_startkwargs)


sigma_pos,sigma_flux,sigma_time = [[0.003]*4]*2,[0.3]*4,[0.02,2000,2000,2000]
default_sigmas = [sigma_pos,sigma_flux,sigma_time]

default_solve_method = 'lenstronomy'
raytrace_with_default = 'lenstronomy'
default_file_identifier = 'run'

def default_gridrmax(srcsize):

    if srcsize <= 0.0006:
        return 0.06
    elif srcsize <= 0.0012:
        return 0.13
    elif srcsize <= 0.0016:
        return 0.16
    else:
        return 0.18

def default_res(srcsize):

    return 0.001

def default_Rein_deflection(cone_angle_size):

    return max(cone_angle_size,0)
