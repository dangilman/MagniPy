powerlaw_defaults = {'log_ML':6,'log_MH':10,'plaw_index':-1.9,'turnover_index':1.3}

spatial_defaults = {'Rmax_z_kpc':250,'theta_max':3,'nfw_core_kpc':6.1*25} # theta_max in arcseconds; radius in image plane
spatial_defaults['cone_base'] = 2*spatial_defaults['theta_max']

profile_defaults = {'c_turnover':True,'core':1e-6}

filter_args = {'mindis':0.5,'log_masscut_low':7}

zstep = 0.02

default_source_shape = 'GAUSSIAN'
default_source_size = 0.0012*2.355**-1 #FWHM 10pc

concentration_turnover = True

default_sigma8 = 0.82

default_halo_mass_function = 'reed07'

kappa_Rein_default = 0.5

#from astropy.cosmology import WMAP9 as cosmology
from astropy.cosmology import FlatLambdaCDM
default_cosmology = FlatLambdaCDM(H0=70,Om0=0.3,Ob0=0.046)

from MagniPy.LensBuild.lens_assemble import Deflector
from MagniPy.MassModels.SIE import SIE
startkwargs = {'R_ein':1,'ellip':0.2,'ellip_theta':-80,'x':0,'y':0,'shear':0.02,'shear_theta':0}
default_SIE = Deflector(subclass=SIE(),tovary=True,varyflags=['1','1','1','1','1','1','1','0','0','0'],redshift=None,**startkwargs)

sigma_pos,sigma_flux,sigma_time = [[0.003]*4]*2,[0.3]*4,[0,2,2,2]
default_sigmas = [sigma_pos,sigma_flux,sigma_time]

default_solve_method = 'lenstronomy'
raytrace_with_default = 'lenstronomy'
default_file_identifier = 'run'

def default_gridrmax(srcsize):

    if srcsize <= 0.0013*2.355**-1:
        return 0.045
    elif srcsize <= 0.0018*2.355**-1:
        return 0.08
    else:
        raise Exception('manually enter grid size')

def default_res(srcsize):

    if srcsize <= 0.0013*2.355**-1:
        return 0.0008
    elif srcsize <= 0.0018*2.355**-1:
        return 0.001
    else:
        raise Exception('manually enter grid size')

def default_Rein_deflection(cone_angle_size):

    return max(cone_angle_size-0.1,0)
