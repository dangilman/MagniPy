from MagniPy.LensBuild.main_deflector import Deflector
from MagniPy.MassModels.SIE import SIE
import numpy as np

default_source_shape = 'GAUSSIAN'
default_source_size_kpc = 0.01 #10 parsec

def get_default_SIE_random(z,varyflags=['1','1','1','1','1','1','1','0','0','0']):
    default_startkwargs_random = {'theta_E': np.absolute(np.random.normal(1, 0.1)),
                                  'ellip': np.absolute(np.random.normal(0.3, 0.05)),
                                  'ellip_theta': np.random.uniform(-90, 90),
                                  'center_x': 0, 'center_y': 0, 'shear': np.absolute(np.random.uniform(0.08, 0.03)),
                                  'shear_theta': np.random.uniform(-90, 90), 'gamma': 2}

    return Deflector(subclass=SIE(),tovary=True,varyflags=varyflags,redshift=z,lens_params=None,**default_startkwargs_random)

default_startkwargs = {'theta_E':1,'center_x':0,'center_y':0,'ellip':0.12,'ellip_theta':14,'shear':0.05,'shear_theta':-20,'gamma':2}
def get_default_SIE(z,varyflags = ['1','1','1','1','1','1','1','0','0','0']):
    return Deflector(subclass=SIE(),tovary=True,varyflags=varyflags,redshift=z,**default_startkwargs)


sigma_pos,sigma_flux,sigma_time = [[0.003]*4]*2,[0.3]*4,[0.02,2000,2000,2000]
default_sigmas = [sigma_pos,sigma_flux,sigma_time]

default_solve_method = 'lenstronomy'
raytrace_with_default = 'lenstronomy'
default_file_identifier = 'run'

def default_res(size):

    return 0.001

def default_Rein_deflection(cone_angle_size):

    return max(cone_angle_size,0)
