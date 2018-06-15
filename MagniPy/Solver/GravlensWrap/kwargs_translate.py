import numpy as np
from MagniPy.util import cart_to_polar
from lenstronomy.Util.param_util import ellipticity2phi_q

def model_translate_togravlens(args, name):

    newargs = {}

    if name == 'SPEMD':

        newargs['phi_G'],newargs['q'] = ellipticity2phi_q(args['e1'],args['e2'])

        newargs['phi_G'] = newargs['phi_G'] - 0.5*np.pi

        newargs['ellip_theta'] = 180 * newargs['phi_G'] * np.pi ** -1

        newargs['ellip'] = 1- newargs['q']

        newargs['gamma'] = args['gamma']

        newargs['center_x'] = args['center_x']

        newargs['center_y'] = args['center_y']

        newargs['theta_E'] = args['theta_E'] * (((1+newargs['q']**2)*(2*newargs['q'])**-1)**.5)**-1

        del newargs['q']
        del newargs['phi_G']

        return newargs

    elif name == 'SERSIC_NFW':

        newargs['phi_G'], newargs['q'] = ellipticity2phi_q(args['SERSIC']['e1'], args['SERSIC']['e2'])

        newargs['phi_G'] = newargs['phi_G'] - 0.5 * np.pi

        newargs['ellip_theta'] = 180 * newargs['phi_G'] * np.pi ** -1

        newargs['ellip'] = 1 - newargs['q']

        newargs['center_x'] = args['SERSIC']['center_x']

        newargs['center_y'] = args['SERSIC']['center_y']

        newargs['n_sersic'] = args['SERSIC']['n_sersic']

        newargs['R_sersic'] = args['SERSIC']['R_sersic']

        newargs['k_eff'] = args['SERSIC']['k_eff']

        del newargs['q']
        del newargs['phi_G']

        return newargs

    elif name == 'SERSIC_NFW_disk':

        newargs['phi_G'], newargs['q'] = args['phi_G'], args['q']

        newargs['phi_G'] = newargs['phi_G'] - 0.5 * np.pi

        newargs['ellip_theta'] = 180 * newargs['phi_G'] * np.pi ** -1

        newargs['ellip'] = 1 - newargs['q']

        newargs['center_x'] = args['center_x']

        newargs['center_y'] = args['center_y']

        newargs['n_sersic'] = args['n_sersic']

        newargs['R_sersic'] = args['R_sersic']

        newargs['k_eff'] = args['k_eff']

        newargs['n_sersic_disk'] = args['n_sersic_disk']

        newargs['R_disk'] = args['R_disk']

        newargs['k_eff_disk'] = args['k_eff_disk']

        del newargs['q']
        del newargs['phi_G']

        return newargs

    else:
        return args



def lenstronomy_to_gravlens(param,param_name,**kwargs):

    if param_name == 'phi_G':

        return param + 0.5*np.pi

    elif param_name == 'theta_E':

        q = kwargs['q']

        factor = (((1+q**2)*(2*q)**-1)**.5)**-1
        return param*factor

    else:
        raise Exception('param name not recognized')