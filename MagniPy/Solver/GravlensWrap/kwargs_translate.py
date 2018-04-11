import numpy as np
from MagniPy.util import cart_to_polar
from lenstronomy.Util.param_util import ellipticity2phi_q

def model_translate_togravlens(args, name):

    newargs = {}

    if name == 'SPEMD':

        newargs['phi_G'],newargs['q'] = ellipticity2phi_q(-args['e1'],-args['e2'])
        newargs['gamma'] = args['gamma']
        newargs['center_x'] = args['center_x']
        newargs['center_y'] = args['center_y']

        newargs['theta_E'] = args['theta_E'] * (((1+newargs['q']**2)*(2*newargs['q'])**-1)**.5)**-1

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