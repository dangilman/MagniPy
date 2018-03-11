import numpy as np

def lenstronomy_to_gravlens(param,param_name,**kwargs):

    if param_name == 'phi_G':

        return param + 0.5*np.pi

    elif param_name == 'theta_E':

        q = kwargs['q']

        factor = (((1+q**2)*(2*q)**-1)**.5)**-1
        return param*factor

    else:
        raise Exception('param name not recognized')