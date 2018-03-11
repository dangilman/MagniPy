import numpy as np

def gravlens_to_lenstronomy(param,param_name,**kwargs):

    if param_name == 'phi_G':

        return param - 0.5*np.pi

    elif param_name == 'theta_E':

        q = kwargs['q']

        factor = (((1+q**2)*(2*q)**-1)**.5)
        return param*factor

    elif param_name == 'gamma':

        return param

    else:
        raise Exception('param name not recognized')