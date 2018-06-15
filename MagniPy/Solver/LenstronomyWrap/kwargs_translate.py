import numpy as np
from MagniPy.util import polar_to_cart
from lenstronomy.Util.param_util import phi_q2_ellipticity
from copy import deepcopy

def model_translate_tolenstronomy(args,name):

    # convert gravlens arguments to lenstronomy arguments

    if name not in ['SPEMD','TNFW','NFW','PJaffe','POINT_MASS','EXTERNAL_SHEAR','CONVERGENCE','SERSIC_NFW','SERSIC_NFW_DISK']:

        raise Exception(name + ' not recognized.')

    newargs = {}

    if name =='SPEMD':

        newargs['e1'], newargs['e2'] = phi_q2_ellipticity(args['phi_G'],args['q'])
        newargs['gamma'] = args['gamma']
        newargs['center_x'] = args['center_x']
        newargs['center_y'] = args['center_y']
        newargs['theta_E'] = args['theta_E']
        newargs['gamma'] = args['gamma']

        return newargs

    elif name=='SERSIC_NFW':
        newargs = deepcopy(args)
        newargs['e1'], newargs['e2'] = phi_q2_ellipticity(args['phi_G'], args['q'])
        del newargs['phi_G']
        del newargs['q']

        _newargs = {}

        _newargs['SERSIC'] = {'e1':newargs['e1'],'e2':newargs['e2'],
                            'center_x':newargs['center_x'],'center_y':newargs['center_y'],
                             'R_sersic':newargs['R_sersic'],'n_sersic':newargs['n_sersic'],'k_eff':newargs['k_eff']}
        _newargs['NFW'] = {'center_x': newargs['center_x'], 'center_y': newargs['center_y'],
                             'theta_Rs': newargs['theta_Rs'], 'Rs': newargs['Rs']}
        return _newargs

    elif name=='SERSIC_NFW_DISK':

        newargs = deepcopy(args)
        newargs['e1'], newargs['e2'] = phi_q2_ellipticity(args['phi_G'], args['q'])
        newargs['e1_disk'], newargs['e2_disk'] = phi_q2_ellipticity(args['phi_G'], args['q_disk'])
        del newargs['phi_G']
        del newargs['q']

        _newargs = {}

        _newargs['SERSIC'] = {'e1': newargs['e1'], 'e2': newargs['e2'],
                             'center_x': newargs['center_x'], 'center_y': newargs['center_y'],
                             'R_sersic': newargs['R_sersic'], 'n_sersic': newargs['n_sersic'],
                             'k_eff': newargs['k_eff']}
        _newargs['NFW'] = {'center_x': newargs['center_x'], 'center_y': newargs['center_y'],
                          'theta_Rs': newargs['theta_Rs'], 'Rs': newargs['Rs']}


        _newargs['SERSIC_DISK'] = {'e1':newargs['e1_disk'],'e2':newargs['e2_disk'],
                             'center_x':newargs['center_x'],'center_y':newargs['center_y'],
                             'R_sersic':newargs['R_disk'],'n_sersic':newargs['n_sersic_disk'],'k_eff':newargs['k_eff_disk']}


        return _newargs

    elif name == 'EXTERNAL_SHEAR':

        return args

    elif name == 'NFW' or name == 'TNFW':

        return args

    elif name == 'PJaffe':

        return args

    elif name == 'POINT_MASS':

        return args

    elif name == 'CONVERGENCE':

        return args

    else:
        raise Warning('specify translated kwargs for lens model.')

def gravlens_to_lenstronomy(param,param_name,**kwargs):

    if param_name == 'phi_G':

        return param - 0.5*np.pi

    elif param_name == 'theta_E':

        q = kwargs['q']

        factor = Rein_gravlens_to_lenstronomy(q)
        return param*factor

    elif param_name == 'gamma':

        return param

    else:
        raise Exception('param name not recognized')

def Rein_gravlens_to_lenstronomy(q):

    # turns a gravlens Einstein radius to lenstronomy einstein radius

    return (((1+q**2)*(2*q)**-1)**.5)