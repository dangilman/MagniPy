from MagniPy.util import *
from kwargs_translate import lenstronomy_to_gravlens
from MagniPy.Solver.LenstronomyWrap.kwargs_translate import *

def gravlens_to_kwargs(model_string, shr_coords):

    if model_string[0]=='alpha':

        x,y = float(model_string[2]),float(model_string[3])
        e1 = float(model_string[4])
        e2 = float(model_string[5])

        if shr_coords==1:
            ellip,ellip_theta = cart_to_polar(e1, e2)
        else:
            ellip,ellip_theta = e1,e2

        q = 1-ellip

        R_ein = float(model_string[1])*(((1+q**2)*(2*q)**-1)**.5)

        phi_G = ellip_theta*np.pi*180**-1 + 0.5*np.pi

        shear = float(model_string[6])
        shear_theta = float(model_string[7])

        if shr_coords==1:
            shear,shear_theta = cart_to_polar(shear, shear_theta)

        gamma = 3-float(model_string[10])

        return {'theta_E':R_ein,'q':q,'phi_G':phi_G,'shear':shear,
                'shear_theta':shear_theta,'center_x':x,'center_y':y,'gamma':gamma}

    elif model_string[0]=='ptmass':

        name = 'ptmass'
        R_ein = float(model_string[1])
        x,y = float(model_string[2]),float(model_string[3])

        return {'name':name,'R_ein':R_ein,'x':x,'y':y}

def kwargs_to_gravlens(deflector=None,shr_coords=None):

    args = deflector.gravlens_args

    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'

    if deflector.profname=='SPEMD' or deflector.profname=='SIE':

        p0 = 'alpha'

        p1 = str(args['theta_E'])

        p2 = str(args['center_x'])
        p3 = str(args['center_y'])

        if shr_coords == 1:
            p4,p5 = polar_to_cart(args['ellip'],args['ellip_theta'])
        else:
            p4,p5 = args['ellip'],args['ellip_theta']

        p4,p5 = str(p4),str(p5)

        if deflector.has_shear:

            if shr_coords == 1:
                s, spa = polar_to_cart(deflector.shear, deflector.shear_theta)
            else:
                s, spa = deflector.shear, deflector.shear_theta

            p6 = str(s)
            p7 = str(spa)

        if args['gamma']==2:
            p10 = '1'
        else:
            p10 = str(3-args['gamma'])

    elif deflector.profname == 'NFW':

        p0 = 'nfw'
        p1 = args['theta_Rs']*(4*args['Rs']*(1+np.log(0.5)))**-1
        p2 = args['center_x']
        p3 = args['center_y']
        p8 = args['Rs']


    elif deflector.profname == 'TNFW':

        p0 = 'tnfw3'
        p1 = args['theta_Rs']*(4*args['Rs']*(1+np.log(0.5)))**-1
        p2 = args['center_x']
        p3 = args['center_y']
        p8 = args['Rs']
        p9 = args['r_trunc']*args['Rs']**-1
        p10 = '1'

    elif deflector.profname =='POINT_MASS':

        p0 = 'ptmass'
        p1 = str(args['theta_E'])
        p2 = str(args['center_x'])
        p3 = str(args['center_y'])

    elif deflector.profname =='PJaffe':

        p0 = 'pjaffe'
        p1 = str(args['b'])
        p2 = str(args['center_x'])
        p3 = str(args['center_y'])
        p8 = str(args['rt'])

    elif deflector.profname == 'CONVERGENCE':

        p0 = 'convrg'
        p1 = str(args['kappa_ext'])

    elif deflector.profname == 'SERSIC':

        p0 = 'sersic'
        p1 = str(args['k_eff'])
        p2 = str(args['center_x'])
        p3 = str(args['center_y'])

        if shr_coords==1:
            p4, p5 = polar_to_cart(args['ellip'], args['ellip_theta'])
        else:
            p4,p5 = args['ellip'],args['ellip_theta']
        p4 = str(args[p4])
        p5 = str(args[p5])

        if deflector.has_shear:
            if shr_coords == 1:
                s,spa = polar_to_cart(deflector.shear,deflector.shear_theta)
            else:
                s, spa = deflector.shear, deflector.shear_theta
            p6 = str(s)
            p7 = str(spa)

        p8 = str(args['r_eff'])
        p10 = str(args['n_sersic'])

    else:
        raise Exception('profile '+str(deflector.profname)+' not recognized.')

    return str(p0) + ' ' + str(p1) + ' ' + str(p2) + ' ' + str(p3) + ' ' + str(p4) + ' ' + str(p5) + ' ' + \
           str(p6) + ' ' + str(p7) + ' ' + str(p8) + ' ' + str(p9) + ' ' + str(p10)+' '