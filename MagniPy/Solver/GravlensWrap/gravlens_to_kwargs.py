from MagniPy.util import *

def gravlens_to_kwargs(model_string, deflector=None):

    if model_string[0]=='alpha':

        x,y = float(model_string[2]),float(model_string[3])
        e1 = float(model_string[4])
        e2 = float(model_string[5])
        ellip,ellip_theta = cart_to_polar(e1, e2)

        q = 1-ellip
        prefactor = ((1 + q ** 2) * (2 * q)**-1) ** .5
        prefactor = 1
        R_ein = float(model_string[1]) * prefactor**-1
        phi_G = ellip_theta*np.pi*180**-1
        shear = float(model_string[6])
        shear_theta = float(model_string[7])

        shear,shear_theta = cart_to_polar(shear, shear_theta)


        return {'theta_E':R_ein,'q':q,'phi_G':phi_G,'shear':shear,
                'shear_theta':shear_theta,'center_x':x,'center_y':y}

    elif model_string[0]=='ptmass':

        name = 'ptmass'
        R_ein = float(model_string[1])
        x,y = float(model_string[2]),float(model_string[3])

        return {'name':name,'R_ein':R_ein,'x':x,'y':y}

def kwargs_to_gravlens(deflector=None):

    args = deflector.args

    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'

    if deflector.profname=='SPEMD' or deflector.profname=='SIE':
        q = args['q']
        #prefactor = ((1 + q ** 2) * (2 * q) ** -1) ** .5
        p0 = 'alpha'
        prefactor = 1
        p1 = str(args['theta_E']*prefactor**-1)
        p2 = str(args['center_x'])
        p3 = str(args['center_y'])

        p4,p5 = polar_to_cart(1-args['q'],(args['phi_G'])*180*np.pi**-1)

        p4,p5 = str(p4),str(p5)


        if deflector.has_shear:
            s,spa = polar_to_cart(deflector.shear,deflector.shear_theta)
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
        p9 = args['t']*args['Rs']**-1
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

    return str(p0) + ' ' + str(p1) + ' ' + str(p2) + ' ' + str(p3) + ' ' + str(p4) + ' ' + str(p5) + ' ' + \
           str(p6) + ' ' + str(p7) + ' ' + str(p8) + ' ' + str(p9) + ' ' + str(p10)+' '