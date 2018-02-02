from MagniPy.util import *

def translate(model_string):

    if model_string[0]=='alpha':

        name = 'SIE'
        R_ein = float(model_string[1])
        x,y = float(model_string[2]),float(model_string[3])
        ellip = float(model_string[4])
        ellip_theta = float(model_string[5])
        ellip,ellip_theta = cart_to_polar(ellip, ellip_theta)
        shear = float(model_string[6])
        shear_theta = float(model_string[7])
        shear,shear_theta = cart_to_polar(shear, shear_theta)


        return {'name':name,'R_ein':R_ein,'ellip':ellip,'ellip_theta':ellip_theta,'shear':shear,'shear_theta':shear_theta,'x':x,'y':y}

    elif model_string[0]=='ptmass':

        name = 'ptmass'
        R_ein = float(model_string[1])
        x,y = float(model_string[2]),float(model_string[3])

        return {'name':name,'R_ein':R_ein,'x':x,'y':y}