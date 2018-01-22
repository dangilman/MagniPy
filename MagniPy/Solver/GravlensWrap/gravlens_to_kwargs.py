from MagniPy.util import *

def translate(model_string):

    if model_string[0]=='alpha':

        name = 'SIE'
        b = float(model_string[1])
        x,y = float(model_string[2]),float(model_string[3])
        ellip = float(model_string[4])
        ellip_theta = float(model_string[5])
        shear = float(model_string[6])
        shear_theta = float(model_string[7])

        ellip,ellip_theta = shr_convert(ellip,ellip_theta,polar_to_cart=True)
        shear, shear_theta = shr_convert(shear, shear_theta, polar_to_cart=True)

        return {'name':name,'b':b,'ellip':ellip,'ellip_theta':ellip_theta,'shear':shear,'shear_theta':shear_theta,'x':x,'y':y}