def lensmod_to_kwargs(subx, suby, sub_b, subtrunc, subcore, ellip=0,ellip_PA=0,shear=0,shear_theta=0, subprofile=''):

    if subprofile=='pjaffe':

        return {'b':sub_b,'x0':subx,'y0':suby,'rcore':subcore,'rtrunc':subtrunc}

    elif subprofile=='nfw':

        return {'x0': subx, 'y0': suby, 'ks': sub_b,'rs': subtrunc}

    elif subprofile=='sersic':

        return {'x0': subx, 'y0': suby,'n':subtrunc,'re':subcore,'ke':sub_b,'ellip':[ellip],'ellip_PA':[ellip_PA]}

    elif subprofile=='SIE':

        return {'x0':subx,'y0': suby,'rcore':subcore,'rtrunc':subtrunc,'b':sub_b,'ellip':ellip,'ellip_theta':ellip_PA,
                'shear':shear,'shear_theta':shear_theta}

    elif subprofile=='tnfw' or subprofile=='tnfw3':
        return {'x0': subx, 'y0': suby, 'ks': sub_b, 'rs': subcore, 'tau':subtrunc}
