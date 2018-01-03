import numpy as np

def integrate_profile(profname,limit,inspheres=False,**kwargs):
    if profname=='nfw':
        rs=kwargs['rs']
        ks=kwargs['ks']
        n=limit*rs**-1
        if inspheres:
            rho0 = 86802621404*ks*rs**-1
            n*=rs
            r200 = kwargs['c']*rs

            return 4*np.pi*rho0*rs**3*(np.log(1+r200*n**-1)- n*(n+r200)**-1)

        else:
            return 2*np.pi*rs**2*ks*(np.log(.25*n**2)+2*np.arctanh(np.sqrt(1-n**2))*(np.sqrt(1-n**2))**-1)
    elif profname=='SIE':
        b = kwargs['SIE_Rein']
        return np.pi*limit*b

def rotate(xcoords,ycoords,angle):
    angle*=np.pi*180**-1
    #angle+=-np.pi*.5

    return xcoords*np.cos(angle)+ycoords*np.sin(angle),-xcoords*np.sin(angle)+ycoords*np.cos(angle)

def dr(x1,x2,y1,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def img_sept(x,y):

    return np.sort(np.array([dr(x[0],x[1],y[0],y[1]),dr(x[0],x[2],y[0],y[2]),dr(x[0],x[3],y[0],y[3]),
         dr(x[1],x[2],y[1],y[2]),dr(x[1],x[3],y[1],y[3]),dr(x[2],x[3],y[2],y[3])]))

def identify(x,y,RE):

    separations = img_sept(x,y)

    if separations[0]>=.7*RE:
        return 0

    else:

        if separations[1]<=1.2*RE:
            return 2
        else:
            return 1
