import numpy as np
from scipy.integrate import quad

def ellip_r_square(u,x,y,q):

    return np.sqrt(u*x**2+u*y**2*(1-(1-q**2)*u)**-1)

def Jn(power,x,y,q,convergence_function,**kwargs):

    def integrand(u,x,y,q,N):

        return convergence_function(r=ellip_r_square(u,x,y,q),**kwargs)*(1-(1-q**2)*u)**(-N-0.5)

    return quad(integrand, 0, 1, args=(x,y,q,power))[0]


def xdef(x,y,q,convergence_function,**kwargs):

    if isinstance(x,float) and isinstance(y,float):
        return q*x*Jn(0,x,y,q,convergence_function,**kwargs)
    elif isinstance(x,np.ndarray) and isinstance(y,np.ndarray):

        assert x.shape==y.shape

        xvals = np.ravel(x)
        yvals = np.ravel(y)
        L = len(xvals)
        deflection = np.zeros_like(xvals)

        for i in range(0,L):

            deflection[i]=q*xvals[i]*Jn(0,xvals[i],yvals[i],q,convergence_function,**kwargs)

        deflection=deflection.reshape(int(len(x)),int(len(x)))

        return deflection

def ydef(x,y,q,convergence_function,**kwargs):

    if isinstance(x,float) and isinstance(y,float):
        return q*y*Jn(1,x,y,q,convergence_function,**kwargs)
    elif isinstance(x,np.ndarray) and isinstance(y,np.ndarray):

        assert x.shape == y.shape

        xvals = np.ravel(x)
        yvals = np.ravel(y)
        L = len(xvals)
        deflection = np.zeros_like(xvals)

        for i in range(0,len(x)):
            deflection[i]=q*yvals[i]*Jn(1,xvals[i],yvals[i],q,convergence_function,**kwargs)

        deflection = deflection.reshape(int(len(x)), int(len(x)))

        return deflection

def xydef(x,y,q,convergence_function,**kwargs):

    return xdef(x,y,q,convergence_function,**kwargs),ydef(x,y,q,convergence_function,**kwargs)
