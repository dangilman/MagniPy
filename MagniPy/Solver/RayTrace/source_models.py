import numpy as np
from scipy.special import gamma

class GAUSSIAN:

    def __init__(self,x,y,width):

        self.xcenter, self.ycenter = x, y
        self.width = width * 2.355 ** -1

    def __call__(self,betax,betay):

        dx,dy = -self.xcenter + np.array(betax),-self.ycenter + np.array(betay)

        return (2*np.pi*self.width**2)**-1*np.exp(-0.5*(dx**2+dy**2)*self.width**-2)

class TORUS:

    def __init__(self,x,y,inner,outer):

        self.xcenter,self.ycenter,self.width,self.inner= x,y,outer-inner,inner

    def __call__(self,betax,betay):

        dx,dy = -self.xcenter + np.array(betax),-self.ycenter + np.array(betay)
        dr = (dx**2+dy**2)**.5
        torus = np.zeros_like(dx)

        inds = (self.inner<=dr) & (dr<=self.inner+self.width)

        torus[inds] = 1

        return torus

class GAUSSIAN_TORUS:
    def __init__(self, x, y, width, inner, outer):
        self.gaussian = GAUSSIAN(x, y, width)
        self.torus = TORUS(x, y, inner, outer)

    def __call__(self, betax, betay):
        return self.gaussian(betax, betay) + self.torus(betax, betay)

class GAUSSIAN_SERSIC:
    def __init__(self, x, y, width, r_half, n_sersic):
        self.gaussian = GAUSSIAN(x, y, width)
        self.sersic = SERSIC(x, y, r_half, n_sersic)

    def __call__(self, betax, betay):
        return self.gaussian(betax, betay) + self.sersic(betax, betay)

class SERSIC:

    def __init__(self,x,y,r_half,n_sersic):

        self.xcenter, self.ycenter, self.r_half, self.n = x, y, r_half,n_sersic

    def __call__(self, betax, betay):

        dx, dy = -self.xcenter + np.array(betax), -self.ycenter + np.array(betay)
        dr = (dx ** 2 + dy ** 2) ** .5
        bn = 1.9992*self.n - 0.3271 + 4*(405*self.n)**-1
        norm = self.r_half**2*(2*np.pi*self.n)*np.exp(bn)*(bn)**(-2*self.n)*gamma(2*self.n)

        return norm**-1*np.exp(-bn*((dr*self.r_half**-1)**self.n**-1-1))
