import numpy as np
from grid_location import Local

class NFW:

    def def_angle(self,x_grid,y_grid,x0=None,y0=None,rs=None,ks=None,rt=None):

        assert rs > 0

        x = x_grid - x0
        y = y_grid - y0
        tau = rt*rs**-1

        r = np.sqrt(x ** 2 + y ** 2 + 0.0000000000001)
        xnfw = r*rs**-1

        softening = 0.0001
        xnfw[np.where(xnfw<softening)]=softening

        magdef = 4*ks*rs*self.t_fac(xnfw,tau)*xnfw**-1

        return magdef * x * r ** -1, magdef * y * r ** -1

    def F(self,x):

        if isinstance(x, np.ndarray):
            nfwvals = np.ones_like(x)

            inds1 = np.where(x < 1)
            inds2 = np.where(x > 1)
            nfwvals[inds1] = (1 - x[inds1] ** 2) ** -.5 * np.arccosh(x[inds1]**-1)
            nfwvals[inds2] = (x[inds2] ** 2 - 1) ** -.5 * np.arccos(x[inds2]**-1)
            return nfwvals

        elif isinstance(x, float) or isinstance(x, int):
            if x == 1:
                return 1
            if x < 1:
                return (1 - x ** 2) ** -.5 * np.arctanh((1 - x ** 2) ** .5)
            else:
                return (x ** 2 - 1) ** -.5 * np.arctan((x ** 2 - 1) ** .5)

    def L(self,x,tau):

        return np.log(x*(tau+np.sqrt(tau**2+x**2))**-1)

    def t_fac(self, x, tau):
        return tau ** 2 * (tau ** 2 + 1) ** -2 * (
        (tau ** 2 + 1 + 2 * (x ** 2 - 1)) * self.F(x) + tau * np.pi + (tau ** 2 - 1) * np.log(tau) +
        np.sqrt(tau ** 2 + x ** 2) * (-np.pi + self.L(x, tau) * (tau ** 2 - 1) * tau ** -1))

    def convergence(self,r=None,ks=None,rs=None,xy=None):
        if xy is not None:
            assert isinstance(xy,list)
            assert len(xy)==2

            r = np.sqrt(xy[0]**2+xy[1]**2)
        if r is None:
            r = np.sqrt(self.x**2+self.y**2)
        return 2*ks*(1-self.F(r*rs**-1))*((r*rs**-1)**2-1)**-1

