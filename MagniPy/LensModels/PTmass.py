import numpy as np

class PointMass:

    def __init__(self, xgrid, ygrid):
        self.x, self.y = xgrid, ygrid

    def def_angle(self,x0=None, y0=None, b=None):

        x = self.x - x0
        y = self.y - y0
        r = np.sqrt(x ** 2 + y ** 2 + 0.0000000001)

        magdef = b**2*r**-1

        return magdef * x * r ** -1, magdef * y * r ** -1

    def F(self,x):

        if type(x) is np.ndarray:
            inds1=np.where(x<1)
            inds2=np.where(x>1)
            nfwvals=x
            nfwvals[inds1]=(1 - x[inds1] ** 2)**-.5 * np.arctanh((1 - x[inds1] ** 2)**.5)
            nfwvals[inds2]=(x[inds2] ** 2-1)**-.5 * np.arctan((x[inds2] ** 2-1)**.5)

            return nfwvals

        else:
            if x<1:
                return (pow(1 - x ** 2, -.5) * np.arctanh(pow(1 - x ** 2, .5)))
            elif x>1:
                return (pow(x ** 2 - 1, -.5) * np.arctan(pow(x ** 2 - 1, .5)))
            else:
                return 1
