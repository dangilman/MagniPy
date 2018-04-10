import numpy as np
from scipy.integrate import dblquad

class GaussianKernel:

    def __init__(self,Nsamples=None):

        self.h = self.scotts_factor(n=Nsamples)**2

    def evaluate(self,x_point,y_point,x_sample,y_sample,boundary_correction=None,**correction_kwargs):

        dx2 = np.array(x_point - x_sample) ** 2*self.h**-2
        dy2 = np.array(y_point - y_sample) ** 2*self.h**-2

        func = np.exp(-0.5*(dx2 + dy2))

        norm = 1

        if boundary_correction == 're_normlize':

            def integrand(d_y,d_x,h):
                return np.exp(-0.5*(d_y**2+d_x**2)*h**-2)

            ylow,yhigh,xlow,xhigh = correction_kwargs['ymin'],correction_kwargs['ymax'],correction_kwargs['xmin'],correction_kwargs['xmax']

            norm = dblquad(integrand,xlow,xhigh,ylow,yhigh,args=(self.h))[0]
            print norm
            a=input('continue')

        return func*norm**-1

    def scotts_factor(self,n,d=2):

        return n ** (-1. / (d + 4))

