import numpy as np
from methods import *
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from getdist.mcsamples import *
from math import erf

class KDE_scipy:

    def __init__(self,p1_range,p2_range=None,steps=100):

        if p2_range is not None:
            self.X, self.Y = np.linspace(p1_range[0],p1_range[1],steps),np.linspace(p2_range[0],p2_range[1],steps)
            self.dimension = 2
        else:
            self.X = np.linspace(p1_range[0],p1_range[1],steps)
            self.dimension = 1

    def boundary_correction(self,data,kernel_size,xrange,yrange=None):

        factor = 2

        if self.dimension==1:

            dis_low = (data - xrange[0]) * (kernel_size*factor)**-1
            dis_high = (xrange[1] - data) * (kernel_size*factor)**-1
            weights_low = np.ones_like(data)
            indslow = np.where(dis_low<1)
            weights_low[indslow] = erf(dis_low[indslow])**-1

            indshigh = np.where(dis_high<1)
            weights_high = np.ones_like(data)
            weights_high[indshigh] = erf(dis_high)**-1

            data*=weights_high*weights_low

        else:
            raise ValueError('not yet implemented')

        return data

    def density(self,data):

        if self.dimension==1:

            kernel = gaussian_kde(data.T)

            positions = self.X

            return kernel(positions)

        elif self.dimension == 2:
            kernel = gaussian_kde(data.T)
            xx,yy = np.meshgrid(self.X,self.Y)
            positions = np.vstack([xx.ravel(),yy.ravel()])

            return kernel(positions).reshape(len(self.X),len(self.X))

class KDE:

    def __init__(self,p1_range,p2_range=None,steps=100,boundary_correction_order=1,dim=2,weights=None):


        self.steps = steps
        self.xmin,self.xmax = p1_range[0],p1_range[1]
        self.pranges = {'x': [p1_range[0], p1_range[1]]}

        if p2_range is not None:
            self.pranges.update({'y': [p2_range[0], p2_range[1]]})
            self.ymin,self.ymax = p2_range[0],p2_range[1]

        self.boundary_correction_order = boundary_correction_order
        self.dim = dim
        self.weights = None

    def density(self,data,steps=100):

        if self.dim ==1:

            self.kde_getdist = MCSamples(samples=data, names=['x'], ranges=self.pranges)

            density = self.kde_getdist.get1DDensityGridData('x', get_density=True,
                                                            boundary_correction_order=self.boundary_correction_order)

            xvals = np.linspace(self.xmin, self.xmax, self.steps)

            return density.Prob(xvals),xvals,_

        elif self.dim == 2:

            self.kde_getdist = MCSamples(samples=data, names=['x', 'y'], ranges=self.pranges)

            density = self.kde_getdist.get2DDensityGridData('x','y',get_density=True,boundary_correction_order=self.boundary_correction_order)

            xvals = np.linspace(self.xmin,self.xmax,self.steps)
            yvals = np.linspace(self.ymin,self.ymax,self.steps)
            xx,yy = np.meshgrid(xvals,yvals)

            return density.Prob(xx,yy),xvals,yvals

