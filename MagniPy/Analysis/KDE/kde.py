import numpy as np
from methods import *
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from getdist.mcsamples import *
from getdist import plots


class KDE_scipy:

    def __init__(self,p1_range,p2_range,steps=100):

        self.X, self.Y = np.linspace(p1_range[0],p1_range[1],steps),np.linspace(p2_range[0],p2_range[1],steps)

    def density(self,data):

        kernel = gaussian_kde(data.T)
        xx,yy = np.meshgrid(self.X,self.Y)
        positions = np.vstack([xx.ravel(),yy.ravel()])

        return kernel(positions).reshape(len(self.X),len(self.X))

class KDE:

    def __init__(self,p1_range,p2_range,steps=100,boundary_correction_order=1,dim=2):

        self.pranges = {'x':[p1_range[0],p1_range[1]],'y':[p2_range[0],p2_range[1]]}
        self.steps = steps
        self.xmin,self.xmax = p1_range[0],p1_range[1]
        self.ymin,self.ymax = p2_range[0],p2_range[1]
        self.boundary_correction_order = boundary_correction_order
        self.dim = dim

    def density(self,data,steps=100):

        self.kde_getdist = MCSamples(samples=data, names=['x', 'y'], ranges=self.pranges)

        if self.dim ==1:

            density = self.kde_getdist.get1DDensityGridData('x', get_density=True,
                                                            boundary_correction_order=self.boundary_correction_order)

            xvals = np.linspace(self.xmin, self.xmax, self.steps)


            return density.Prob(xvals)

        elif self.dim == 2:
            density = self.kde_getdist.get2DDensityGridData('x','y',get_density=True,boundary_correction_order=self.boundary_correction_order)

            xvals = np.linspace(self.xmin,self.xmax,self.steps)
            yvals = np.linspace(self.ymin,self.ymax,self.steps)
            xx,yy = np.meshgrid(xvals,yvals)

            return density.Prob(xx,yy)

