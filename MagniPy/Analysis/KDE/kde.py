import numpy as np
from kernels import *
from scipy.stats import gaussian_kde


class KDE_scipy:

    def __init__(self,dim):

        self.dimension = dim

    def __call__(self,data,X=None,Y=None,**kwargs):

        if self.dimension==1:

            kernel = gaussian_kde(data.T)

            positions = X

            return kernel(positions)

        elif self.dimension == 2:

            if 'bw_method' in kwargs:
                kernel = gaussian_kde(data.T,bw_method=kwargs['bw_method'])
            else:
                kernel = gaussian_kde(data.T)

            xx,yy = np.meshgrid(X,Y)

            positions = np.vstack([xx.ravel(),yy.ravel()])

            return kernel(positions).reshape(len(X),len(X)),xx.ravel(),yy.ravel()

        else:

            kernel = gaussian_kde(data)

            coords = np.meshgrid(*kwargs['params'])

            positions = np.vstack([coords[i].ravel() for i in range(0,len(coords))])

            return kernel(positions)

class KernelDensity:

    def __init__(self,reweight=False,scale=1,bandwidth_scale=1,kernel='Gaussian'):

        self.reweight = reweight
        self.scale = scale
        self.bandwidth_scale = bandwidth_scale

        if kernel == 'Gaussian':
            self._kernel_function = Gaussian2d
        elif kernel == 'Silverman':
            self._kernel_function = Silverman2d
        elif kernel == 'Epanechnikov':
            self._kernel_function = Epanechnikov
            self.bandwidth_scale = 2.5
        elif kernel == 'Sigmoid':
            self._kernel_function = Sigmoid
        else:
            raise Exception('Kernel function not recognized.')

    def _kernel(self, xsamples, ysamples, pranges=None, prior_weights=None):

        h = self._scotts_factor(n=len(xsamples)) * self.bandwidth_scale

        functions = []

        data_cov = np.cov(np.array([xsamples, ysamples]))

        hx = h*data_cov[0][0]**0.5
        hy = h*data_cov[1][1]**0.5

        covmat = [[hx ** -2, 0], [0, hy ** -2]]

        if prior_weights is None:
            prior_weights = np.ones(len(xsamples))

        for i in range(0, len(xsamples)):
            xi, yi = xsamples[i], ysamples[i]

            if self.reweight:
                weight = prior_weights[i]*self.boundary.renormalize(xi, yi, hx, hy, pranges)
            else:
                weight = prior_weights[i]

            functions.append(self._kernel_function(xi, yi, weight=weight, covmat=covmat))

        return functions

    def _scotts_factor(self, n, d=2):

        return n ** (-1. / (d + 4))

    def __call__(self, data, xpoints, ypoints, pranges_training, prior_weights=None):

        assert len(xpoints) == len(ypoints)

        xx, yy = np.meshgrid(xpoints, ypoints)
        xx, yy = xx.ravel(), yy.ravel()

        self.boundary = Boundary(scale=self.scale)

        functions = self._kernel(data[:, 0], data[:, 1], pranges_training, prior_weights)

        xy = zip(xx,yy)

        estimate = np.zeros(len(xy))

        for i, coord in enumerate(xy):

            for func in functions:

                estimate[i] += func(*xy[i])

        return estimate,xx,yy
