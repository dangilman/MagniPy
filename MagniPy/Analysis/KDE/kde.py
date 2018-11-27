import numpy as np
from MagniPy.Analysis.KDE.kernels import *
from scipy.stats import gaussian_kde
from scipy.signal import fftconvolve
from numpy.linalg import inv

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

class KernelDensity1D(object):

    def __init__(self, scale = 1, bandwidth_scale=1,kernel='Gaussian', reweight = True):

        self._kernel_function = Gaussian1d
        self.scale = scale
        self.bandwidth_scale = bandwidth_scale
        self.reweight = reweight

    def _scotts_factor(self, n, d=1):

        return n ** (-1. / (d + 4))

    def _convolve_1d(self, data, nbins, xx, xc, prior_weights):

        hb, _ = np.histogram(data, bins=nbins, weights=prior_weights)

        h = self._scotts_factor(n=nbins) * self.bandwidth_scale
        sigma = h * np.cov(data) ** 0.5

        def _boundary_func():
            B = np.ones_like(xx)
            return np.pad(B, nbins, mode='constant')

        def _kernel_func():

            r = -0.5 * ((xx - xc) ** 2 * sigma ** -2)

            return np.exp(r).T

        fft1 = fftconvolve(hb.T, _kernel_func(), mode='same')

        fft2 = fftconvolve(_kernel_func(), _boundary_func(), mode='same')

        return fft1 * (fft2) ** -1

    def __call__(self, data, xpoints, pranges_true, prior_weights=None):

        xx = xpoints
        estimate = self._convolve_1d(data, len(xx), xx, np.mean(pranges_true[0]), prior_weights)

        return estimate, xpoints

class KernelDensity1D_old(object):

    def __init__(self, scale = 1, bandwidth_scale=1,kernel='Gaussian', reweight = True):

        self._kernel_function = Gaussian1d
        self.scale = scale
        self.bandwidth_scale = bandwidth_scale
        self.reweight = reweight

    def _kernel(self, samples, pranges, prior_weights):

        h = self._scotts_factor(n=len(samples)) * self.bandwidth_scale
        sigma = h * np.cov(samples) ** 0.5
        functions = []

        if prior_weights is None:
            prior_weights = np.ones(len(samples))

        for i in range(0, len(samples)):

            if self.reweight:

                boundary_weight = self.boundary_1d.renormalize(samples[i], sigma, pranges)
            else:
                boundary_weight = 1

            functions.append(self._kernel_function(samples[i], weight=boundary_weight*prior_weights[i],
                                                   width=sigma))

        return functions

    def _scotts_factor(self, n, d=1):

        return n ** (-1. / (d + 4))

    def __call__(self, data, xpoints, pranges_true, prior_weights=None):

        self.boundary_1d = Boundary1D(scale=self.scale)

        functions = self._kernel(data, pranges_true, prior_weights)

        estimate = np.zeros(len(xpoints))

        for i, coord in enumerate(xpoints):

            for func in functions:
                estimate[i] += func(coord)

        return estimate, xpoints

class KernelDensity2D(object):

    def __init__(self,reweight=False,scale=5,bandwidth_scale=1,kernel='Gaussian'):

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

    def _scotts_factor(self, n, d=2):

        return n ** (-1. / (d + 4))

    def _convolve_2d(self, data, nbins, xx, yy, xc, yc, prior_weights):

        hb, _, _ = np.histogram2d(data[:,0], data[:,1], bins=nbins, weights=prior_weights)

        h = self._scotts_factor(n=nbins) * self.bandwidth_scale
        data_cov = np.cov(np.array([data[:,0], data[:,1]]))

        hx = h * data_cov[0][0] ** 0.5
        hy = h * data_cov[1][1] ** 0.5

        def _boundary_func():

            B = np.ones_like(xx)
            return np.pad(B, nbins, mode='constant')

        def _kernel_func():

            r = -0.5*((xx-xc)**2*hx**-2 + (yy-yc)**2*hy**-2)

            return np.exp(r).T

        def _kernel_func_xi():

            r = -0.5*((xx-xc)**2*hx**-2 + (yy-yc)**2*hy**-2)

            return xx*np.exp(r).T

        def _kernel_func_yi():

            r = -0.5*((xx-xc)**2*hx**-2 + (yy-yc)**2*hy**-2)

            return yy*np.exp(r).T

        def _kernel_func_xyi():

            r = -0.5*((xx-xc)**2*hx**-2 + (yy-yc)**2*hy**-2)

            return xx*yy*np.exp(r).T

        W0 = fftconvolve(_kernel_func(), _boundary_func(), mode='same').reshape(nbins, nbins)
        #W1i = fftconvolve(_kernel_func_xi(), _boundary_func(), mode='same').reshape(nbins, nbins)
        #W1j = fftconvolve(_kernel_func_yi(), _boundary_func(), mode='same').reshape(nbins, nbins)
        #W2ij = fftconvolve(_kernel_func_xyi(), _boundary_func(), mode='same').reshape(nbins, nbins)
        #W2_inv = inv(W2ij)

        #A0 = (W0 - W1i*W2ij*W1j)**-1
        #A1 = (-W2_inv*W1j) * A0

        fx = fftconvolve(hb.T, _kernel_func(), mode='same').reshape(nbins, nbins)
        #norm = (W0*A0 + A1 * W1i) ** -1
        return fx*W0**-1

    def __call__(self, data, xpoints, ypoints, pranges_true, prior_weights=None):

        assert len(xpoints) == len(ypoints)

        xx, yy = np.meshgrid(xpoints, ypoints)

        self.boundary = Boundary2D(scale=self.scale)

        estimate = self._convolve_2d(data, len(xpoints), xx, yy,
                                     np.mean(pranges_true[0]), np.mean(pranges_true[1]), prior_weights)

        return estimate, xx.ravel(), yy.ravel()

