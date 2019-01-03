import numpy as np
from MagniPy.Analysis.KDE.kernels import *
from scipy.stats import gaussian_kde
from scipy.signal import fftconvolve
from numpy.linalg import inv
from copy import deepcopy
from numpy.fft import fftshift, fft2, ifft2
from fastkde import fastKDE
from MagniPy.Analysis.KDE.getdist_kde import GetDistWrap

class KDE_getdist(object):

    def __init__(self, bandwidth_scale=1):

        self.bandwidth_scale = bandwidth_scale

        self.gdwrap = GetDistWrap()

    def __call__(self, data, pnames):

        kwargs = {'samples': data}

        estimate, xx, yy = self.gdwrap(param_names = pnames, kwargs=kwargs)

        return estimate, xx, yy


class KDE_fast(object):

    def __init__(self, dim):

        self.dimension = dim

    def __call__(self, data,X=None,Y=None,**kwargs):

        if self.dimension == 1:

            pdf, _ = fastKDE.pdf(data, axes = X, numPoints = len(X))

            return pdf

        elif self.dimension == 2:

            pdf, _ = fastKDE.pdf(data[:, 0], data[:, 1], numPoints = len(X), axes = [X, Y])

            xx, yy = np.meshgrid(X, Y)

            return pdf, xx.ravel(), yy.ravel()


class KDE_scipy(object):

    def __init__(self,dim):

        self.dimension = dim
        self._kde1d = KernelDensity1D(bandwidth_scale=1.05)
        self._kde2d = KernelDensity2D(bandwidth_scale=1)

    def __call__(self,data,X=None,Y=None,**kwargs):

        if self.dimension==1:

            #kernel = gaussian_kde(data.T)
            #positions = X

            density = self._kde1d(data, X, kwargs['pranges_true'])

            return density

        elif self.dimension == 2:

            h = np.shape(data)[0] ** (-1. / (6))
            data_cov = np.cov(np.array([data[:, 0], data[:, 1]]))
            hx = h * data_cov[0][0] ** 0.5
            hy = h * data_cov[1][1] ** 0.5

            #data = self.mirror(deepcopy(data), kwargs['pranges_true'], 2.5*hx, 2.5*hy)

            kernel = gaussian_kde(data.T)

            xx,yy = np.meshgrid(X,Y)

            positions = np.vstack([xx.ravel(),yy.ravel()])

            #density = kernel(positions).reshape(len(X), len(X))
            density = np.exp(kernel.logpdf(positions))
            #norm = self._kde2d.boundary_norm(data, len(xx), xx, yy, np.mean(X), np.mean(Y), kwargs['prior_weights'])
            #density *= norm ** -1

            return density,xx.ravel(),yy.ravel()

    def mirror(self, data, pranges, hx, hy):

        new_data = deepcopy(data)
        for row in range(0, int(data.shape[0])):

            new = self._flip(pranges[0], pranges[1], data[row][0], data[row][1], hx, hy)

            if new is not None:

                new_data = np.vstack((new_data, new))

        return new_data

    def _flip(self, prangex, prangey, coordx, coordy, dx, dy):

        condx1, condx2 = np.absolute(coordx - prangex[0]) < dx, np.absolute(coordx - prangex[1]) < dx
        condy1, condy2 = np.absolute(coordy - prangey[0]) < dy, np.absolute(coordy - prangey[1]) < dy

        if condx1 or condx2 or condy1 or condy2:
            new = np.array([coordx, coordy])

            if condx1:
                new[0] = self._flip_down(coordx, prangex[0])
            elif condx2:
                new[0] = self._flip_up(coordx, prangex[1])
            if condy1:
                new[1] = self._flip_down(coordy, prangey[0])
            elif condy2:
                new[1] = self._flip_up(coordy, prangey[1])

            return new
        else:
            return None


    def _flip_down(self, coord, b):

        dx = coord - b

        return coord - 2 * dx

    def _flip_up(self, coord, b):

        dx = b - coord

        return coord + 2 * dx


class KernelDensity1D(object):

    def __init__(self, scale = 1, bandwidth_scale=1, kernel='Gaussian', reweight = True):

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

        return estimate

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

    def __init__(self, bandwidth_scale=1):
        self.bandwidth_scale = bandwidth_scale

    def _scotts_factor(self, n, d=2):
        return n ** (-1. / (d + 4))

    def _boundary_kernel(self, xx, nbins):
        B = np.ones_like(xx)
        return B
        return np.pad(B, 0, mode='constant')

    def _gaussian_kernel(self, hx, hy, xx, xc, yy, yc):
        r = -0.5 * ((xx - xc) ** 2 * hx ** -2 + (yy - yc) ** 2 * hy ** -2)

        return np.exp(r)

    def _convolve_2d(self, data, nbins, xpoints, ypoints, xc, yc, prior_weights, boundary_correction):

        #xmin, xmax = xpoints[0], xpoints[-1]
        #ymin, ymax = ypoints[0], ypoints[-1]
        xbins, ybins = np.linspace(xpoints[0], xpoints[-1], nbins), np.linspace(ypoints[0], ypoints[-1], nbins)
        dx, dy = (xbins[1] - xbins[0]), (ybins[1] - ybins[0])
        #dx, dy = 0, 0

        hb, _, _ = np.histogram2d(data[:, 0], data[:, 1], bins = (xbins - dx, ybins+dy),weights=prior_weights)

        xbins = np.linspace(xpoints[0], xpoints[-1], len(hb))
        ybins = np.linspace(ypoints[0], ypoints[-1], len(hb))

        xx, yy = np.meshgrid(xbins, ybins)

        h = self._scotts_factor(n=nbins) * self.bandwidth_scale
        data_cov = np.cov(np.array([data[:, 0], data[:, 1]]))

        hx = h * data_cov[0][0] ** 0.5
        hy = h * data_cov[1][1] ** 0.5

        density = fftconvolve(hb.T, self._gaussian_kernel(hx, hy, xx, xc, yy, yc), mode='same')

        #density = density.reshape(L,L)

        if boundary_correction:
            boundary_norm = fftconvolve(self._gaussian_kernel(hx, hy, xx, xc, yy, yc),
                                        self._boundary_kernel(xx, nbins), mode='same')
            #boundary_norm = boundary_norm[0:50,0:50]
            #boundary_norm.reshape(L,L)
            density *= boundary_norm ** -1

        return density

    def __call__(self, data, xpoints, ypoints, pranges, prior_weights, boundary_correction=True):

        assert len(xpoints) == len(ypoints)

        xx, yy = np.meshgrid(xpoints, ypoints)

        x_center, y_center = np.mean(pranges[0]), np.mean(pranges[1])
        estimate = self._convolve_2d(data, len(xpoints), xpoints, ypoints,
                                     x_center, y_center, prior_weights, boundary_correction)

        return estimate, xx.ravel(), yy.ravel()


class KernelDensity2D_old(object):

    def __init__(self,reweight=True,scale=5,bandwidth_scale=1,kernel='Gaussian'):

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

    def _kernel(self, xsamples, ysamples, pranges_true=None, prior_weights=None):

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
                boundary_weight = self.boundary.renormalize(xi, yi, hx, hy, pranges_true)

                weight = prior_weights[i]*boundary_weight
            else:
                weight = prior_weights[i]

            functions.append(self._kernel_function(xi, yi, weight=weight, covmat=covmat))

        return functions

    def _scotts_factor(self, n, d=2):

        return n ** (-1. / (d + 4))

    def __call__(self, data, xpoints, ypoints, pranges_true, prior_weights=None):

        assert len(xpoints) == len(ypoints)

        xx, yy = np.meshgrid(xpoints, ypoints)
        xx, yy = xx.ravel(), yy.ravel()

        self.boundary = Boundary2D(scale=self.scale)

        functions = self._kernel(data[:, 0], data[:, 1], pranges_true, prior_weights)

        xy = list(zip(xx, yy))

        estimate = np.zeros(len(xy))

        for i, coord in enumerate(xy):

            for func in functions:
                estimate[i] += func(*xy[i])

        return estimate, xx, yy
