import numpy as np
from MagniPy.Analysis.KDE.kernels import *
from scipy.stats import gaussian_kde
from scipy.signal import fftconvolve
from copy import deepcopy

class KDE_nD(object):

    def __init__(self, bandwidth_scale=1, ranges=None, nbins=None):

        self.bandwidth_scale = bandwidth_scale

        self._ranges = ranges
        self._nbins = nbins

    def _scotts_factor(self, n, d):
        return 1.05 * n ** (-1. / (d + 4))

    def _boundary_kernel(self, shape, nbins):

        B = np.ones(shape)
        return B
        # return np.pad(B, nbins, mode='constant', constant_values=0)

    def _gaussian_kernel(self, inverse_cov_matrix, coords_centered, dimension, n_reshape):

        def _gauss(_x):
            return np.exp(-0.5 * np.dot(np.dot(_x, inverse_cov_matrix), _x))

        z = [_gauss(coord) for coord in coords_centered]

        return np.reshape(z, tuple([n_reshape] * dimension))

    def _get_coordinates(self, ranges=None, nbins=None):

        points = []

        if ranges is None:
            ranges = self._ranges
        if nbins is None:
            nbins = self._nbins

        for i in range(0, len(ranges)):
            points.append(np.linspace(ranges[i][0], ranges[i][1], self._nbins))
        return points

    def NDhistogram(self, data, weights):

        """

        :param data: data to make the histogram. Shape (nsamples, ndim)
        :param coordinates: np.linspace(min, max, nbins) for each dimension
        :param ranges: parameter ranges corresponding to columns in data
        :param weights: param weights
        :return:
        """

        coordinates = self._get_coordinates()

        histbins = []
        for i, coord in enumerate(coordinates):
            histbins.append(np.linspace(self._ranges[i][0], self._ranges[i][-1], len(coord) + 1))


        H, _ = np.histogramdd(data, range=self._ranges, bins=histbins, weights=weights)

        return H.T

    def _compute_ND(self, data, coordinates, ranges, weights, boundary_order=1):

        histbins = []

        X = np.meshgrid(*coordinates)
        cc_center = np.vstack([X[i].ravel() - np.mean(ranges[i]) for i in range(len(X))]).T

        dimension = int(np.shape(data)[1])
        h = self.bandwidth_scale * self._scotts_factor(len(coordinates[0]), dimension)

        for i, coord in enumerate(coordinates):
            histbins.append(np.linspace(ranges[i][0], ranges[i][-1], len(coord) + 1))

        H = self.NDhistogram(data, weights)

        covariance = h * np.cov(data.T)
        c_inv = np.linalg.inv(covariance)
        n = len(coordinates[0])
        gaussian_kernel = self._gaussian_kernel(c_inv, cc_center, dimension, n)

        density = fftconvolve(H, gaussian_kernel, mode='same')

        if boundary_order == 1:
            boundary_kernel = self._boundary_kernel(shape=np.shape(H), nbins=np.shape(H)[0])
            boundary_normalization = fftconvolve(gaussian_kernel, boundary_kernel, mode='same')

            density *= boundary_normalization ** -1

        return density

    def __call__(self, data, ranges, weights=None, boundary_order = 1):

        """

        :param data: data to make the histogram. Shape (nsamples, ndim)
        :param points: np.linspace(min, max, nbins) for each dimension
        :param ranges: parameter ranges corresponding to columns in data
        :param weights: param weights
        :return:
        """

        points = self._get_coordinates(ranges, len(ranges[0]))

        density = self._compute_ND(data, points, ranges, weights,
                                   boundary_order=boundary_order)

        return density

