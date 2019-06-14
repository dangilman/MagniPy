import numpy as np
from MagniPy.Analysis.KDE.kde import KDE_nD

class IndepdendentDensities(object):

    def __init__(self, density_samples_list):

        """

        :param density_samples_list: a list of DensitySamples instances
        """
        self.densities = density_samples_list

    def projection_1D(self, pname):

        proj = 1
        for den in self.densities:
            proj *= den.projection_1D(pname)
        return proj * np.max(proj) ** -1

    def projection_2D(self, p1, p2):

        proj = 1
        for den in self.densities:
            proj *= den.projection_2D(p1, p2)
        return proj * np.max(proj) ** -1

class DensitySamples(object):

    def __init__(self, data_list, param_names, param_ranges, weight_list, bwidth_scale=0.7,
                 nbins=12, use_kde=False, from_file=False):

        self.single_densities = []

        self._n = 0

        if weight_list is None:
            weight_list = [None] * len(data_list)

        for j, (data, weights) in enumerate(zip(data_list, weight_list)):
            self._n += 1
            if from_file is not False:
                density = np.loadtxt(from_file[j])
            else:
                density = None
            self.single_densities.append(SingleDensity(data, param_names, param_ranges, weights,
                                                       bwidth_scale, nbins, use_kde, density=density))

    def projection_1D(self, pname):

        proj = 0
        for den in self.single_densities:
            proj += den.projection_1D(pname)
        return proj * np.max(proj) ** -1

    def projection_2D(self, p1, p2):

        proj = 0
        for den in self.single_densities:
            proj += den.projection_2D(p1, p2)
        return proj

class SingleDensity(object):

    def __init__(self, data, param_names, param_ranges, weights, bwidth_scale, nbins,kde, density):

        self.param_names = param_names
        self.param_range_list = param_ranges

        kernel = KDE_nD(bandwidth_scale=bwidth_scale, ranges=param_ranges, nbins=nbins)

        if density is not None:
            self.density = density
        else:
            if kde:
                self.density = kernel(data, param_ranges, weights)
            else:
                self.density = kernel.NDhistogram(data, weights)

    def projection_1D(self, pname):

        sum_inds = []
        if pname not in self.param_names:
            raise Exception('no param named '+pname)
        for i, name in enumerate(self.param_names):
            if pname != name:
                sum_inds.append(len(self.param_names) - (i + 1))

        projection = np.sum(self.density, tuple(sum_inds))

        return projection


    def projection_2D(self, p1, p2):

        if p1 not in self.param_names or p2 not in self.param_names:
            raise Exception(p1 + ' or '+ p2 + ' not in '+str(self.param_names))
        sum_inds = []
        for i, name in enumerate(self.param_names):
            if p1 != name and p2 != name:
                sum_inds.append(len(self.param_names) - (i + 1))

        tpose = False
        for name in self.param_names:
            if name == p1:
                break
            elif name == p2:
                tpose = True
                break

        projection = np.sum(self.density, tuple(sum_inds))
        if tpose:
            projection = projection.T

        return projection