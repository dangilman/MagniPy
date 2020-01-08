import numpy as np
from MagniPy.Analysis.KDE.kde import KDE_nD

class IndepdendentDensities(object):

    def __init__(self, density_samples_list):

        """

        :param density_samples_list: a list of DensitySamples instances
        """
        if not isinstance(density_samples_list, list):
            raise Exception('must pass in a list of DensitySamples instances.')
        self.densities = density_samples_list

    @property
    def product(self):

        prod = 1
        for di in self.densities:
            prod *= di.averaged
        return prod * np.max(prod) ** -1

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

class DensitySamplesNew(object):

    def __init__(self, data_list, param_names, param_ranges, weight_list, bwidth_scale=0.7,
                 nbins=12, use_kde=False, from_file=False):

        self.single_densities = []

        self._n = 0

        for j, data in enumerate(data_list):
            self._n += 1
            if from_file is not False:

                density = np.load(from_file+'_'+str(j+1)+'.npy')

            else:
                density = None

            w = self._weights(weight_list, j)
            print(w.shape, data.shape)
            self.single_densities.append(SingleDensity(data, param_names, param_ranges, w,
                                                       bwidth_scale, nbins, use_kde, density=density))

    def _weights(self, wlist, idx):

        N = len(wlist)
        w = 1
        for n in range(0, N):
            w *= wlist[n][idx]
        return w

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

    @property
    def averaged(self):
        avg = 0
        for di in self.single_densities:
            avg += di.density
        return avg

    def save_to_file(self, fname):

        fname_base = fname + '_'
        for i, single_density in enumerate(self.single_densities):
            np.save(fname_base+str(i+1), single_density.density)
            #with open(fname_base+str(i+1)+'.txt', 'w') as f:
            #    f.write('np.'+str(repr(single_density.density)))



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

class DensitySamples(object):

    def __init__(self, data_list, param_names, weight_list, param_ranges=None, bwidth_scale=0.7,
                 nbins=12, use_kde=False, from_file=False, param_ranges_compute_idx=0, samples_width_scale=4):

        self.single_densities = []

        self._n = 0

        if weight_list is None:
            weight_list = [None] * len(data_list)

        if param_ranges is None:
            self.param_ranges = self._compute_param_ranges(data_list[param_ranges_compute_idx], samples_width_scale)
        else:
            self.param_ranges = param_ranges

        for j, (data, weights) in enumerate(zip(data_list, weight_list)):
            self._n += 1
            if from_file is not False:

                density = np.load(from_file+'_'+str(j+1)+'.npy')

            else:
                density = None

            self.single_densities.append(SingleDensity(data, param_names, self.param_ranges, weights,
                                                       bwidth_scale, nbins, use_kde, density=density))

    def _compute_param_ranges(self, data, scale):

        param_ranges = []

        for column_idx in range(0, int(data.shape[1])):

            data_mean = np.mean(data[:, column_idx])
            data_std = np.std(data[:, column_idx])
            data_low = data_mean - scale * data_std
            data_max = data_mean + scale * data_std
            print(data_std)
            data_low = max(data_low, min(data[:, column_idx]))
            data_max = min(data_max, max(data[:, column_idx]))

            param_ranges.append([data_low, data_max])

        return param_ranges

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

    @property
    def averaged(self):
        avg = 0
        for di in self.single_densities:
            avg += di.density
        return avg

    def save_to_file(self, fname):

        fname_base = fname + '_'
        for i, single_density in enumerate(self.single_densities):
            np.save(fname_base+str(i+1), single_density.density)
            #with open(fname_base+str(i+1)+'.txt', 'w') as f:
            #    f.write('np.'+str(repr(single_density.density)))

