import numpy as np
from getdist import plots, MCSamples
import warnings
warnings.filterwarnings("ignore")

class GetDistWrap(object):

    def __init__(self, posteriors_list=[None],
                 pnames=['a0_area', 'log_m_break', 'source_size_kpc', 'LOS_normalization', 'SIE_gamma'], tol=2000):

        self._tol = tol
        self.nlenses = len(posteriors_list)
        self.pnames = pnames
        self.pranges = self._set_pranges(pnames)
        self.single_lenses = []
        self.lens_posteriors = self._build(posteriors_list)
        self.param_likelihood = {}

    def __call__(self, param_names=None, kwargs={}):

        if param_names is None:
            param_names = self.pnames

        for i, post in enumerate(self.lens_posteriors):

            wi, xx, yy = post.get_averaged_density(param_names, **kwargs)
            L = int(wi.shape[0]**0.5)
            wi = wi.reshape(L,L)

        return wi, xx, yy

    def evaluate_p1_p2(self, p1, p2):

        for i, post in enumerate(self.lens_posteriors):
            wi, xx, yy = post.get_averaged_density_p1_p2(p1, p2, self._tol)

            if i == 0:
                weights = wi
            else:
                weights *= wi
        L = int(len(xx) ** 0.5)
        return weights.reshape(L, L), xx, yy

    def _build(self, posteriors):

        single_lenses = []
        for n in range(self.nlenses):
            single_lenses.append(SingleLensPosterior(posteriors[n], self.pnames, self.pranges, self._tol))
        return single_lenses

    def _set_pranges(self, pnames):

        ranges = {}
        for name in pnames:
            if name == 'a0_area':
                ranges.update({name: [0, 0.045]})
            elif name == 'log_m_break':
                ranges.update({name: [4.8, 10]})
            elif name == 'source_size_kpc':
                ranges.update({name: [0.025, 0.05]})
            elif name == 'SIE_gamma':
                ranges.update({name: [2, 2.2]})
            elif name == 'LOS_normalization':
                ranges.update({name: [0.7, 1.3]})
        return ranges


class SingleLensPosterior(object):

    def __init__(self, lens_posteriors, pnames, pranges, nsamps):

        if lens_posteriors is None:
            self._n_real = 1
        else:
            self._n_real = len(lens_posteriors)
        self.posterior_list = lens_posteriors
        self.pnames = pnames
        self.pranges = pranges
        self._nsamps = nsamps

        self._xycoords = self._build_coords(pranges, pnames)

    def get_averaged_density(self, pnames, **kwargs):

        if len(pnames) == 2:
            return self.get_averaged_density_p1_p2(pnames[0], pnames[1], **kwargs)
        else:
            raise Exception('not yet implemented')

    def get_averaged_density_p1_p2(self, p1, p2, tol=None, samples = None):

        names = [p1, p2]
        mean_weights = 0
        xx, yy = np.meshgrid(self._xycoords[p1], self._xycoords[p2])
        xcoords, ycoords = xx.ravel(), yy.ravel()

        for n in range(self._n_real):
            if samples is None:
                samples = self._posterior_to_samples(self.posterior_list[n], names, tol)
            mcsamples = self._samples_to_MCsamples(samples, names)
            mean_weights += self._mcsamples_to_densities_p1_p2(mcsamples, names, xcoords, ycoords)

        return mean_weights * self._n_real ** -1, xcoords, ycoords

    def _build_coords(self, pranges, pnames, nsteps=20):

        xycoords = {}

        for pname in pnames:
            xycoords.update({pname: np.linspace(pranges[pname][0], pranges[pname][1], nsteps)})

        return xycoords

    def _posterior_to_samples(self, post, pnames, nsamples):

        samples = np.empty(shape=(nsamples, len(pnames)))
        for i, pname in enumerate(pnames):
            samples[:, i] = post.samples[pname]

        return samples

    def _samples_to_MCsamples(self, samples, pnames):
        ranges = []
        for p in pnames:
            ranges.append(self.pranges[p])

        mcsamples = MCSamples(samples=samples, names=pnames, ranges=ranges)

        return mcsamples

    def _mcsamples_to_densities(self, mcsamples, pnames, coords):

        density = mcsamples.get2DDensity(pnames, normalized=True, boundary_correction_order = 0)

        return density.Prob(*coords)

    def _mcsamples_to_densities_p1_p2(self, mcsamples, pnames, xx, yy):

        density = mcsamples.get2DDensity(x=pnames[0], y=pnames[1], normalized=True)

        return density.Prob(xx, yy)
