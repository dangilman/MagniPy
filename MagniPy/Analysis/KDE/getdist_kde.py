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



