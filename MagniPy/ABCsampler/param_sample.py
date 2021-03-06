import numpy as np
from MagniPy.ABCsampler.probability_distributions import ProbabilityDistribution

class ParamSample(object):

    recognized_param_precision = {}
    recognized_param_precision['a0_area'] = 5
    recognized_param_precision['sigma_sub'] = 5
    recognized_param_precision['f_sub'] = 7
    recognized_param_precision['log_f_sub'] = 2
    recognized_param_precision['logmhm'] = 3
    recognized_param_precision['SIE_gamma'] = 4
    recognized_param_precision['core_ratio'] = 4
    recognized_param_precision['source_size_kpc'] = 6
    recognized_param_precision['SIDMcross'] = 3
    recognized_param_precision['lens_redshift'] = 3

    def __init__(self, params_to_vary = {}, Nsamples=int,decimals=5,**kwargs):
        """

        :param params_to_vary (list): list of strings corresponding to parameters to vary
        :param param_vary_args: list of dictionaries specifying the prior for each parameter
        :param Nsamples: number of samples per parameter
        """

        self.Nsamples = Nsamples

        self.dimension = len(params_to_vary.keys())

        self.priors = []

        self.param_names = params_to_vary.keys()

        for pname in params_to_vary.keys():
            pargs = params_to_vary[pname]
            if 'decimals' in pargs:
                decimals = pargs['decimals']
            elif pname in self.recognized_param_precision.keys():
                decimals = self.recognized_param_precision[pname]

            if pargs['prior_type'] == 'initialized':
                self.priors.append(
                    ProbabilityDistribution(distribution_type=pargs['prior_type'], args=pargs, decimals=decimals,macromodel=kwargs['macromodel']))
            else:
                self.priors.append(ProbabilityDistribution(distribution_type=pargs['prior_type'],args=pargs,decimals=decimals))

    def sample(self,scale_by='dimension'):

        if scale_by == 'dimension':

            if self.dimension==1:

                samples = self.sample_d1()

            elif self.dimension==2:

                samples = self._sample_d2()

            else:
                raise Exception('cannot handle full sampling in more than 2 dimensions.')

            return np.array(samples)

        elif scale_by=='Nsamples':

            samples = np.zeros((self.Nsamples,self.dimension))

            for d in range(0,self.dimension):
                samples[:,d] = self.priors[d].draw(self.Nsamples)

            return np.array(samples)

    def _sample_d1(self,index=0):

        return self.priors[index].draw(self.Nsamples)

    def _sample_d2(self):

        samples = np.zeros((self.Nsamples**2, 2))

        parameters = self._sample_d1()

        for i, param in enumerate(parameters):

            newsamples = self._sample_d1(index=1)

            samples[i * self.Nsamples:(i + 1) * self.Nsamples, 0] = param
            samples[i * self.Nsamples:(i + 1) * self.Nsamples, 1] = newsamples

        return samples



