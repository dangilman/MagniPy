import pandas
from MagniPy.ABCsampler.ChainOps import *
import numpy as numpy

class ChainFromChain(object):

    def __init__(self, chain, indicies, load_flux = False):

        self._chain_parent = chain
        self.params_varied = chain.params_varied
        self.truths = chain.truths
        self.prior_info = chain.prior_info
        self.pranges = chain.pranges
        self.Nlenses = chain.Nlenses
        self.pranges_trimmed = chain.pranges_trimmed

        self.lenses = []
        self._add_lenses(indicies, load_flux)

    def _add_lenses(self, indicies, load_flux):

        for ind in indicies:

            new_lens = deepcopy(self._chain_parent.lenses[ind])

            if len(new_lens._fluxes) == 0 and load_flux:
                new_lens._load_sim(ind, self._chain_parent.params_varied)

            self.lenses.append(new_lens)

    def get_posteriors(self,tol=None, reject_pnames = None, keep_ranges = None):

        posteriors = []

        for lens in self.lenses:

            lens.draw(tol, reject_pnames, keep_ranges)

            posteriors.append(lens.posterior)

        return posteriors

    def _add_perturbations(self, error, L):

        for i, lens in enumerate(self.lenses):
            #print('computing lens '+str(i)+'.... ')
            if error == 0:

                new_statistic = lens.statistic

            else:

                new_obs = lens._fluxes_obs[0] + np.random.normal(0, float(error)*lens._fluxes_obs[0])
                new_fluxes = lens._fluxes[0] + np.random.normal(0, float(error)*lens._fluxes[0])

                new_obs_ratio = new_obs*new_obs[0]**-1

                norm = deepcopy(new_fluxes[:, 0])

                for col in range(0, 4):
                    new_fluxes[:, col] *= norm ** -1

                perturbed_ratios = new_fluxes[:, 1:]
                diff = np.array((perturbed_ratios - new_obs_ratio[1:]) ** 2)
                summary_statistic = np.sqrt(np.sum(diff, 1))

                ordered_inds = np.argsort(summary_statistic)[0:L]

                new_statistic = summary_statistic[ordered_inds]

                lens.add_parameters(fname=chainpath_out+'processed_chains/'+
                                  self._chain_parent._chain_name+'/lens'+str(i+1)+'/samples.txt',
                                    use_pandas=False)

                for pname in lens._parameters[0].keys():
                    lens.parameters[0][pname] = lens._parameters[0][pname][ordered_inds]

            lens.statistic = [new_statistic]

class ChainFromSamples(object):

    def __init__(self,chain_name='',which_lens=None, error=0,
                deplete = False, deplete_fac = 0.5, n_pert = 1, load = True):

        self.params_varied, self.truths, self.prior_info = read_chain_info(chainpath_out + '/processed_chains/' +
                                                                           chain_name + '/simulation_info.txt')
        if 'logmhm' in self.truths:
            self.truths.update({'log_m_break':self.truths['logmhm']})

        self.pranges = self.get_pranges(self.prior_info)
        self.pranges_trimmed = self.pranges

        self._chain_name = chain_name

        Ncores,cores_per_lens,self.Nlenses = read_run_partition(chainpath_out + '/processed_chains/' +
                                                                    chain_name + '/simulation_info.txt')

        self.chain_file_path = chainpath_out + 'chain_stats/' + chain_name +'/'

        self.lenses = []

        for ind in which_lens:

            new_lens = SingleLens(zlens = None, zsource=None, flux_path=chainpath_out + '/processed_chains/' +
                                                                    chain_name+'/')
            if load:
                for ni in range(0,n_pert):
                    #print('loading '+str(ni+1)+'...')
                    fname = 'statistic_'+str(error)+'error_'+str(ni+1)+'.txt'

                    _ = new_lens.add_statistic(fname=self.chain_file_path + 'lens' + str(ind) + '/'+fname)

                    fname = 'params_' + str(error) + 'error_' + str(ni+1) + '.txt'

                    new_lens.add_parameters(fname=self.chain_file_path+'lens'+str(ind)+'/'+fname)

                    if deplete:

                        L = int(len(new_lens.statistic[ni]))
                        keep = np.arange(0, L)
                        u = np.random.rand(L)
                        keep = keep[np.where(u <= deplete_fac)]
                        for pname in new_lens.parameters[ni].keys():
                            new_lens.parameters[ni][pname] = new_lens.parameters[ni][pname][keep]
                        new_lens.statistic[ni] = new_lens.statistic[ni][keep]

            self.lenses.append(new_lens)

    def get_samples(self,tol=None, reject_pnames = None, keep_ranges = None):

        if len(self.lenses) == 1:

            samples = self.lenses[0].draw(tol)
            samples = samples[:, :, :, np.newaxis]

        else:

            L = len(self.lenses)
            temp = self.lenses[0].draw(tol)
            shape = temp.shape
            samples = np.empty(shape=(shape[0], shape[1], shape[2], L))

            for l, lens in enumerate(self.lenses):
                samps = lens.draw(tol)
                samples[:,:,:,l] = samps

        return samples

    def get_pranges(self,info):

        pranges = {}

        for keys in info.keys():
            pname = info[keys]
            if pname['prior_type'] == 'Gaussian':
                pranges[keys] = [float(pname['mean']) - float(pname['sigma'])*2,float(pname['mean'])+float(pname['sigma'])*2]
            elif pname['prior_type'] == 'Uniform':
                pranges[keys] = [float(pname['low']),float(pname['high'])]

        return pranges

    def load_param_names(self,fname):

        param_names = []

        with open(fname, 'r') as f:
            keys = f.readline().split(' ')
        for word in keys:
            if word not in ['#', '\n', '']:
                param_names.append(word)

        return param_names

    def import_observed(self,ind,error=0,index=1):

        fname = self.chain_file_path+'lens'+str(ind)+'/fluxratios/'+'observed_'+ str(int(error * 100)) + 'error_'+ str(index)+'.txt'

        return numpy.loadtxt(fname)

    def import_model(self,ind,error=0,index=1):

        fname = self.chain_file_path + 'lens' + str(ind) + '/fluxratios/' + 'model_' + str(
            int(error * 100)) + 'error_' + str(index) + '.txt'

        return numpy.loadtxt(fname)

    def re_weight(self,posteriors,weight_function,indexes=None):

        if indexes is None:

            for posterior in posteriors:

                for func in weight_function:
                    posterior.change_weights(func)

            return posteriors

        else:

            if isinstance(indexes,list):

                assert isinstance(weight_function,list),'If apply a prior on a lens by lens basis, must provide a ' \
                                                        'list of individual weight functions'

                for i,index in enumerate(indexes):
                    posteriors[index-1].change_weights(weight_function[i])

            return posteriors

class SingleLens(object):

    def __init__(self, zlens, zsource, weights=None, flux_path=''):

        self.weights = weights
        self.posterior = None
        self.zlens, self.zsource = zlens, zsource
        self.flux_path = flux_path
        self.parameters, self.statistic = None, []
        self._fluxes = []
        self._fluxes_obs = []
        self._parameters = []

    def _load_sim(self, idx, pnames):

        fluxes = np.array(pandas.read_csv(self.flux_path+'lens'+str(idx)+'/modelfluxes.txt',
                                   header=None,sep=" ",index_col=None).values.astype(float))

        self._fluxes.append(fluxes)
        fluxes_obs = np.loadtxt(self.flux_path+'lens'+str(idx)+'/observedfluxes.txt')
        self._fluxes_obs.append(fluxes_obs)

        parameters = np.loadtxt(self.flux_path+'lens'+str(idx)+'/samples.txt')
        new_dictionary = {}

        for i,pname in enumerate(pnames):

            new_dictionary.update({pname:parameters[:,i].astype(float)})

        self._parameters.append(new_dictionary)

    def draw(self, tol):

        return self.parameters[0:tol]

    def add_parameters(self,fname=None,use_pandas=False):

        if use_pandas:
            params = numpy.squeeze(pandas.read_csv(fname, header=None, sep=" ", index_col=None)).astype(numpy.ndarray)
        else:
            params = numpy.squeeze(np.loadtxt(fname))

        params = self._rescale(params)

        if self.parameters is None:
            self.parameters = params
            self.parameters = self.parameters[:, :, np.newaxis]

        else:
            self.parameters = np.dstack((self.parameters, params))

    def _rescale(self, params):

        if np.min(params[:,0]) < 25:
            params[:,0] *= 1000
        if np.min(params[:,-1]) < 0.045:
            params[:,-1] *= 100

        return params

    def add_weights(self,params,weight_function):
        """
        :param params: dictionary with param names and values
        :param weight_function: a function that takes the 'params' dictionary and returns a
        1d array of weight with same length as params
        :return:
        """
        self.weights = weight_function(params)

    def add_statistic(self,fname):

        statistic = numpy.squeeze(pandas.read_csv(fname, header=None, sep=" ", index_col=None))

        finite = numpy.where(numpy.isfinite(statistic))[0]

        new_stat = statistic[finite].astype(float)

        self.statistic.append(new_stat)

        return finite

class WeightedSamples:

    def __init__(self,params_to_weight=None,weight_args=None):

        if params_to_weight is None:
            self.functions = None
        else:
            self.functions = {}
            for i,param in enumerate(params_to_weight):
                if weight_args[i]['type'] == 'Gaussian':

                    self.functions.update({param:Gaussian(weight_args[i]['mean'],weight_args[i]['sigma'],param)})

                elif weight_args[i]['type'] == 'upper_limit':

                    self.functions.update({param:StepUpperLimit(weight_args[i]['break'],weight_args[i]['sigma'],param)})

                elif weight_args[i]['type'] == 'lower_limit':

                    self.functions.update({param:StepLowerLimit(weight_args[i]['break'],weight_args[i]['sigma'],param)})

                elif weight_args[i]['type'] == 'BinaryUpper':

                    self.functions.update({param:BinaryUpper(weight_args[i]['break'],weight_args[i]['sigma'],param)})

                elif weight_args[i]['type'] == 'BinaryLower':

                    self.functions.update({param:BinaryLower(weight_args[i]['break'],weight_args[i]['sigma'],param)})

    def __call__(self,samples):

        pnames = samples.pnames

        weight = 1

        for name in self.functions.keys():
            function = self.functions[name]
            if name in pnames:

                weight *= function(x=samples.samples[name])

        return weight

class Gaussian(object):

    def __init__(self,mean,sigma,name):

        self.mean = mean
        self.sigma = sigma
        self.name = name

    def __call__(self, **kwargs):
        #return (2*numpy.pi*self.sigma**2)**-0.5*numpy.exp(-0.5*(self.mean-kwargs['x'])**2*self.sigma**-2)

        return numpy.exp(-0.5*(self.mean-kwargs['x'])**2*self.sigma**-2)

class StepUpperLimit(object):

    def __init__(self, break_value, sigma, name):

        self.break_value = break_value
        self.sigma = sigma
        self.name = name

    def __call__(self, **kwargs):

        exponent = kwargs['x'] * self.break_value ** -1

        exp = numpy.exp(-exponent**2 * self.sigma)

        return exp

class StepLowerLimit(object):

    def __init__(self, break_value, sigma, name):
        self.break_value = break_value
        self.sigma = sigma
        self.name = name

    def __call__(self, **kwargs):

        exponent = self.break_value * kwargs['x'] ** -1

        exp = numpy.exp(-exponent**2 * self.sigma)

        return exp

class BinaryUpper(object):

    def __init__(self, break_value, sigma, name):

        self.break_value = break_value
        #self.sigma = sigma
        #self.name = name

    def __call__(self, **kwargs):

        weights = numpy.ones_like(kwargs['x'])
        weights[numpy.where(weights >= self.break_value)] = 0

        return weights


class BinaryLower(object):

    def __init__(self, break_value, sigma, name):
        self.break_value = break_value
        # self.sigma = sigma
        # self.name = name

    def __call__(self, **kwargs):
        weights = numpy.ones_like(kwargs['x'])
        weights[numpy.where(weights <= self.break_value)] = 0

        return weights
