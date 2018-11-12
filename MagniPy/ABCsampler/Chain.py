from MagniPy.Analysis.Statistics.summary_statistics import *
from MagniPy.ABCsampler.ChainOps import *

class FullChains:

    def __init__(self,chain_name='',Nlenses=None,which_lens=None,index=1,error=0, trimmed_ranges=None,
                 zlens_src_file = chainpath_out + '/processed_chains/simulation_zRein.txt'):

        if zlens_src_file is None:
            zd = [None] * len(which_lens), [None] * len(which_lens)
        else:
            zd, zs, _ = np.loadtxt(zlens_src_file, unpack = True)

        self.params_varied, self.truths, self.prior_info = read_chain_info(chainpath_out + '/processed_chains/' +
                                                                           chain_name + '/simulation_info.txt')
        if 'logmhm' in self.truths:
            self.truths.update({'log_m_break':self.truths['logmhm']})

        self.pranges = self.get_pranges(self.prior_info)

        if Nlenses is None:
            Ncores,cores_per_lens,self.Nlenses = read_run_partition(chainpath_out + '/processed_chains/' +
                                                                    chain_name + '/simulation_info.txt')
        else:
            self.Nlenses = Nlenses

        self.chain_file_path = chainpath_out + 'chain_stats/' + chain_name +'/'

        self.lenses = []

        if which_lens is None:
            which_lens = np.arange(1,self.Nlenses+1)

        for ind in which_lens:

            new_lens = SingleLens(zlens = zd[ind], zsource=zs[ind])

            fname = 'statistic_'+str(error)+'error_'+str(index)+'.txt'
            finite = new_lens.add_statistic(fname=self.chain_file_path + 'lens' + str(ind) + '/'+fname)

            fname = 'params_' + str(error) + 'error_' + str(index) + '.txt'
            new_lens.add_parameters(pnames=self.params_varied,finite_inds=finite,
                                    fname=self.chain_file_path+'lens'+str(ind)+'/'+fname)

            self.lenses.append(new_lens)

        if trimmed_ranges is None:
            self.pranges_trimmed = self.pranges
        else:
            self.pranges_trimmed = trimmed_ranges

    def add_derived_parameters(self, new_param_name, transformation_function, pnames_input, new_param_ranges):

        for lens in self.lenses:
            lens.add_derived_parameter(new_param_name, transformation_function, pnames_input)

        self.pranges.update({new_param_name: new_param_ranges})

    def get_posteriors(self,tol=None, reject_pnames = None, keep_ranges = None):

        posteriors = []

        for lens in self.lenses:

            lens.draw(tol, reject_pnames, keep_ranges)

            posteriors.append(lens.posterior)

        return posteriors

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

        return np.loadtxt(fname)

    def import_model(self,ind,error=0,index=1):

        fname = self.chain_file_path + 'lens' + str(ind) + '/fluxratios/' + 'model_' + str(
            int(error * 100)) + 'error_' + str(index) + '.txt'

        return np.loadtxt(fname)

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

class SingleLens:

    def __init__(self, zlens, zsource, weights=None):

        self.weights = weights
        self.posterior = None
        self.zlens, self.zsource = zlens, zsource

    def draw(self,tol, reject_pnames, keep_ranges):

        if reject_pnames is not None:

            for reject_pname, keep_range in zip(reject_pnames, keep_ranges):
                samps = self.parameters[reject_pname]
                indexes = np.where(np.logical_and(samps >= keep_range[0], samps <= keep_range[1]))[0]

                for pname in self.parameters.keys():
                    self.parameters[pname] = self.parameters[pname][indexes]
            print('keeping ' + str(len(self.parameters[pname])) + ' samples')

        inds = np.argsort(self.statistic)[0:tol]

        new_param_dic = {}

        for key in self.parameters.keys():
            values = self.parameters[key]
            new_param_dic.update({key: values[inds]})

        self.posterior = PosteriorSamples(new_param_dic, weights=None)

    def add_derived_parameter(self, new_pname, transformation_function, pnames):

        args = {}

        if not isinstance(pnames, list):
            pnames = [pnames]

        for name in pnames:
            args.update({name: self.parameters[name]})

        kwargs = {'zlens': self.zlens, 'zsrc': self.zsource}

        self.parameters.update({new_pname:transformation_function(**args, **kwargs)})

    def add_parameters(self,pnames=None,fname=None,finite_inds=None):

        params = np.loadtxt(fname)

        params = params[finite_inds]

        new_dictionary = {}

        for i,pname in enumerate(pnames):

            newparams = params[:,i]

            new_dictionary.update({pname:newparams})

        self.parameters = new_dictionary

    def add_weights(self,params,weight_function):
        """
        :param params: dictionary with param names and values
        :param weight_function: a function that takes the 'params' dictionary and returns a
        1d array of weight with same length as params
        :return:
        """
        self.weights = weight_function(params)

    def add_statistic(self,fname):

        statistic = np.loadtxt(fname)

        finite = np.where(np.isfinite(statistic))

        self.statistic = statistic[finite]

        return finite

class PosteriorSamples:

    def __init__(self,samples,weights=None):

        self.samples = samples
        self.pnames = samples.keys()

        for ki in samples.keys():
            self.length = int(len(samples[ki]))
            break

        if weights is None:
            self.weights = np.ones(self.length)
        else:
            assert len(weights) == self.length
            self.weights = weights

    def print_weights(self):

        for weight in self.weights:
            print(weight)

    def change_weights(self,weight_function):

        self.weights *= weight_function(self)
        assert len(self.weights) == self.length

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
        #return (2*np.pi*self.sigma**2)**-0.5*np.exp(-0.5*(self.mean-kwargs['x'])**2*self.sigma**-2)
        return np.exp(-0.5*(self.mean-kwargs['x'])**2*self.sigma**-2)

class StepUpperLimit(object):

    def __init__(self, break_value, sigma, name):

        self.break_value = break_value
        self.sigma = sigma
        self.name = name

    def __call__(self, **kwargs):

        exponent = kwargs['x'] * self.break_value ** -1

        exp = np.exp(-exponent * self.sigma)

        return exp

class StepLowerLimit(object):

    def __init__(self, break_value, sigma, name):
        self.break_value = break_value
        self.sigma = sigma
        self.name = name

    def __call__(self, **kwargs):

        exponent = self.break_value * kwargs['x'] ** -1

        exp = np.exp(-exponent * self.sigma)

        return exp

class BinaryUpper(object):

    def __init__(self, break_value, sigma, name):

        self.break_value = break_value
        #self.sigma = sigma
        #self.name = name

    def __call__(self, **kwargs):

        weights = np.ones_like(kwargs['x'])
        weights[np.where(weights >= self.break_value)] = 0

        return weights


class BinaryLower(object):

    def __init__(self, break_value, sigma, name):
        self.break_value = break_value
        # self.sigma = sigma
        # self.name = name

    def __call__(self, **kwargs):
        weights = np.ones_like(kwargs['x'])
        weights[np.where(weights <= self.break_value)] = 0

        return weights

