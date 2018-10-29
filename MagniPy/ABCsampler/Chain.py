from MagniPy.Analysis.Statistics.summary_statistics import *
from MagniPy.ABCsampler.ChainOps import *

class FullChains:

    def __init__(self,chain_name='',Nlenses=None,which_lens=None,index=1,error=0, trimmed_ranges=None):

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

        self.chain_file_path = chainpath_out + 'processed_chains/' + chain_name +'/'

        self.lenses = []

        if which_lens is None:
            which_lens = np.arange(1,self.Nlenses+1)

        for ind in which_lens:

            new_lens = SingleLens(Nparams=len(self.params_varied))

            fname = 'statistic_'+str(error)+'error_'+str(index)+'.txt'
            finite = new_lens.add_statistic(fname=self.chain_file_path + 'lens' + str(ind) + '/fluxratios/'+fname)

            fname = 'params_' + str(error) + 'error_' + str(index) + '.txt'
            new_lens.add_parameters(pnames=self.params_varied,finite_inds=finite,
                                    fname=self.chain_file_path+'lens'+str(ind)+'/fluxratios/'+fname)

            self.lenses.append(new_lens)

        if trimmed_ranges is None:
            self.pranges_trimmed = self.pranges
        else:
            self.pranges_trimmed = trimmed_ranges

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

    def __init__(self,Nparams=int,weights=None):

        self.weights = weights
        self.posterior = None

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

    def change_weights(self,weight_function):

        self.weights = weight_function(self)
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

                #elif weight_args[i]['type'] == 'step':

                    #self.functions.update({param: Step(weight_args[i]['step'], , param)})

    def __call__(self,samples):

        pnames = samples.pnames

        weight = 1

        for name in self.functions.keys():
            function = self.functions[name]
            if name in pnames:

                weight *= function(x=samples.samples[name])

        return weight

class Gaussian:

    def __init__(self,mean,sigma,name):

        self.mean = mean
        self.sigma = sigma
        self.name = name

    def __call__(self, **kwargs):
        return (2*np.pi*self.sigma**2)**-0.5*np.exp(-0.5*(self.mean-kwargs['x'])**2*self.sigma**-2)



