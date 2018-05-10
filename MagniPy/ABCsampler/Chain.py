from Analysis.Statistics.summary_statistics import *
from ChainOps import *


class FullChains:

    def __init__(self,chain_name='',Nlenses=None,which_lens=None,index=1,error=0):

        self.params_varied, self.truths, self.prior_info = read_chain_info(chainpath + '/processed_chains/' + chain_name + '/simulation_info.txt')

        self.pranges = self.get_pranges(self.prior_info)

        if Nlenses is None:
            Ncores,cores_per_lens,self.Nlenses = read_run_partition(chainpath + '/processed_chains/' + chain_name + '/simulation_info.txt')
        else:
            self.Nlenses = Nlenses

        self.chain_file_path = chainpath + '/processed_chains/' + chain_name +'/'

        self.lenses = []

        if which_lens is None:
            which_lens = np.arange(1,self.Nlenses+1)

        for ind in which_lens:

            new_lens = SingleLens(Nparams=len(self.params_varied))

            new_lens.add_parameters(fname=self.chain_file_path+'lens'+str(ind)+'/samples.txt')

            new_lens.compute_summary_statistic(self.import_observed(ind, error=error, index=index),
                                               self.import_model(ind, error=error, index=index))
            self.lenses.append(new_lens)

    def get_posteriors(self,tol=None):

        posteriors = []

        for lens in self.lenses:

            if lens.posterior is None:

                lens.draw(tol)

            posteriors.append(lens.posterior)

        return posteriors

    def draw(self,tol):

        for lens in self.lenses:

            lens.draw(tol)

    def get_pranges(self,info):

        pranges = {}

        for pname,keys in info.iteritems():
            if keys['prior_type'] == 'Gaussian':
                pranges[pname] = [float(keys['mean']) - float(keys['sigma'])*2,float(keys['mean'])+float(keys['sigma'])*2]
            elif keys['prior_type'] == 'Uniform':
                pranges[pname] = [float(keys['low']),float(keys['high'])]

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

                posterior.change_weights(weight_function)

            return posteriors

        else:

            if isinstance(indexes,list):

                assert isinstance(weight_function,list),'If apply a prior on a lens by lens basis, must provide a ' \
                                                        'list of individual weight functions'
                count = 0
                for i in range(1,len(self.lenses)+1):
                    if i in indexes:
                        posteriors[i-1].change_weights(weight_function[count])
                        count+=1
            return posteriors

class SingleLens:

    def __init__(self,Nparams=int,weights=None):

        self.weights = weights
        self.posterior = None

    def draw(self,tol):

        inds = np.argsort(self.statistic)[0:tol]

        new_param_dic = {}

        for key, values in self.parameters.iteritems():
            new_param_dic.update({key:values[inds]})

        self.posterior = PosteriorSamples(new_param_dic,weights=None)

    def add_parameters(self,params=None,fname=None):

        with open(fname,'r') as f:
            names = f.readline().split(' ')
        pnames = []
        for name in names:
            if name =='#' or name =='\n':
                continue
            else:
                pnames.append(name)

        params = np.loadtxt(fname)

        new_dictionary = {}

        for i,pname in enumerate(pnames):
            new_dictionary.update({pname:params[:,i]})

        self.parameters = new_dictionary

    def add_weights(self,params,weight_function):
        """
        :param params: dictionary with param names and values
        :param weight_function: a function that takes the 'params' dictionary and returns a
        1d array of weight with same length as params
        :return:
        """
        self.weights = weight_function(params)

    def compute_summary_statistic(self,observed=None,model=None):

        self.statistic = quadrature_piecewise(model,observed)

class PosteriorSamples:

    def __init__(self,samples,weights=None):

        self.samples = samples
        self.pnames = samples.keys()
        self.length = len(samples[samples.keys()[0]])

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

    def __call__(self,samples):

        pnames = samples.pnames

        weight = 1

        for name, function in self.functions.iteritems():

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



