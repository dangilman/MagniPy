from ChainOps import *
from summary_statistics import *

class ParamChains:

    def __init__(self,chain_name='',Nlenses=None):

        self.params_varied, self.truths, self.prior_info = read_chain_info(chainpath + '/processed_chains/' + chain_name + '/simulation_info.txt')

        self.pranges = self.get_pranges(self.prior_info)

        if Nlenses is None:
            Ncores,cores_per_lens,self.Nlenses = read_run_partition(chainpath + '/processed_chains/' + chain_name + '/simulation_info.txt')
        else:
            self.Nlenses = Nlenses

        self.chain_file_path = chainpath + '/processed_chains/' + chain_name +'/'

        self.lenses = []

        for i in range(0, self.Nlenses):
            self.lenses.append(SingleLensChain())

    def get_pranges(self,info):

        pranges = {}

        for pname,keys in info.iteritems():
            if keys['prior_type'] == 'Gaussian':
                pranges[pname] = [float(keys['mean']) - float(keys['sigma'])*2,float(keys['mean'])+float(keys['sigma'])*2]
            elif keys['prior_type'] == 'Uniform':
                pranges[pname] = [float(keys['low']),float(keys['high'])]

        return pranges

    def import_all(self,error=0,index=1):

        self.import_parameters(self.Nlenses)
        self.import_observed(self.Nlenses,error=error,index=index)
        self.import_model(self.Nlenses,error=error,index=index)

    def load_param_names(self,fname):

        param_names = []

        with open(fname, 'r') as f:
            keys = f.readline().split(' ')
        for word in keys:
            if word not in ['#', '\n', '']:
                param_names.append(word)

        return param_names

    def import_parameters(self,Nlenses):

        for n in range(0,Nlenses):

            fname = self.chain_file_path+'lens'+str(n+1)+'/samples.txt'

            param_values = np.loadtxt(fname)

            param_names = self.load_param_names(fname)

            param_dic = {}

            for i,key in enumerate(param_names):
                param_dic[key] = param_values[:,i]

            self.lenses[n].add_parameters(param_dic)

    def compute_statistics(self,Nlenses,error,index,statistic='quadrature_piecewise'):

        for i in range(0,Nlenses):

            self.lenses[i].compute_statistic(stat_function=quadrature_piecewise)

    def import_observed(self,Nlenses,error=0,index=1):

        for n in range(0,Nlenses):

            fname = self.chain_file_path+'lens'+str(n+1)+'/fluxratios/'+'observed_'+ str(int(error * 100)) + 'error_'+ str(index)+'.txt'

            self.lenses[n].add_observed_fluxratios(np.loadtxt(fname))

    def import_model(self,Nlenses,error=0,index=1):

        for n in range(0, Nlenses):

            fname = self.chain_file_path + 'lens' + str(n + 1) + '/fluxratios/' + 'model_' + str(
                int(error * 100)) + 'error_' + str(index) + '.txt'

            self.lenses[n].add_model_fluxratios(np.loadtxt(fname))

    def add_weights(self,param_name,weight_kwargs = []):

        if isinstance(param_name,str):

            param_name = [param_name]

            assert len(weight_kwargs==1),'number of weight kwargs must equal number of params.'

        for lens in self.lenses:

            lens.compute_weight(param_name,weight_kwargs)

    def draw(self,tol=1000,error=0,index=1,stat_function='quadrature_piecewise'):

        self.compute_statistics(self.Nlenses,error=error,index=index,statistic=stat_function)

        posterior_samples = []

        for i,lens in enumerate(self.lenses):

            lens_posterior = lens.draw(tol)

            posterior_samples.append(lens_posterior)

        return posterior_samples

class SingleLensChain:

    def __init__(self):

        self.parameters = None
        self.fluxratios = None
        self.observed_fluxratios = None
        self.statistic = None
        self.weights = None

        self.posterior_samples = None

        self.perturbed_observed_fluxratios = []
        self.perturbed_model_fluxratios = []

    def add_observed_fluxratios(self,obs):

        self.observed_fluxratios = obs

    def add_parameters(self,params):

        self.parameters = params

    def add_model_fluxratios(self,mod):

        self.fluxratios = mod

    def compute_statistic(self,stat_function):

        self.statistic = stat_function(self.fluxratios,self.observed_fluxratios)

    def draw(self,tol):

        assert tol>1

        indexes = np.argsort(self.statistic)[0:tol]

        new_param_dic = {}

        for key,values in self.parameters.iteritems():

            new_param_dic.update({key:values[indexes]})

        return new_param_dic

class WeightedSamples:

    def __init__(self,params=None,weight_args=None):

        if params is None:
            self.functions = None
        else:
            self.functions = {}
            for i,param in enumerate(params):
                if weight_args[i]['type'] == 'Gaussian':
                    self.functions.update({param:Gaussian(weight_args[i]['mean'],weight_args[i]['sigma'],param)})

    def function(self,x,y=None):

        weight = np.ones(len(x[x.keys()[0]]))

        if self.functions is None:
            return np.ones(len(x[x.keys()[0]]))

        if x.keys()[0] in self.functions.keys():

            pname = x.keys()[0]

            which_func = self.functions[pname]

            weight *= which_func(x=x[pname])

        if y is not None:
            if y.keys()[0] in self.functions.keys():

                pname = y.keys()[0]

                which_func = self.functions[pname]

                weight *= which_func(x=y[pname])

        return weight

class Gaussian:

    def __init__(self,mean,sigma,name):

        self.mean = mean
        self.sigma = sigma
        self.name = name

    def __call__(self, **kwargs):
        return (2*np.pi*self.sigma**2)**-0.5*np.exp(-0.5*(self.mean-kwargs['x'])**2*self.sigma**-2)



