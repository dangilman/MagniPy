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

        self.weight = [1]*self.Nlenses

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

    def compute_weight(self,param_name,weight_kwargs):

        weight = 1

        for i, things in enumerate(weight_kwargs):
            print things

            func = things['type']

            if func == 'Gaussian':

                mean = weight_kwargs[i]['mean']
                sigma = weight_kwargs[i]['sigma']

                weight *= (2 * np.pi*sigma**2) ** -.5 * np.exp(
                    -0.5 * (self.parameters[param_name[i]] - mean) ** 2 * sigma ** -2)

            elif func == 'exp_high_trunc':

                cutoff = weight_kwargs[i]['cutoff']

                weight *= np.exp(-(self.parameters[param_name[i]] * cutoff ** -1) ** 2)

        self.weight = weight

    def draw(self,tol):

        assert tol>1

        indexes = np.argsort(self.statistic)[0:tol]

        new_param_dic = {}

        for key,values in self.parameters.iteritems():

            new_param_dic.update({key:values[indexes]})

        if self.weight is None:
            weight = np.ones(len(indexes))
        else:
            weight = self.weight

        return new_param_dic,weight[indexes]

