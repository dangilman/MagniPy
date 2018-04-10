from ChainOps import *
from SummaryStatistics import *

class Chains:

    def __init__(self, chain_name=''):

        self.params_varied, self.truths = read_chain_info(chainpath + '/processed_chains/' + chain_name + '/simulation_info.txt')

        Ncores,cores_per_lens,self.Nlenses = read_run_partition(chainpath + '/processed_chains/' + chain_name + '/simulation_info.txt')

        self.chain_file_path = chainpath + '/processed_chains/' + chain_name +'/'

    def import_all(self,error,index=1):

        self.lenses = []

        for i in range(0, self.Nlenses):
            self.lenses.append(SingleLensChain())

        summary_statistics = self.compute_statistics(self.Nlenses,error,index)

        parameters = self.import_parameters(self.Nlenses)

        for i,lens in enumerate(self.lenses):

            lens.add_statistic(summary_statistics[i])
            lens.add_parameters(parameters[i])

    def import_parameters(self,Nlenses):

        params = []
        for n in range(0,Nlenses):
            fname = self.chain_file_path+'lens'+str(n+1)+'samples.txt'
            params.append(np.loadtxt(fname))

        return params


    def compute_statistics(self,Nlenses,error,index,statistic='quadrature_piecewise'):

        observed = self.import_observed(Nlenses,error=error,index=index)
        model = self.import_model(Nlenses, error=error, index=index)

        if statistic=='quadrature_piecewise':
            return quadrature_piecewise(model,observed)
        else:
            raise Exception('statistic name '+statistic+' not recognized.')

    def import_observed(self,Nlenses,error=0,index=1):

        obs = []

        if error==0:
            index = ''

        for n in range(0,Nlenses):

            fname = self.chain_file_path+'lens'+str(n+1)+'/fluxratios/'+'observed_'+ str(int(error * 100)) + 'error_'+ str(index)+'.txt'

            obs.append(np.loadtxt(fname))

        return obs

    def import_model(self,Nlenses,error=0,index=1):

        model = []

        if error == 0:
            index = ''

        for n in range(0, Nlenses):

            fname = self.chain_file_path + 'lens' + str(n + 1) + '/fluxratios/' + 'model_' + str(
                int(error * 100)) + 'error_' + str(index) + '.txt'

            model.append(np.loadtxt(fname))

        return model

    def draw(self,tol=1000):

        posterior_samples = []

        for i,lens in enumerate(self.lenses):

            posterior_samples.append(lens.parameters[np.argsort(lens.statistic)[0:tol]])

        return posterior_samples


class SingleLensChain:

    def __init__(self):

        self.parameters = None
        self.fluxratios = None
        self.observed_fluxratios = None

        self.perturbed_observed_fluxratios = []
        self.perturbed_model_fluxratios = []

    def add_observed_fluxratios(self,obs):

        self.observed_fluxratios = obs

    def add_parameters(self,params):

        self.parameters = params

    def add_model_fluxratios(self,mod):

        self.fluxratios = mod

    def add_statistic(self,stat):

        self.statistic = stat
