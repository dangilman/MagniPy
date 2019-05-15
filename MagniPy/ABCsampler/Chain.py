import pandas
from MagniPy.ABCsampler.ChainOps import *
import numpy as numpy
from MagniPy.Analysis.KDE.kde import KDE_nD
from time import time

class JointDensity(object):

    def __init__(self, param_x, param_y, array):

        self.param_x, self.param_y = param_x, param_y
        self._plist = [self.param_x, self.param_y]

        self.array = np.array(array.T)

    def __call__(self, p1, p2):

        if p1 in self._plist and p2 in self._plist:

            return self.array, True

        else:
            return None, False


class MarginalDensity(object):

    def __init__(self, param, array):

        self.param = param

        self.array = np.array(array)

    def _confidence_int(self):

        p = [0.68, 0.95]
        high = []
        low = []
        N = len(self.array)
        for pi in p:
            low.append(np.sort(self.array)[::-1][int(N * pi)])
            high.append(np.sort(self.array)[int(N * pi)])
        return [low, high]

    def __call__(self, param):

        if param == self.param:

            return self.array, True

        else:
            return None, False

class ChainFromChain(object):

    def __init__(self, chain, indicies, load_flux = False):

        self._chain_parent = chain
        self.params_varied = chain.params_varied
        self.truths = chain.truths
        self.prior_info = chain.prior_info
        self.pranges = chain.pranges
        self.Nlenses = chain.Nlenses

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

                lens.add_parameters(pnames=self.params_varied,fname=chainpath_out+'processed_chains/'+
                                  self._chain_parent._chain_name+'/lens'+str(i+1)+'/samples.txt',use_pandas=False)

                for pname in lens._parameters[0].keys():
                    lens.parameters[0][pname] = lens._parameters[0][pname][ordered_inds]

            lens.statistic = [new_statistic]

class ChainFromSamples(object):

    def __init__(self,chain_name='',which_lens=None, error=0, trimmed_ranges=None,
        deplete = False, deplete_fac = 0.5, n_pert = 1, load = True, statistics=None,
                 parameters = None, from_parent = False):
        try:
            self.params_varied, self.truths, self.prior_info = read_chain_info(chainpath_out + '/processed_chains/' +
                                                                       chain_name + '/simulation_info.txt')
            Ncores, cores_per_lens, self.Nlenses = read_run_partition(chainpath_out + '/processed_chains/' +
                                                                      chain_name + '/simulation_info.txt')

        except:
            self.params_varied, self.truths, self.prior_info = read_chain_info(chainpath_out + '/chain_stats/' +
                                                                               chain_name + '/simulation_info.txt')
            Ncores, cores_per_lens, self.Nlenses = read_run_partition(chainpath_out + '/chain_stats/' +
                                                                      chain_name + '/simulation_info.txt')

        if 'logmhm' in self.truths:
            self.truths.update({'log_m_break':self.truths['logmhm']})

        self.pranges = self.get_pranges(self.prior_info)
        self._chain_name = chain_name
        self.n_pert = n_pert


        self.chain_file_path = chainpath_out + 'chain_stats/' + chain_name +'/'

        self.lenses = []
        self.error = error

        if which_lens is None:
            which_lens = numpy.arange(1,self.Nlenses+1)

        for ind in which_lens:

            #print('loading '+str(ind)+'...')

            if from_parent is not False:
                assert isinstance(from_parent, ChainFromChain)

                new_lens = from_parent.lenses[ind]
            else:
                new_lens = SingleLens(zlens = None, zsource=None, flux_path=chainpath_out + '/processed_chains/' +
                                                                    chain_name+'/', ID = ind)

            if load and not from_parent:
                for ni in range(0,n_pert):
                    #print('loading '+str(ni+1)+'...')
                    fname = 'statistic_'+str(error)+'error_'+str(ni+1)+'.txt'

                    if statistics is None:
                        finite = new_lens.add_statistic(fname=self.chain_file_path + 'lens' + str(ind) + '/'+fname)
                    else:
                        new_lens.statistic.append(statistics[ind-1])

                    if parameters is None:
                        #print(self.params_varied)
                        fname = 'params_' + str(error) + 'error_' + str(ni+1) + '.txt'
                        new_lens.add_parameters(pnames=self.params_varied,finite_inds=finite,
                                        fname=self.chain_file_path+'lens'+str(ind)+'/'+fname)
                    else:
                        new_dictionary = {}
                        for i, pname in enumerate(self.params_varied):
                            new_dictionary.update({pname: parameters[ind-1][:, i].astype(float)})

                        new_lens.parameters.append(new_dictionary)

                    if deplete:

                        L = int(len(new_lens.statistic[ni]))

                        keep = np.arange(0, L)

                        u = np.random.rand(L)

                        keep = keep[np.where(u <= deplete_fac)]

                        for pname in new_lens.parameters[ni].keys():

                            new_lens.parameters[ni][pname] = new_lens.parameters[ni][pname][keep]
                        new_lens.statistic[ni] = new_lens.statistic[ni][keep]

            self.lenses.append(new_lens)

    def eval_KDE(self, bandwidth_scale = 1, tol = 2500, nkde_bins = 20,
                 save_to_file = True, smooth_KDE = True, weights = None):

        if not hasattr(self, '_kernel'):

            self._kernel = KDE_nD(bandwidth_scale)

        posteriors = self.get_posteriors(tol)

        points = []
        ranges = []
        for i, pi in enumerate(self.params_varied):
            points.append(np.linspace(self.pranges[pi][0], self.pranges[pi][1], nkde_bins))
            ranges.append(self.pranges[pi])

        density = np.ones(tuple([nkde_bins]*len(self.params_varied)))
        print_time = True
        counter = 0

        for n in range(len(self.lenses)):

            t0 = time()

            density_n = 0
            param_weight = 1

            for p in range(0, self.n_pert):

                data = np.empty(shape = (tol, len(self.params_varied)))
                for i, pi in enumerate(self.params_varied):
                    data[:,i] = posteriors[n][p].samples[pi]

                    if weights is not None and pi in weights.keys():

                        if pi in weights.keys():

                            param_weight *= weights[pi](data[:,i])

                    else:
                        param_weight = None

                if smooth_KDE:
                    density_n += self._kernel(data, points, ranges, weights=param_weight)
                else:
                    density_n += numpy.histogramdd(data, range = ranges,
                                                   density=True, bins=nkde_bins,
                                                   weights=param_weight)[0]

            t_elpased = np.round((time() - t0) * 60 ** -1, 1)
            if print_time:
                print(str(t_elpased) + ' min per lens.')
                print_time = False
            elif counter%5 == 0:
                print('completed '+str(counter) + ' of '+str(len(self.lenses)) + '...')
            counter += 1
            density *= density_n

        self.density = density

        self._density_projections(density, save_to_file, bandwidth_scale)

    def get_projection(self, params, bandwidth_scale=None, load_from_file = True):

        if load_from_file:
            assert bandwidth_scale is not None
            if len(params) == 1:
                fname = self._fnamemarginal(params[0], bandwidth_scale)
                return np.loadtxt(fname)
            else:
                fname1 = self._fnamejoint(params[0], params[1], bandwidth_scale)
                fname2 = self._fnamejoint(params[1], params[0], bandwidth_scale)
                try:
                    return np.loadtxt(fname1)
                except:
                    return np.loadtxt(fname2)

        assert hasattr(self, 'density')

        if len(params) == 1:

            for marg in self.marginal_densities:
                out, give_back = marg(params[0])

                if give_back:
                    return out

        else:

            for joint in self.joint_densities:
                out, give_back = joint(params[0], params[1])
                if give_back:
                    return out

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

    def _density_projections(self, density, save_to_file, bandwidth_scale):

        self.joint_densities = []
        self.marginal_densities = []
        pinds = np.arange(0,len(self.params_varied))

        for i, p in enumerate(self.params_varied[::-1]):
            inds = deepcopy(pinds)
            inds = inds[np.where(inds != i)]
            d_i = np.sum(density, axis=tuple(inds))
            marg = MarginalDensity(p, d_i)
            self.marginal_densities.append(marg)

            if save_to_file:
                fname = self._fnamemarginal(p, bandwidth_scale)
                np.savetxt(fname, X = marg.array)

        if len(density.shape) == 5:
            #self._projections_5D(density, bandwidth_scale, save_to_file)
            self._projections_5D_SIDM(density, bandwidth_scale, save_to_file)
        elif len(density.shape) == 3:
            self._projections_3D(density, bandwidth_scale, save_to_file)
        elif len(density.shape) == 2:
            self._projections_2D(density, bandwidth_scale, save_to_file)

    def _fnamejoint(self, px, py, bandwidth_scale):

        if not os.path.exists(self.chain_file_path + 'computed_densities/'):
            create_directory(self.chain_file_path + 'computed_densities/')

        string = str(len(self.lenses))+'lens_'+str(self.error)+'error_'
        string += str(self.n_pert)+'avg_' + px + '__' + py  + '_'+ str(bandwidth_scale) +'.txt'
        return self.chain_file_path + 'computed_densities/' + string

    def _fnamemarginal(self, p, bandwidth_scale):

        if not os.path.exists(self.chain_file_path + 'computed_densities/'):
            create_directory(self.chain_file_path + 'computed_densities/')

        string = str(len(self.lenses)) + 'lens_' + str(self.error) + 'error_'
        string += str(self.n_pert) + 'avg_' + p + '_' + str(bandwidth_scale) +'.txt'
        return self.chain_file_path + 'computed_densities/' + string

    def _projections_2D(self, density, bandwidth_scale, save_to_file):

        proj_logmcore = density

        px, py = 'SIDMcross', 'a0_area'
        self.joint_densities.append(JointDensity(px, py, proj_logmcore))

        if save_to_file:
            for j in self.joint_densities:
                fname = self._fnamejoint(j.param_x, j.param_y, bandwidth_scale)
                np.savetxt(fname, X=j.array)

    def _projections_3D(self, density, bandwidth_scale, save_to_file):

        proj_a0logm = np.sum(density, axis=1).T
        proj_a0core = np.sum(density, axis=2).T
        proj_logmcore = np.sum(density, axis=0).T

        px, py = 'a0_area', 'log_m_break'
        self.joint_densities.append(JointDensity(px, py, proj_a0logm))

        py, px = 'a0_area', 'core_ratio'
        self.joint_densities.append(JointDensity(px, py, proj_a0core))

        py, px = 'log_m_break', 'core_ratio'
        self.joint_densities.append(JointDensity(px, py, proj_logmcore))

        if save_to_file:
            for j in self.joint_densities:
                fname = self._fnamejoint(j.param_x, j.param_y, bandwidth_scale)
                np.savetxt(fname, X=j.array)

    def _projections_5D_SIDM(self, density, bandwidth_scale, save_to_file):

        proj_LOSa0 = np.sum(density, axis=(2, 3, 4)).T
        proj_a0cross = np.sum(density, axis=(1, 3, 4)).T
        proj_a0SIE = np.sum(density, axis=(1, 2, 4)).T
        proj_srca0 = np.sum(density, axis=(1, 2, 3)).T

        proj_crossLOS = np.sum(density, axis=(0, 3, 4))
        proj_sieLOS = np.sum(density, axis=(0, 2, 4))
        proj_srcLOS = np.sum(density, axis=(0, 2, 3)).T

        proj_SIEcross = np.sum(density, axis=(0, 1, 4))
        proj_srccross = np.sum(density, axis=(0, 1, 3))

        proj_srcSIE = np.sum(density, axis=(0, 1, 2))

        px, py = 'LOS_normalization', 'a0_area'
        self.joint_densities.append(JointDensity(px, py, proj_LOSa0))

        py, px = 'a0_area', 'SIDMcross'
        self.joint_densities.append(JointDensity(px, py, proj_a0cross))

        py, px = 'a0_area', 'SIE_gamma'
        self.joint_densities.append(JointDensity(px, py, proj_a0SIE))

        px, py = 'source_size_kpc', 'a0_area'
        self.joint_densities.append(JointDensity(px, py, proj_srca0))

        px, py = 'SIDMcross', 'LOS_normalization'
        self.joint_densities.append(JointDensity(px, py, proj_crossLOS))

        px, py = 'SIE_gamma', 'LOS_normalization'
        self.joint_densities.append(JointDensity(px, py, proj_sieLOS))

        py, px = 'source_size_kpc', 'LOS_normalization'
        self.joint_densities.append(JointDensity(px, py, proj_srcLOS))

        px, py = 'SIE_gamma', 'SIDMcross'
        self.joint_densities.append(JointDensity(px, py, proj_SIEcross))

        px, py = 'source_size_kpc', 'SIDMcross'
        self.joint_densities.append(JointDensity(px, py, proj_srccross))

        px, py = 'source_size_kpc', 'SIE_gamma'
        self.joint_densities.append(JointDensity(px, py, proj_srcSIE))

        if save_to_file:
            for j in self.joint_densities:
                fname = self._fnamejoint(j.param_x, j.param_y, bandwidth_scale)
                np.savetxt(fname, X = j.array)

    def _projections_5D_WDM(self, density, bandwidth_scale, save_to_file):

        proj_LOSa0 = np.sum(density, axis=(2, 3, 4)).T
        proj_a0logm = np.sum(density, axis=(1, 3, 4)).T
        proj_a0SIE = np.sum(density, axis=(1, 2, 4)).T
        proj_srca0 = np.sum(density, axis=(1, 2, 3)).T

        proj_logmLOS = np.sum(density, axis=(0, 3, 4)).T
        proj_sieLOS = np.sum(density, axis=(0, 2, 4)).T
        proj_srcLOS = np.sum(density, axis=(0, 2, 3)).T

        proj_SIElogm = np.sum(density, axis=(0, 1, 4))
        proj_srclogm = np.sum(density, axis=(0, 1, 3))

        proj_srcSIE = np.sum(density, axis=(0, 1, 2))

        px, py = 'LOS_normalization', 'a0_area'
        self.joint_densities.append(JointDensity(px,py,proj_LOSa0))

        py, px = 'a0_area', 'log_m_break'
        self.joint_densities.append(JointDensity(px, py, proj_a0logm))

        py, px = 'a0_area', 'SIE_gamma'
        self.joint_densities.append(JointDensity(px, py, proj_a0SIE))

        px, py = 'source_size_kpc', 'a0_area'
        self.joint_densities.append(JointDensity(px, py, proj_srca0))

        px, py = 'log_m_break', 'LOS_normalization'
        self.joint_densities.append(JointDensity(px, py, proj_logmLOS))

        px, py = 'SIE_gamma', 'LOS_normalization'
        self.joint_densities.append(JointDensity(px, py, proj_sieLOS))

        py, px = 'source_size_kpc', 'LOS_normalization'
        self.joint_densities.append(JointDensity(px, py, proj_srcLOS))

        px, py = 'SIE_gamma', 'log_m_break'
        self.joint_densities.append(JointDensity(px, py, proj_SIElogm))

        px, py = 'source_size_kpc', 'log_m_break'
        self.joint_densities.append(JointDensity(px, py, proj_srclogm))

        px, py = 'source_size_kpc', 'SIE_gamma'
        self.joint_densities.append(JointDensity(px, py, proj_srcSIE))

        if save_to_file:
            for j in self.joint_densities:
                fname = self._fnamejoint(j.param_x, j.param_y, bandwidth_scale)
                np.savetxt(fname, X = j.array)

class SingleLens(object):

    def __init__(self, zlens, zsource, weights=None, flux_path='', ID = None):

        self._ID = ID
        self.weights = weights
        self.posterior = None
        self.zlens, self.zsource = zlens, zsource
        self.flux_path = flux_path
        self.parameters, self.statistic = [], []
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

    def draw(self,tol, reject_pnames, keep_ranges):

        self.posterior = []

        if reject_pnames is not None:

            for i in range(0, len(self.statistic)):

                for reject_pname, keep_range in zip(reject_pnames, keep_ranges):
                    samps = self.parameters[i][reject_pname]
                    indexes = numpy.where(numpy.logical_and(samps >= keep_range[0], samps <= keep_range[1]))[0]

                    for pname in self.parameters[i].keys():
                        self.parameters[i][pname] = self.parameters[i][pname][indexes]
                #print('keeping ' + str(len(self.parameters[i][pname])) + ' samples')

        for i in range(0, len(self.statistic)):
            inds = numpy.argsort(self.statistic[i])[0:tol]

            new_param_dic = {}

            for key in self.parameters[i].keys():
                values = self.parameters[i][key]
                new_param_dic.update({key: values[inds]})

            self.posterior.append(PosteriorSamples(new_param_dic, weights=None))

    def add_parameters(self,pnames=None,fname=None,finite_inds=None,use_pandas=True):

        if use_pandas:
            params = numpy.squeeze(pandas.read_csv(fname, header=None, sep=" ", index_col=None)).astype(numpy.ndarray)
        else:
            params = numpy.squeeze(np.loadtxt(fname))

        params = numpy.array(params)

        new_dictionary = {}

        for i,pname in enumerate(pnames):

            new_params = params[:,i].astype(float)
            #new_params = round_to(params[:,i].astype(float), rounding[pname])

            new_dictionary.update({pname: new_params})

        self.parameters.append(new_dictionary)

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

class PosteriorSamples:

    def __init__(self,samples,weights=None):

        self.samples = samples
        self.pnames = samples.keys()

        for ki in samples.keys():
            self.length = int(len(samples[ki]))
            break

        if weights is None:
            self.weights = numpy.ones(self.length)
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