import matplotlib.pyplot as plt
from MagniPy.Analysis.Statistics.singledensity import *
from MagniPy.ABCsampler.Chain import *
from MagniPy.Analysis.Visualization.Joint2D import Joint2D

def sample_chain(chain_name='',which_lenses=None, parameters=[]):

    full_chain = FullChains(chain_name,which_lens=which_lenses)

    posteriors = full_chain.get_posteriors(400)

    prior_weights_global_gamma = WeightedSamples(params_to_weight=['SIE_gamma'],weight_args=[{'type':'Gaussian','mean':2.08,'sigma':.05}])

    prior_weights_global_shear = WeightedSamples(params_to_weight=['SIE_shear'],
                                                 weight_args=[{'type': 'Gaussian', 'mean': 0.05, 'sigma': .01}])

    posteriors = full_chain.re_weight(posteriors,[prior_weights_global_gamma],indexes=[1])

    densities = []

    for i,posterior in enumerate(posteriors):

        single_density = SingleDensity(pnames=parameters,samples=posterior,pranges=full_chain.pranges,
                                       reweight=True,kernel_function='Gaussian',scale=5,bandwidth_scale=1.12)

        density,coordsx,coordsy = single_density()

        densities.append(density)

    joint = Joint2D(densities)

    joint.make_plot(param_names=parameters,param_ranges=full_chain.pranges,truths=full_chain.truths)

    plt.show()

sample_chain('singleplane_test',which_lenses=[1,2,3,4,5,6],parameters=['fsub','logmhm'])
