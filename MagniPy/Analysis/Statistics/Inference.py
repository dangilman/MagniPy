import matplotlib.pyplot as plt
from MagniPy.Analysis.Statistics.singledensity import *
from MagniPy.ABCsampler.Chain import *
from MagniPy.Analysis.Visualization.Joint2D import Joint2D

def sample_chain(chain_name='',which_lenses=None, parameters=[],error=None,index=None):

    full_chain = FullChains(chain_name,which_lens=which_lenses,error=error,index=index)

    posteriors = full_chain.get_posteriors(150)

    prior_weights_global_gamma = WeightedSamples(params_to_weight=['SIE_gamma'],weight_args=[{'type':'Gaussian','mean':2.08,'sigma':0.05}])


    #prior_weights_global_fsub = WeightedSamples(params_to_weight=['fsub'],
    #                                             weight_args=[{'type': 'Gaussian', 'mean': 0.01, 'sigma': .002}])

    posteriors = full_chain.re_weight(posteriors,[prior_weights_global_gamma],indexes=[1])

    densities = []

    for i,posterior in enumerate(posteriors):

        single_density = SingleDensity(pnames=parameters,samples=posterior,pranges=full_chain.pranges,
                                       reweight=True,kernel_function='Gaussian',scale=5,bandwidth_scale=1)

        density,coordsx,coordsy = single_density()

        densities.append(density)

    joint = Joint2D(densities)

    joint.make_plot(param_names=parameters,param_ranges=full_chain.pranges,truths=full_chain.truths)

    plt.show()

sample_chain('singleplane_run',which_lenses=[20],parameters=['fsub','logmhm'],error=0,index=1)
