import matplotlib.pyplot as plt
from MagniPy.Analysis.Statistics.singledensity import *
from MagniPy.ABCsampler.Chain import *
from MagniPy.Analysis.Visualization.Joint2D import Joint2D

def sample_chain(chain_name='',which_lenses=None,
                 parameters=[],error=None,
                 index=None, tol = 500, savename=None):

    full_chain = FullChains(chain_name,which_lens=which_lenses,error=error,index=index)

    posteriors = full_chain.get_posteriors(tol)

    prior_weights_global_cpower = WeightedSamples(params_to_weight=['c_power'],weight_args=[{'type':'Gaussian','mean':-0.17,'sigma':0.02}])

    prior_weights_global_srcsize = WeightedSamples(params_to_weight=['source_size_kpc'],
                                                 weight_args=[{'type': 'Gaussian', 'mean': 0.02, 'sigma': 0.002}])

    #posteriors = full_chain.re_weight(posteriors,[prior_weights_global_srcsize],indexes=None)

    densities = []
    param_ranges = []

    for i,posterior in enumerate(posteriors):

        single_density = SingleDensity(pnames=parameters,samples=posterior,pranges=full_chain.pranges,
                                       reweight=True,kernel_function='Gaussian',scale=5,bandwidth_scale=1)

        density,pranges = single_density()

        densities.append(density)
        param_ranges = pranges

    joint = Joint2D(densities)

    joint.make_plot(param_names=parameters,param_ranges=param_ranges,truths=full_chain.truths,
                    filled_contours=True, color_index=0)
    if savename is not None:
        plt.savefig(savename)
    #joint.make_plot(param_names=parameters, param_ranges=full_chain.pranges,filled_contours=True)
    plt.show()

#sample_chain('LOS_CDM_1',which_lenses=[1,2,3,4,5],parameters=['fsub','log_m_break'],error=0,index=1,tol = 500, savename='fsub_logmhm.pdf')
#sample_chain('LOS_CDM_1',which_lenses=[1,2,3,4,5],parameters=['c_power','log_m_break'],error=0,index=1,tol = 500, savename='cpower_logmhm.pdf')
#sample_chain('LOS_CDM_1',which_lenses=[1,2,3,4,5],parameters=['source_size_kpc','log_m_break'],error=0,index=1,tol = 500, savename='srcsize_logmhm.pdf')
#sample_chain('CDM_diverse',which_lenses=[1,2,3,4,5,6,7,8,9,10],parameters=['log_m_break','source_size_kpc'],error=0,index=1,tol = 1000, savename='logmhm_srcsize_error0.pdf')
#sample_chain('CDM_diverse',which_lenses=[2],parameters=['fsub','log_m_break'],error=4,index=1,tol = 1800, savename='fsub_logmhm_error4_weightedsrcsize.pdf')
#sample_chain('LOS_CDM_1',which_lenses=[1,2,3,4,5],parameters=['fsub','log_m_break'],error=0,index=1,tol = 800, savename='fsub_logmhm.pdf')
#sample_chain('LOS_CDM_1',which_lenses=[1,2,3,4,5],parameters=['c_power','log_m_break'],error=0,index=1,tol = 1500, savename='cpower_logmhm_srcsizeweight.pdf')
#sample_chain('LOS_CDM_1',which_lenses=[1,2,3,4,5],parameters=['source_size_kpc','log_m_break'],error=0,index=1,tol = 800, savename='srcsize_logmhm_allweight.pdf')
