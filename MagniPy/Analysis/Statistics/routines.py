from MagniPy.Analysis.Statistics.singledensity import SingleDensity
from MagniPy.ABCsampler.Chain import WeightedSamples
from copy import deepcopy, copy
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from MagniPy.paths import *

def legend_info(params, Nlenses, errors, colors, weighted, axes, weight_params=None, scale_scale = 1):

    def build_latex(mean, sigma):
        string = "$\sim$"+' '+"$\mathcal{N} \ $"+'('+str(mean)+', '+str(sigma)+')'
        return string

    if len(params) == 2:
        ax_idx = 1
        xbox = 0
        ybox = 0.3
        delta = 0.6*scale_scale
        nsamp_scale = 0

    elif len(params) == 3:
        ax_idx = 1
        xbox = 1.1
        ybox = -0.1
        delta = 0.75*scale_scale
        nsamp_scale = 0

    elif len(params) == 4:
        ax_idx = 3
        xbox = -1.1
        ybox = -0.35
        delta = 1.1*scale_scale
        nsamp_scale = 0

    elif len(params) == 5:
        ax_idx = 4
        xbox = -1.1
        ybox = -0.85
        delta = 1.3*scale_scale
        nsamp_scale = 0

    nlens_string = str(int(Nlenses)) + ' lenses'
    flux_strings = []
    weight_lines = []
    if weighted[0] or weighted[1]:
        weight_string = ['Parameter weights:']
        nsamp_scale += 0
    else:
        weight_string = []
        nsamp_scale += -0.2
    weight_lines.append(Line2D([0], [0], color='k', lw=0))
    do_weight = True

    for i, ei in enumerate(errors):
        if ei == 0:
            flux_strings.append('pefect flux\nmeasurements')
        else:
            flux_strings.append(str(int(ei))+'% flux\nuncertainties')
        if weighted[i]:
            flux_strings[-1] += '\n(with re-weighted\nsamples)'

        if do_weight and weighted[i]:
            do_weight = False

            for keyi in weight_params.keys():

                mean, sigma = weight_params[keyi]['mean'], weight_params[keyi]['sigma']

                if keyi=='SIE_gamma':
                    mean = r'$\gamma_{i}$'
                    keyi = r'$\gamma_{\rm{macro}}$' + ' ' + build_latex(mean, sigma)
                elif keyi == 'a0_area':
                    keyi = r'$\sigma_{\rm{sub}}$'+ ' ' + build_latex(mean, sigma)
                elif keyi == 'log_m_break':
                    keyi = r'$\log_{10} \left(m_{\rm{hm}}\right)$' + ' ' + build_latex(mean, sigma)
                elif keyi == 'source_size_kpc':
                    sigma, mean = int(sigma*1000), int(mean*1000)
                    keyi = r'$\sigma_{\rm{src}} \left[\rm{pc}\right]$' + ' ' + build_latex(mean, sigma)
                elif keyi == 'LOS_normalization':
                    keyi = r'$\delta_{\rm{LOS}}$' + ' ' + build_latex(mean, sigma)
                weight_string += [keyi]
                weight_lines.append(Line2D([0], [0], color='k', lw=0))

    custom_lines = []
    for col in range(0,len(errors)):
        custom_lines.append(Line2D([0], [0], color=colors[col][1], lw=8))

    legend1 = axes[ax_idx].legend(custom_lines, flux_strings, loc = 'lower left',
                                  bbox_to_anchor = (xbox, ybox), fontsize=14, frameon = False,
                                  labelspacing=1.5)

    legend2 = axes[ax_idx].legend(weight_lines, weight_string, loc = 'lower left',
                                  bbox_to_anchor = (xbox-0.35*delta, ybox-0.5*delta), fontsize=15, frameon = False)

    axes[ax_idx].add_artist(legend1)

    #axes[ax_idx].legend(custom_lines, flux_strings, loc = loc, fontsize=14)
    axes[ax_idx].annotate(nlens_string, xy=(xbox+0.1*delta,ybox+1.75*delta+nsamp_scale-0.5), xycoords='axes fraction', fontsize=16)



def weight_posteriors(params_to_reweight, posteriors_to_reweight, which_lenses):

    weighted_posteriors = []
    for post_i in posteriors_to_reweight:
        weighted = False
        for param in params_to_reweight.keys():
            p = params_to_reweight[param]

            if weighted:
                weighted_i = reweight_posteriors_individually(weighted_i, param, p['mean'], p['sigma'],
                                                                       np.array(which_lenses) - 1,
                                                                       post_to_reweight=np.arange(0, len(posteriors_to_reweight)))

            else:
                weighted_i = reweight_posteriors_individually(post_i, param, p['mean'], p['sigma'],
                                                                       np.array(which_lenses) - 1,
                                                                       post_to_reweight=np.arange(0, len(posteriors_to_reweight)))
                weighted = True

        weighted_posteriors.append(weighted_i)

    return weighted_posteriors

def duplicate_with_subset(full_chain, lens_indicies):

    new_chains = []

    for indexes in lens_indicies:

        chain = deepcopy(full_chain)

        lenses = []

        for index in indexes:
            lenses.append(chain.lenses[index])

        chain.lenses = lenses

        new_chains.append(chain)

    return new_chains

def build_densities_precomputed(sim_list, chain_name, Nlenses, parameters):

    densities = []
    for sim in sim_list:
        for lens in range(1, Nlenses+1):
            fname = prefix + 'data/sims/densities/' + chain_name + 'lens' + str(lens) + '/'
            fname += str(parameters[0]) + '_' + str(parameters[1]) + '.txt'
            density = np.loadtxt(fname)
            norm = np.sum(density) * np.shape(density)[0] ** -2
            density *= norm
            densities.append(density)


def build_densities(sim_list, parameters, pranges, xtrim=None, ytrim=None, bandwidth_scale = 1, use_kde_joint = True,
                    use_kde_marginal=True, steps=None, reweight = True, pre_computed = False, chain_name = None,
                    errors = None):

    sim_densities, sim_pranges = [], []
    pranges_list = [pranges] * len(sim_list)

    if pre_computed:
        for k, (sim, pranges) in enumerate(zip(sim_list, pranges_list)):

            densities = []
            for lens in range(1, 51):
                try:
                    fname = prefix + 'data/sims/densities/' + chain_name + '/lens' + str(lens) + '/'
                    fname += str(parameters[0]) + '_' + str(parameters[1]) + '_error_'+str(errors[k])+'.txt'
                    density = np.loadtxt(fname)
                except:
                    fname = prefix + 'data/sims/densities/' + chain_name + '/lens' + str(lens) + '/'
                    fname += str(parameters[1]) + '_' + str(parameters[0]) + '_error_' + str(errors[k]) + '.txt'
                    density = np.loadtxt(fname)
                norm = np.sum(density) * np.shape(density)[0] ** -2
                density *= norm
                densities.append(density)

            sim_densities.append(densities)
            sim_pranges.append(pranges)

        return sim_densities, sim_pranges

    else:
        for sim, pranges in zip(sim_list, pranges_list):
            densities = []

            for i, posterior in enumerate(sim):

                if len(parameters) == 1:
                    kde_flag = use_kde_marginal
                else:
                    kde_flag = use_kde_joint

                L = len(posterior)
                for i, post in enumerate(posterior):
                    single_density = SingleDensity(pnames=parameters, samples=post,
                                               pranges=pranges,
                                               reweight=reweight,
                                               kernel_function='Gaussian',
                                               scale=5,
                                               bandwidth_scale=bandwidth_scale, use_kde=kde_flag,
                                               steps=steps)

                    den, param_ranges = single_density(xtrim=xtrim, ytrim=ytrim)
                    if i == 0:
                        density = den
                    else:
                        density += den

                norm=np.sum(density*L**-1) * np.shape(density)[0]**-2
                densities.append(density*L**-1 * norm ** -1)

            sim_densities.append(densities)
            sim_pranges.append(param_ranges)

        return sim_densities, sim_pranges

def duplicate_with_cuts(full_chain, tol, pnames_reject_list, keep_ranges_list):

    chain = deepcopy(full_chain)

    assert len(pnames_reject_list) == len(keep_ranges_list)

    posterior = chain.get_posteriors(tol, reject_pnames=pnames_reject_list,
                                     keep_ranges=keep_ranges_list)

    if keep_ranges_list is None:
        pranges_new = chain.pranges
    else:

        pranges_new = copy(chain.pranges)

        for k, name in enumerate(pnames_reject_list):
            pranges_new.update({name:keep_ranges_list[k]})

    return posterior, pranges_new

def reweight_posteriors_individually(posteriors, weight_param, weight_means, weight_sigmas,
                                     index_lists, post_to_reweight, weight_type = 'Gaussian'):

    weighted_posteriors = []

    weight_list = []

    assert len(weight_means) == len(weight_sigmas)

    for j, idx in enumerate(index_lists):

        if weight_type == 'Gaussian':
            w = WeightedSamples(params_to_weight=[weight_param],
                            weight_args=[{'type': 'Gaussian', 'mean': weight_means[j], 'sigma': weight_sigmas[j]}])
        elif weight_type == 'lower_limit':
            w = WeightedSamples(params_to_weight=[weight_param],
                                weight_args= [{'type': 'lower_limit', 'break': weight_means[j], 'sigma': weight_sigmas[j]}])
        elif weight_type == 'upper_limit':
            w = WeightedSamples(params_to_weight=[weight_param],
                                weight_args=[{'type': 'upper_limit', 'break': weight_means[j], 'sigma': weight_sigmas[j]}])
        elif weight_type == 'binary_upper':
            w = WeightedSamples(params_to_weight=[weight_param],
                                weight_args=[{'type': 'BinaryUpper', 'break': weight_means[j], 'sigma': weight_sigmas[j]}])
        elif weight_type == 'binary_lower':
            w = WeightedSamples(params_to_weight=[weight_param],
                                weight_args=[{'type': 'BinaryLower', 'break': weight_means[j], 'sigma': weight_sigmas[j]}])


        weight_list.append(w)

    for i, post in enumerate(posteriors):

        if i not in post_to_reweight:
            new = copy(post)
            weighted_posteriors.append(new)
            continue

        new_post = []
        count = 0

        for j, post_j in enumerate(post):

            weighted_post = copy(post_j)

            if j in index_lists:

                weighted_post.change_weights(weight_list[count])
                count += 1
                new_post.append(weighted_post)
            else:
                new_post.append(weighted_post)

        weighted_posteriors.append(new_post)

    return weighted_posteriors

def reweight_posteriors(posteriors, weight_classes, index_lists=None):

    assert len(posteriors) == len(weight_classes)

    weighted_posteriors = []

    for j, (post, weight) in enumerate(zip(posteriors, weight_classes)):

        if weight is None:
            weighted_posteriors.append(post)
            continue

        new_post = []

        for i, post_i in enumerate(post):

            weighted_post = copy(post_i)

            if index_lists[j] is None:

                for wi in weight:
                    weighted_post.change_weights(wi)
                new_post.append(weighted_post)

            else:
                if i+1 in index_lists[j]:

                    for wi in weight:
                        weighted_post.change_weights(wi)
                    new_post.append(weighted_post)

                else:
                    new_post.append(weighted_post)


        weighted_posteriors.append(new_post)

    return weighted_posteriors

def barplothist(bar_heights, coords, rebin):

    if rebin is not None:
        new = []
        if len(bar_heights) % rebin == 0:
            fac = int(len(bar_heights) / rebin)
            for i in range(0, len(bar_heights), fac):
                new.append(np.mean(bar_heights[i:(i + fac)]))

            bar_heights = np.array(new)
        else:
            raise Exception('not implemented here.')

    bar_width = np.absolute(coords[-1] - coords[0]) * len(bar_heights) ** -1
    bar_centers = []
    for i in range(0, len(bar_heights)):
        bar_centers.append(coords[0] + bar_width * (0.5 + i))

    integral = np.sum(bar_heights) * bar_width * len(bar_centers) ** -1

    bar_heights = bar_heights * integral ** -1

    return bar_centers, bar_width, bar_heights





