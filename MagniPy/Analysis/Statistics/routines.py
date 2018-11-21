from MagniPy.Analysis.Statistics.singledensity import SingleDensity
from MagniPy.ABCsampler.Chain import WeightedSamples
from copy import deepcopy, copy
import numpy as np

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

def build_densities(sim_list, parameters, pranges, xtrim=None, ytrim=None, bandwidth_scale = 1, use_kde_joint = True,
                    use_kde_marginal=False, steps=None, reweight = True):

    sim_densities, sim_pranges = [], []
    pranges_list = [pranges] * len(sim_list)
    for sim, pranges in zip(sim_list, pranges_list):
        densities = []

        for i, posterior in enumerate(sim):

            if len(parameters) == 1:
                kde_flag = use_kde_marginal
            else:
                kde_flag = use_kde_joint

            single_density = SingleDensity(pnames=parameters, samples=posterior,
                                           pranges=pranges,
                                           reweight=reweight,
                                           kernel_function='Gaussian',
                                           scale=5,
                                           bandwidth_scale=bandwidth_scale, use_kde=kde_flag,
                                           steps=steps)

            density, param_ranges = single_density(xtrim=xtrim, ytrim=ytrim)

            densities.append(density)

        sim_densities.append(densities)
        sim_pranges.append(param_ranges)

    return sim_densities, sim_pranges

def duplicate_with_cuts(full_chain, tol, pnames_reject_list, keep_ranges_list):

    chain = deepcopy(full_chain)

    assert len(pnames_reject_list) == len(keep_ranges_list)

    posteriors, prange_list = [], []

    for pnames_reject, keep_ranges in zip(pnames_reject_list, keep_ranges_list):

        posterior = chain.get_posteriors(tol, reject_pnames=pnames_reject,
                                         keep_ranges=keep_ranges)
        posteriors.append(posterior)

        if keep_ranges is None:
            prange_list.append(chain.pranges)
        else:

            range_copy = copy(chain.pranges)

            for k, name in enumerate(pnames_reject):
                range_copy.update({name:keep_ranges[k]})

            prange_list.append(range_copy)

    return posteriors, prange_list

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





