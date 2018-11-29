from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from MagniPy.Analysis.Visualization.Joint2D import Joint2D
from MagniPy.Analysis.Visualization.density1D import Density1D
from MagniPy.Analysis.Statistics.routines import *
from scipy.misc import comb

class TriPlot(object):

    cmap = 'gist_heat'

    # default_contour_colors = (colors.cnames['orchid'], colors.cnames['darkviolet'], 'k')
    default_contour_colors = [(colors.cnames['grey'], colors.cnames['black'], 'k'),
                              (colors.cnames['skyblue'], colors.cnames['blue'], 'k'),
                              (colors.cnames['coral'], 'r', 'k'),
                              (colors.cnames['orchid'], colors.cnames['darkviolet'], 'k')]
    truth_color = 'r'

    def __init__(self, posteriors=[], parameter_names = [], pranges = [], parameter_trim = None,
                 fig_size = 8, bandwidth_scale = 1, truths=None, steps = 10, kde_joint =True,
                 kde_marginal = True, reweight = True):

        self._init(fig_size)
        self._steps = steps
        self._reweight = reweight
        self._kde_joint, self._kde_marginal = kde_joint, kde_marginal
        if parameter_trim is None:
            parameter_trim = {}
            for pname in parameter_names:
                parameter_trim.update({pname: None})

        self._nparams = len(parameter_names)
        self._grid = self._init_grid(self._nparams, parameter_names)

        simulations, simulation_pranges, marginal_densities, marginal_ranges = self._get_sims(posteriors, parameter_names, pranges, parameter_trim, bandwidth_scale)

        self.simulation_densities = simulations
        self.marginal_densities = marginal_densities
        self.marginal_pranges = marginal_ranges

        self.parameter_names, self.parameter_ranges = parameter_names, pranges

        self._truths = truths

        self.fig = plt.figure(1)

        N = len(parameter_names)
        size_scale = N * 0.1 + 1
        self.fig.set_size_inches(fig_size * size_scale, fig_size * size_scale)

    def _init(self, fig_size):

        self._tick_lab_font = 12 * fig_size * 7**-1

        plt.rcParams['axes.linewidth'] = 2.5*fig_size*7**-1

        plt.rcParams['xtick.major.width'] = 2.5*fig_size*7**-1
        plt.rcParams['xtick.major.size'] = 6*fig_size*7**-1
        plt.rcParams['xtick.minor.size'] = 2*fig_size*7**-1

        plt.rcParams['ytick.major.width'] = 2.5*fig_size*7**-1
        plt.rcParams['ytick.major.size'] = 6*fig_size*7**-1
        plt.rcParams['ytick.minor.size'] = 2*fig_size*7**-1

    def _subplot_index(self, col, row):

        return col * row + col + 1

    def makeplot(self, levels=[0.05,0.22,1], filled_contours=True, contour_alpha = 0.6,
                 spacing = [0.1, 0.1, 0.05, 0.05, 0.2, 0.11], rebin=20):

        axis = self._makeplot(levels = levels, filled_contours = filled_contours, contour_alpha = contour_alpha,
                       rebin = rebin)

        plt.subplots_adjust(left=spacing[0], bottom=spacing[1], right=1-spacing[2], top=1-spacing[3],
                            wspace=spacing[4], hspace=spacing[5])

        return axis, self.default_contour_colors

    def _makeplot(self, levels = None, filled_contours=None, contour_alpha = None, rebin=15):

        plot_index = 1
        joint_k = 0
        marginal_indexes = []
        densities = []
        marginal_axes = []
        marginal_names = []
        marginal_ranges = []

        axis = []

        for row in range(0, self._nparams):
            for col in range(0, self._nparams):

                ax = plt.subplot(self._nparams, self._nparams, plot_index)
                axis.append(ax)

                pnames = self._grid[row, col]

                if pnames is None:
                    ax.axis('off')

                elif pnames[-1]=='marginal':
                    marginal_axes.append(ax)
                    marginal_indexes.append(plot_index)
                    marginal_names.append(pnames[0])
                    marginal_ranges.append(self.parameter_ranges[pnames[0]])

                else:

                    param_ranges = {pnames[0]:self.parameter_ranges[pnames[0]],
                                    pnames[1]:self.parameter_ranges[pnames[1]]}

                    joint = Joint2D(self.simulation_densities[joint_k], ax=ax, fig=self.fig)
                    _, joint_info = joint.make_plot(param_ranges=param_ranges, param_names=pnames, filled_contours=filled_contours,
                                    contour_alpha=contour_alpha, levels=levels, truths=self._truths, tick_label_font=self._tick_lab_font)
                    joint_k += 1

                    if row != self._nparams-1:
                        ax.set_xticklabels([])
                        ax.set_xticks([])
                        ax.set_xlabel('')

                    if col > 0:
                        ax.set_yticklabels([])
                        ax.set_yticks([])
                        ax.set_ylabel('')

                    if row == self._nparams - 1:
                        densities.append(joint_info)

                plot_index += 1

        for i in range(0, len(densities) + 1):

            oneD = Density1D(ax = marginal_axes[i], fig = self.fig)

            if i < len(densities):

                xlabel_on = False

            else:

                xlabel_on = True

            marginalized = self.marginal_densities[marginal_names[i]]

            oneD.make_marginalized(marginalized, marginal_names[i], marginal_ranges[i],
                                   xlabel_on = xlabel_on, truths=self._truths, rebin=rebin,
                                   tick_label_font = self._tick_lab_font)

        return axis

    def _init_grid(self, nparams, param_names):

        grid = np.zeros(shape=(nparams, nparams), dtype=object)

        for col in range(0, nparams):
            for row in range(0, nparams):

                if row == col:
                    if row == 0:
                        grid[row, row] = [param_names[col], param_names[row+1], 'marginal']
                    else:
                        grid[row, row] = [param_names[col], param_names[0], 'marginal']
                elif col > row:
                    grid[row, col] = None
                else:
                    grid[row, col] = [param_names[col], param_names[row]]

        return grid

    def _get_sims(self, posteriors, pnames, param_ranges, param_trim, bandwidth_scale):

        L = len(pnames)
        simulations = []
        simulation_pranges = []
        marginal_densities = {}
        marginal_ranges = {}

        if L < 2:
            raise Exception('must have at least 2 parameters.')

        for k, name in enumerate(pnames):
            marg, marg_range = build_densities(posteriors, [name], {name: param_ranges[name]}, xtrim=param_trim[name],
                                               steps=self._steps, use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            marginal_densities.update({name:marg})
            marginal_ranges.update({name:marg_range})

        if L == 2:

            parameters = [pnames[0], pnames[1]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}

            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]

            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps, 
                                                use_kde_joint=self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)

            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

        elif L == 3:

            parameters = [pnames[0], pnames[1]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}

            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]

            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)

            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

            parameters = [pnames[0], pnames[2]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}

            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]

            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)

            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

            parameters = [pnames[1], pnames[2]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}

            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]
            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim,
                                                steps=self._steps, use_kde_joint = self._kde_joint,
                                                use_kde_marginal=self._kde_marginal, reweight = self._reweight)

            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

        elif L == 4:

            parameters = [pnames[0], pnames[1]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}

            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]
            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

            parameters = [pnames[0], pnames[2]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}

            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]
            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

            parameters = [pnames[0], pnames[3]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}

            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]
            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

            parameters = [pnames[1], pnames[2]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}
            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]
            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,

                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

            parameters = [pnames[1], pnames[3]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}

            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]
            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

            parameters = [pnames[2], pnames[3]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}
            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]
            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

        elif L == 5:

            parameters = [pnames[0], pnames[1]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}

            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]
            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

            parameters = [pnames[0], pnames[2]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}

            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]
            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

            parameters = [pnames[0], pnames[3]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}

            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]
            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

            parameters = [pnames[0], pnames[4]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}

            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]
            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

            parameters = [pnames[1], pnames[2]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}
            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]
            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

            parameters = [pnames[1], pnames[3]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}

            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]
            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

            parameters = [pnames[1], pnames[4]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}

            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]
            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

            parameters = [pnames[2], pnames[3]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}
            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]
            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

            parameters = [pnames[2], pnames[4]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}
            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]
            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

            parameters = [pnames[3], pnames[4]]
            pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}
            if param_trim is None:
                xtrim, ytrim = None, None
            else:
                xtrim = param_trim[parameters[0]]
                ytrim = param_trim[parameters[1]]
            sims, sim_pranges = build_densities(posteriors, parameters,
                                                pranges, bandwidth_scale=bandwidth_scale,
                                                xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            simulations.append(sims)
            simulation_pranges.append(sim_pranges)

        return simulations, simulation_pranges, marginal_densities, marginal_ranges
