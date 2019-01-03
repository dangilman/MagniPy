from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from MagniPy.Analysis.Visualization.posterior_plots import _Joint2D, Density1D
from MagniPy.Analysis.Statistics.routines import *
from scipy.misc import comb

class TriPlot(object):

    cmap = 'gist_heat'

    # default_contour_colors = (colors.cnames['orchid'], colors.cnames['darkviolet'], 'k')
    default_contour_colors = [(colors.cnames['lightgreen'],colors.cnames['green'], 'k'),
                              (colors.cnames['orchid'], colors.cnames['darkviolet'], 'k'),
                              (colors.cnames['grey'], colors.cnames['black'], 'k'),
                              (colors.cnames['skyblue'], colors.cnames['blue'], 'k'),
                              (colors.cnames['coral'], 'r', 'k')]
    truth_color = 'r'

    spacing = np.array([0.1, 0.1, 0.05, 0.05, 0.2, 0.11])
    spacing_scale = 1

    def __init__(self, posteriors=[], parameter_names = [], pranges = [], parameter_trim = None,
                 fig_size = 8, bandwidth_scale = 1, truths=None, steps = 10, kde_joint =True,
                 kde_marginal = True, reweight = True, pre_computed = False, chain_name = None,
                 errors = None):

        self._pre_computed = pre_computed
        self._chain_name = chain_name
        self._errors = errors

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

        self._posterior_grid = self._get_sims(posteriors,
                   self._grid, parameter_names, pranges, parameter_trim, bandwidth_scale)

        self.parameter_names, self.parameter_ranges = parameter_names, pranges

        self._truths = truths

        self.fig = plt.figure(1)

        N = len(parameter_names)
        size_scale = N * 0.1 + 1
        self.fig.set_size_inches(fig_size * size_scale, fig_size * size_scale)

    def _init(self, fig_size):

        self._tick_lab_font = 12 * fig_size * 7**-1
        self._label_font = 15 * fig_size * 7**-1
        plt.rcParams['axes.linewidth'] = 2.5*fig_size*7**-1

        plt.rcParams['xtick.major.width'] = 2.5*fig_size*7**-1
        plt.rcParams['xtick.major.size'] = 6*fig_size*7**-1
        plt.rcParams['xtick.minor.size'] = 2*fig_size*7**-1

        plt.rcParams['ytick.major.width'] = 2.5*fig_size*7**-1
        plt.rcParams['ytick.major.size'] = 6*fig_size*7**-1
        plt.rcParams['ytick.minor.size'] = 2*fig_size*7**-1

    def _subplot_index(self, col, row):

        return col * row + col + 1

    def makeplot(self, levels=[0.05,0.22,1], filled_contours=True, contour_alpha = 0.6, rebin=20, compute_bayes_factor = False):

        axis, bayes_factor = self._makeplot(levels = levels, filled_contours = filled_contours, contour_alpha = contour_alpha,
                       rebin = rebin, compute_bayes_factor = compute_bayes_factor)

        plt.subplots_adjust(left=self.spacing[0]*self.spacing_scale, bottom=self.spacing[1]*self.spacing_scale,
                            right=1-self.spacing[2]*self.spacing_scale, top=1-self.spacing[3]*self.spacing_scale,
                            wspace=self.spacing[4]*self.spacing_scale, hspace=self.spacing[5]*self.spacing_scale)

        return axis, self.default_contour_colors, bayes_factor

    def _makeplot(self, levels = None, filled_contours=None, contour_alpha = None, rebin=15,
                  compute_bayes_factor = False):

        plot_index = 1

        densities = []

        axis = []
        bayes_factor = {}

        for row in range(0, self._nparams):
            for col in range(0, self._nparams):

                ax = plt.subplot(self._nparams, self._nparams, plot_index)
                axis.append(ax)

                cell = self._posterior_grid[row, col]

                if cell is None:
                    ax.axis('off')

                elif cell.type == 'marginal':

                    if col < self._nparams-1:
                        xlabel_on = False
                    else:
                        xlabel_on = True

                    oneD = Density1D(ax=ax, fig=self.fig)
                    oneD.default_contour_colors = self.default_contour_colors

                    oneD.make_plot_1D(cell.posterior, cell.pnames[0],
                                      cell.ranges[0][cell.pnames[0]],
                                      xlabel_on=xlabel_on, truths=self._truths, rebin=rebin,
                                      tick_label_font=self._tick_lab_font,
                                      label_size=self._label_font)
                    if compute_bayes_factor is not False:
                        if cell.pnames[0] in compute_bayes_factor.keys():

                            cut = compute_bayes_factor[cell.pnames[0]]
                            _bayes_factor = oneD._compute_bayes_factor(cell.posterior,
                                                       cell.ranges[0][cell.pnames[0]], cut, rebin)
                            bayes_factor.update({cell.pnames[0]: _bayes_factor})

                else:

                    joint = _Joint2D(cell.posterior, ax=ax, fig=self.fig, cmap=self.cmap)
                    joint.default_contour_colors = self.default_contour_colors
                    _, joint_info = joint.make_plot(param_ranges=cell.ranges[0], param_names=cell.pnames, filled_contours=filled_contours,
                                    contour_alpha=contour_alpha, levels=levels, truths=self._truths,
                                                    tick_label_font=self._tick_lab_font, label_size=self._label_font)

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

        return axis, bayes_factor

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

    def _get_sims(self, posteriors, grid, pnames, param_ranges, param_trim, bandwidth_scale):

        L = np.shape(grid)[0]

        marginal_densities = {}
        marginal_ranges = {}

        grid_post = np.zeros_like(grid, dtype=object)

        if L < 2:
            raise Exception('must have at least 2 parameters.')

        for k, name in enumerate(pnames):
            marg, marg_range = build_densities(posteriors, [name], {name: param_ranges[name]}, xtrim=param_trim[name],
                                               steps=self._steps, use_kde_joint = self._kde_joint, use_kde_marginal=self._kde_marginal, reweight = self._reweight)
            marginal_densities.update({name:marg})
            marginal_ranges.update({name:marg_range})

        for col in range(0, L):
            for row in range(0, L):

                cell = grid[row, col]

                if cell is None:
                    grid_post[row, col] = None
                    continue
                if cell[-1]=='marginal':
                    name = cell[0]
                    marg, marg_range = build_densities(posteriors, [name], {name: param_ranges[name]},
                                                       xtrim=param_trim[name],
                                                       steps=self._steps, use_kde_joint=self._kde_joint,
                                                       use_kde_marginal=self._kde_marginal, reweight=self._reweight)
                    grid_post[row, col] = GridCell(marg, marg_range, [name], type='marginal')

                else:
                    parameters = [cell[0], cell[1]]
                    pranges = {parameters[0]: param_ranges[parameters[0]], parameters[1]: param_ranges[parameters[1]]}
                    if param_trim is None:
                        xtrim, ytrim = None, None
                    else:
                        xtrim = param_trim[parameters[0]]
                        ytrim = param_trim[parameters[1]]
                    sims, sim_pranges = build_densities(posteriors, parameters,
                                                        pranges, bandwidth_scale=bandwidth_scale,
                                                        xtrim=xtrim, ytrim=ytrim, steps=self._steps,
                                                        use_kde_joint=self._kde_joint,
                                                        use_kde_marginal=self._kde_marginal, reweight=self._reweight,
                                                        pre_computed=self._pre_computed,
                                                        chain_name = self._chain_name, errors = self._errors)

                    grid_post[row, col] = GridCell(sims, sim_pranges, parameters, type='joint')

        return grid_post

class GridCell(object):

    def __init__(self, posterior, ranges, pnames, type):

        self.posterior = posterior
        self.ranges = ranges
        self.pnames = pnames

        self.type = type



