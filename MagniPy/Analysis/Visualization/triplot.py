from matplotlib import colors
import matplotlib.pyplot as plt
from MagniPy.Analysis.KDE.kde import *
import numpy as np

class TriPlot(object):
    cmap = 'gist_heat'

    # default_contour_colors = (colors.cnames['orchid'], colors.cnames['darkviolet'], 'k')
    _default_contour_colors = [(colors.cnames['dodgerblue'], colors.cnames['blue'], 'k'),
                               (colors.cnames['orchid'], colors.cnames['darkviolet'], 'k'),
                               (colors.cnames['darkslategrey'], colors.cnames['black'], 'k')]

    truth_color = 'g'

    spacing = np.array([0.1, 0.1, 0.05, 0.05, 0.2, 0.11])
    spacing_scale = 1

    def __init__(self, parameter_names, parameter_ranges, chains):

        """
        :param parameter_names: param names (dictionary)
        :param parameter_ranges: parameter limits (dictionary)
        :param samples: samples that form the probability distribution (numpy array)

        shape is (N_samples (tol), N_parameters (len(parameter_names)),
        N_realizations (n_pert), N_posteriors (n_lenses))

        """
        self.param_names = parameter_names
        self.parameter_ranges = parameter_ranges

        self.chains = chains

        self._computed_densities = {}

    def make_triplot(self, contour_colors=None, levels=[0.05, 0.22, 1],
                     filled_contours=True, contour_alpha=0.6, param_names=None,
                     fig_size=8, truths=None, load_from_file=True,
                     transpose_idx = None, bandwidth_scale = 0.7):

        self.fig = plt.figure(1)
        self._init(fig_size)

        axes = []
        counter = 1
        n_subplots = len(param_names)
        for row in range(n_subplots):
            for col in range(n_subplots):
                axes.append(plt.subplot(n_subplots, n_subplots, counter))
                counter += 1

        if contour_colors is None:
            contour_colors = self._default_contour_colors
        self._auto_scale = []
        for i, chain in enumerate(self.chains):
            self._make_triplot_i(chain, axes, i, contour_colors, levels, filled_contours, contour_alpha, param_names,
                                 fig_size, truths, load_from_file = load_from_file,
                                 transpose_idx = transpose_idx, bandwidth_scale = bandwidth_scale)

        for k in range(len(param_names)):
            scales = []
            for c in range(0,len(self.chains)):
                scales.append(self._auto_scale[c][k])
            maxh = np.max(scales) * 1.1
            axes[int((len(param_names)+1) * k)].set_ylim(0, maxh)

        self._auto_scale = []
        plt.subplots_adjust(left=self.spacing[0] * self.spacing_scale, bottom=self.spacing[1] * self.spacing_scale,
                            right=1 - self.spacing[2] * self.spacing_scale,
                            top=1 - self.spacing[3] * self.spacing_scale,
                            wspace=self.spacing[4] * self.spacing_scale, hspace=self.spacing[5] * self.spacing_scale)

    def _make_triplot_i(self, chain, axes, color_index, contour_colors=None, levels=[0.05, 0.22, 1],
                        filled_contours=True, contour_alpha=0.6, param_names=None, fig_size=8,
                        truths=None, labsize=15, tick_label_font=14,
                        load_from_file = True, transpose_idx=None, bandwidth_scale = 0.7):

        if param_names is None:
            param_names = self.param_names

        size_scale = len(param_names) * 0.1 + 1
        self.fig.set_size_inches(fig_size * size_scale, fig_size * size_scale)

        marg_in_row, plot_index = 0, 0
        n_subplots = len(param_names)
        self._reference_grid = None
        autoscale = []

        for row in range(n_subplots):

            marg_done = False
            for col in range(n_subplots):

                if col < marg_in_row:

                    density = chain.get_projection([param_names[row], param_names[col]], bandwidth_scale,
                                                   load_from_file=load_from_file)

                    if transpose_idx is not None and plot_index in transpose_idx:
                        print(param_names[row],param_names[col])
                        density = density.T

                    extent, aspect = self._extent_aspect([param_names[col], param_names[row]])
                    pmin1, pmax1 = extent[0], extent[1]
                    pmin2, pmax2 = extent[2], extent[3]

                    xtick_locs, xtick_labels, xlabel, rotation = self._ticks_and_labels(param_names[col])
                    ytick_locs, ytick_labels, ylabel, _ = self._ticks_and_labels(param_names[row])

                    if row == n_subplots - 1:

                        axes[plot_index].set_xticks(xtick_locs)
                        axes[plot_index].set_xticklabels(xtick_labels, fontsize=tick_label_font, rotation=rotation)

                        if col == 0:
                            axes[plot_index].set_yticks(ytick_locs)
                            axes[plot_index].set_yticklabels(ytick_labels, fontsize=tick_label_font)
                            axes[plot_index].set_ylabel(ylabel, fontsize=labsize)
                        else:
                            axes[plot_index].set_yticks([])
                            axes[plot_index].set_yticklabels([])
                        axes[plot_index].set_xlabel(xlabel, fontsize=labsize)


                    elif col == 0:
                        axes[plot_index].set_yticks(ytick_locs)
                        axes[plot_index].set_yticklabels(ytick_labels, fontsize=tick_label_font)
                        axes[plot_index].set_xticks([])
                        axes[plot_index].set_ylabel(ylabel, fontsize=labsize)

                    else:
                        axes[plot_index].set_xticks([])
                        axes[plot_index].set_yticks([])
                        axes[plot_index].set_xticklabels([])
                        axes[plot_index].set_yticklabels([])


                    if filled_contours:
                        coordsx = np.linspace(extent[0], extent[1], density.shape[0])
                        coordsy = np.linspace(extent[2], extent[3], density.shape[1])

                        axes[plot_index].imshow(density.T, extent=extent, aspect=aspect,
                                                origin='lower', cmap=self.cmap, alpha=0)
                        self._contours(coordsx, coordsy, density.T, axes[plot_index], extent=extent,
                                       contour_colors=contour_colors[color_index], contour_alpha=contour_alpha,
                                       levels=levels)
                        axes[plot_index].set_xlim(pmin1, pmax1)
                        axes[plot_index].set_ylim(pmin2, pmax2)


                    else:
                        axes[plot_index].imshow(density.T, origin='lower', cmap=self.cmap, alpha=1, vmin=0,
                                                vmax=np.max(density), aspect=aspect, extent=extent)
                        axes[plot_index].set_xlim(pmin1, pmax1)
                        axes[plot_index].set_ylim(pmin2, pmax2)

                    if truths is not None:
                        t1, t2 = truths[param_names[col]], truths[param_names[row]]
                        axes[plot_index].scatter(t1, t2, color=self.truth_color, s=50)
                        axes[plot_index].axvline(t1, linestyle='--', color=self.truth_color, linewidth=3)
                        axes[plot_index].axhline(t2, linestyle='--', color=self.truth_color, linewidth=3)

                elif marg_in_row == col and marg_done is False:

                    marg_done = True
                    marg_in_row += 1

                    density = chain.get_projection([param_names[col]], bandwidth_scale,
                                                   load_from_file)

                    xtick_locs, xtick_labels, xlabel, rotation = self._ticks_and_labels(param_names[col])
                    pmin, pmax = self._get_param_minmax(param_names[col])
                    coords = np.linspace(pmin, pmax, len(density))

                    bar_centers, bar_width, bar_heights = self._bar_plot_heights(density, coords, None)

                    bar_heights *= np.sum(bar_heights) ** -1 * len(bar_centers) ** -1
                    autoscale.append(np.max(bar_heights))

                    for i, y in enumerate(bar_heights):
                        x1, x2 = bar_centers[i] - bar_width * .5, bar_centers[i] + bar_width * .5

                        axes[plot_index].plot([x1, x2], [y, y], color=contour_colors[color_index][1],
                                              alpha=0.6)
                        axes[plot_index].fill_between([x1, x2], y, color=contour_colors[color_index][1],
                                                      alpha=0.6)
                        axes[plot_index].plot([x1, x1], [0, y], color=contour_colors[color_index][1],
                                              alpha=0.6)
                        axes[plot_index].plot([x2, x2], [0, y], color=contour_colors[color_index][1],
                                              alpha=0.6)
                    axes[plot_index].set_xlim(pmin, pmax)
                    #axes[plot_index].set_ylim(0, hmax * 1.1 * self._hmax_scale)
                    axes[plot_index].set_yticks([])

                    low95 = self._confidence_int(bar_centers, bar_heights, 0.05)
                    high95 = self._confidence_int(bar_centers, bar_heights, 0.95)

                    if low95 is not None:
                        axes[plot_index].axvline(low95, color=contour_colors[color_index][1],
                                                 alpha=0.8, linewidth=2.5, linestyle='-.')
                    if high95 is not None:
                        axes[plot_index].axvline(high95, color=contour_colors[color_index][1],
                                                 alpha=0.8, linewidth=2.5, linestyle='-.')

                    if col != n_subplots - 1:
                        axes[plot_index].set_xticks([])
                    else:
                        axes[plot_index].set_xticks(xtick_locs)
                        axes[plot_index].set_xticklabels(xtick_labels, fontsize=tick_label_font)
                        axes[plot_index].set_xlabel(xlabel, fontsize=labsize)

                    if truths is not None:

                        t = deepcopy(truths[param_names[col]])
                        pmin, pmax = self._get_param_minmax(param_names[col])
                        if t <= pmin:
                            t = pmin * 1.075

                        axes[plot_index].axvline(t, linestyle='--', color=self.truth_color, linewidth=3)

                else:
                    axes[plot_index].axis('off')

                plot_index += 1
        self._auto_scale.append(autoscale)

    def _confidence_int(self, centers, heights, percentile):

        total = np.sum(heights)
        summ, index = 0, 0
        while summ < total * percentile:
            summ += heights[index]
            index += 1

        if index == len(centers) or index == 1:
            return None

        return centers[index - 1]

    def _extent_aspect(self, param_names):

        aspect = (self.parameter_ranges[param_names[0]][1] - self.parameter_ranges[param_names[0]][0]) * \
                 (self.parameter_ranges[param_names[1]][1] - self.parameter_ranges[param_names[1]][0]) ** -1

        extent = [self.parameter_ranges[param_names[0]][0], self.parameter_ranges[param_names[0]][1],
                  self.parameter_ranges[param_names[1]][0],
                  self.parameter_ranges[param_names[1]][1]]

        return extent, aspect

    def _init(self, fig_size):

        self._tick_lab_font = 12 * fig_size * 7 ** -1
        self._label_font = 15 * fig_size * 7 ** -1
        plt.rcParams['axes.linewidth'] = 2.5 * fig_size * 7 ** -1

        plt.rcParams['xtick.major.width'] = 2.5 * fig_size * 7 ** -1
        plt.rcParams['xtick.major.size'] = 6 * fig_size * 7 ** -1
        plt.rcParams['xtick.minor.size'] = 2 * fig_size * 7 ** -1

        plt.rcParams['ytick.major.width'] = 2.5 * fig_size * 7 ** -1
        plt.rcParams['ytick.major.size'] = 6 * fig_size * 7 ** -1
        plt.rcParams['ytick.minor.size'] = 2 * fig_size * 7 ** -1

    def _get_param_minmax(self, pname):

        ranges = self.parameter_ranges[pname]
        return ranges[0], ranges[1]

    def _get_param_inds(self, params):

        inds = []

        for pi in params:

            for i, name in enumerate(self.param_names):

                if pi == name:
                    inds.append(i)
                    break

        return np.array(inds)

    def _bar_plot_heights(self, bar_heights, coords, rebin):

        if rebin is not None:
            new = []
            if len(bar_heights) % rebin == 0:
                fac = int(len(bar_heights) / rebin)
                for i in range(0, len(bar_heights), fac):
                    new.append(np.mean(bar_heights[i:(i + fac)]))

                bar_heights = np.array(new)
            else:
                raise ValueError('must be divisible by rebin.')

        bar_width = np.absolute(coords[-1] - coords[0]) * len(bar_heights) ** -1
        bar_centers = []
        for i in range(0, len(bar_heights)):
            bar_centers.append(coords[0] + bar_width * (0.5 + i))

        integral = np.sum(bar_heights) * bar_width * len(bar_centers) ** -1

        bar_heights = bar_heights * integral ** -1

        return bar_centers, bar_width, bar_heights

    def _contours(self, x, y, grid, ax, linewidths=4, filled_contours=True, contour_colors='',
                  contour_alpha=1, extent=None, levels=[0.05, 0.22, 1]):

        levels = np.array(levels) * np.max(grid)
        X, Y = np.meshgrid(x, y)

        if filled_contours:

            ax.contour(X, Y, grid, levels, extent=extent,
                       colors=contour_colors, linewidths=linewidths, zorder=1, linestyles=['dashed', 'solid'])

            ax.contourf(X, Y, grid, [levels[0], levels[1]], colors=[contour_colors[0], contour_colors[1]],
                        alpha=contour_alpha * 0.5, zorder=1,
                        extent=extent)

            ax.contourf(X, Y, grid, [levels[1], levels[2]], colors=[contour_colors[1], contour_colors[2]],
                        alpha=contour_alpha, zorder=1,
                        extent=extent)


        else:
            ax.contour(X, Y, grid, extent=extent, colors=contour_colors,
                       levels=np.array(levels) * np.max(grid),
                       linewidths=linewidths)

    def _ID_joint_params(self, target_params, params):

        if params[0] in target_params and params[1] in target_params:
            return True
        else:
            return False

    def _ticks_and_labels(self, pname):

        rotation = 0
        if pname == 'a0_area':
            name = r'$\Sigma_{\rm{sub}}\times 10^{2} \ \left[kpc^{-2}\right]$'
            tick_labels = [0, 0.9, 1.8, 2.7, 3.6, 4.5]
            tick_locs = [0, 0.9, 1.8, 2.7, 3.6, 4.5]
            rotation = 45
        elif pname == 'SIE_gamma':
            name = r'$\gamma_{\rm{macro}}$'
            tick_labels = [2, 2.05, 2.1, 2.15, 2.2]
            tick_locs = [2, 2.05, 2.1, 2.15, 2.2]
            rotation = 45
        elif pname == 'source_size_kpc':
            name = r'$\sigma_{\rm{src}}$'
            tick_labels = [25, 30, 35, 40, 45, 50]
            tick_locs = tick_labels
        elif pname == 'log_m_break':
            name = r'$\log_{10}{m_{\rm{hm}}}$'
            tick_labels = [5, 6, 7, 8, 9, 10]
            tick_locs = tick_labels
        elif pname == 'LOS_normalization':
            name = r'$\delta_{\rm{LOS}}$'
            tick_labels = [0.7, 0.85, 1.0, 1.15, 1.3]
            tick_locs = [0.7, 0.85, 1.0, 1.15, 1.3]

        return tick_locs, tick_labels, name, rotation
