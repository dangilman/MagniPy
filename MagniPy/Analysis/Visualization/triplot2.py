from matplotlib import colors
import matplotlib.pyplot as plt
from MagniPy.Analysis.KDE.kde import *
import numpy as np
import matplotlib.gridspec as gridspec


class TriPlot2(object):

    cmap = 'gist_heat'

    # default_contour_colors = (colors.cnames['orchid'], colors.cnames['darkviolet'], 'k')
    _default_contour_colors = [(colors.cnames['darkslategrey'], colors.cnames['black'], 'k'),
                               (colors.cnames['dodgerblue'], colors.cnames['blue'], 'k'),
                               (colors.cnames['orchid'], colors.cnames['darkviolet'], 'k'),
                               ]

    truth_color = 'g'

    spacing = np.array([0.1, 0.1, 0.05, 0.05, 0.2, 0.11])
    #spacing = np.array([0.49, 0.49, 0.5, 0.5, 0., 0.0])
    spacing_scale = 1.

    cmap_call = plt.get_cmap(cmap)
    _color_eval = 0.9

    def __init__(self, NDdensity_instance_list, parameter_names, parameter_ranges):


        self.param_names = parameter_names
        self._nchains = len(NDdensity_instance_list)

        if isinstance(parameter_ranges, list):
            self._prange_list = parameter_ranges
            self.parameter_ranges = {}
            for i, pname in enumerate(self.param_names):
                self.parameter_ranges.update({pname:parameter_ranges[i]})
        elif isinstance(parameter_ranges, dict):
            self.parameter_ranges = parameter_ranges
            self._prange_list = []
            for pi in self.param_names:
                self._prange_list.append(self.parameter_ranges[pi])

        self._NDdensity_list = NDdensity_instance_list

        #self.set_cmap(self.cmap)

    def _load_projection_1D(self, pname, idx):

        return self._NDdensity_list[idx].projection_1D(pname)

    def _load_projection_2D(self, p1, p2, idx):

        return self._NDdensity_list[idx].projection_2D(p1, p2)

    def set_cmap(self, newcmap, color_eval=0.9, marginal_col=None):

        self.cmap = newcmap
        self.cmap_call = plt.get_cmap(newcmap)
        self._color_eval = color_eval

        self._marginal_col = marginal_col

    def make_joint(self, p1, p2, contour_colors=None, levels=[0.05, 0.22, 1],
                   filled_contours=True, contour_alpha=0.6,
                   fig_size=8, label_scale=1, tick_label_font=12,
                     xtick_label_rotate=0, show_contours=True):

        self.fig = plt.figure(1)
        self._init(fig_size)
        ax = plt.subplot(111)

        if contour_colors is None:
            contour_colors = self._default_contour_colors

        for i in range(self._nchains):
            axes = self._make_joint_i(p1, p2, ax, i, contour_colors=contour_colors, levels=levels,
                      filled_contours=filled_contours, contour_alpha=contour_alpha,
                      labsize=15*label_scale, tick_label_font=tick_label_font,
                               xtick_label_rotate=xtick_label_rotate, show_contours=show_contours)
        return axes

    def make_triplot(self, contour_colors=None, levels=[0.05, 0.22, 1],
                     filled_contours=True, contour_alpha=0.6, param_names=None,
                     fig_size=8, truths=None, load_from_file=True,
                     transpose_idx=None, bandwidth_scale=0.7, label_scale=1, tick_label_font=12,
                     xtick_label_rotate=0, show_contours=False, marginal_alpha=0.6, show_intervals=True):

        self.fig = plt.figure(1)

        self._init(fig_size)

        axes = []
        counter = 1
        n_subplots = len(param_names)

        gs1 = gridspec.GridSpec(n_subplots, n_subplots)
        gs1.update(wspace=0.15, hspace=0.15)

        for row in range(n_subplots):
            for col in range(n_subplots):
                #axes.append(plt.subplot(n_subplots, n_subplots, counter))
                axes.append(plt.subplot(gs1[counter-1]))
                counter += 1

        if contour_colors is None:
            contour_colors = self._default_contour_colors
        self._auto_scale = []

        for i in range(self._nchains):
            axes.append(self._make_triplot_i(axes, i, contour_colors, levels, filled_contours, contour_alpha, param_names,
                                 fig_size, truths, load_from_file=load_from_file, tick_label_font=tick_label_font,
                                 transpose_idx=transpose_idx, bandwidth_scale=bandwidth_scale, xtick_label_rotate=xtick_label_rotate,
                                 label_scale=label_scale, cmap=self.cmap_call, show_contours=show_contours,
                                             marginal_alpha=marginal_alpha, show_intervals=show_intervals))

        for k in range(len(param_names)):
            scales = []
            for c in range(0, self._nchains):
                scales.append(self._auto_scale[c][k])
            maxh = np.max(scales) * 1.1
            print(maxh)
            axes[int((len(param_names) + 1) * k)].set_ylim(0, maxh)

        self._auto_scale = []
        plt.subplots_adjust(left=self.spacing[0] * self.spacing_scale, bottom=self.spacing[1] * self.spacing_scale,
                            right=1 - self.spacing[2] * self.spacing_scale,
                            top=1 - self.spacing[3] * self.spacing_scale,
                            wspace=self.spacing[4] * self.spacing_scale, hspace=self.spacing[5] * self.spacing_scale)

        return axes

    def make_marginal(self, p1, contour_colors=None, levels=[0.05, 0.22, 1],
                      filled_contours=True, contour_alpha=0.6, param_names=None,
                      fig_size=8, truths=None, load_from_file=True,
                      transpose_idx=None, bandwidth_scale=0.7, label_scale=1,
                      cmap=None, xticklabel_rotate=0, bar_alpha=0.7, bar_colors=['k','m','g','r'],
                      height_scale=1.1, show_low=False, show_high=False):

        self.fig = plt.figure(1)
        self._init(fig_size)
        ax = plt.subplot(111)
        self._auto_scale = []

        if contour_colors is None:
            contour_colors = self._default_contour_colors
        self._auto_scale = []
        for i in range(self._nchains):
            out = self._make_marginal_i(p1, ax, i, contour_colors, levels, filled_contours, contour_alpha, param_names,
                                  fig_size, truths, load_from_file=load_from_file,
                                  transpose_idx=transpose_idx, bandwidth_scale=bandwidth_scale,
                                  label_scale=label_scale, cmap=cmap, xticklabel_rotate=xticklabel_rotate,
                                  bar_alpha=bar_alpha, bar_color=bar_colors[i], show_low=show_low, show_high=show_high)

        scales = []
        for c in range(0, self._nchains):
            scales.append(self._auto_scale[c][0])
        maxh = np.max(scales) * height_scale
        ax.set_ylim(0, maxh)
        pmin, pmax = self._get_param_minmax(p1)
        asp = maxh * (pmax - pmin) ** -1
        ax.set_aspect(asp ** -1)

        self._auto_scale = []

        return out

    def _make_marginal_i(self, p1, ax, color_index, contour_colors=None, levels=[0.05, 0.22, 1],
                         filled_contours=True, contour_alpha=0.6, param_names=None, fig_size=8,
                         truths=None, labsize=15, tick_label_font=14,
                         load_from_file=True, transpose_idx=None,
                         bandwidth_scale=0.7, label_scale=None, cmap=None, xticklabel_rotate=0,
                         bar_alpha=0.7, bar_color=None, show_low=False, show_high=False):

        autoscale = []

        density = self._load_projection_1D(p1, color_index)

        xtick_locs, xtick_labels, xlabel, rotation = self._ticks_and_labels(p1)
        pmin, pmax = self._get_param_minmax(p1)

        coords = np.linspace(pmin, pmax, len(density))

        bar_centers, bar_width, bar_heights = self._bar_plot_heights(density, coords, None)

        bar_heights *= np.sum(bar_heights) ** -1 * len(bar_centers) ** -1
        autoscale.append(np.max(bar_heights))

        max_idx = np.argmax(bar_heights)
        max_h = bar_heights[max_idx]

        for bin, bh in enumerate(bar_heights):
            print('probability '+str(np.round(bar_centers[bin], 3))+': ', np.round(bh/max_h,3))
        print('value: ', repr(np.round(bar_centers,3)))
        print('prob: ', repr(np.round(bar_heights/max_h,3)))

        for i, y in enumerate(bar_heights):
            x1, x2 = bar_centers[i] - bar_width * .5, bar_centers[i] + bar_width * .5

            ax.plot([x1, x2], [y, y], color=bar_color,
                        alpha=bar_alpha)
            ax.fill_between([x1, x2], y, color=bar_color,
                            alpha=0.6)
            ax.plot([x1, x1], [0, y], color=bar_color,
                    alpha=bar_alpha)
            ax.plot([x2, x2], [0, y], color=bar_color,
                    alpha=bar_alpha)

        ax.set_xlim(pmin, pmax)

        ax.set_yticks([])

        low95 = self._confidence_int(bar_centers, bar_heights, 0.05)
        high95 = self._confidence_int(bar_centers, bar_heights, 0.95)

        low68 = self._confidence_int(bar_centers, bar_heights, 0.32)
        high68 = self._confidence_int(bar_centers, bar_heights, 0.68)

        print('low/high68:' + str(low68) + ' ' + str(high68))
        print('low/high95:' + str(low95) + ' ' + str(high95))
        mean_of_distribution = 0
        for i in range(0, len(bar_heights)):
            mean_of_distribution += bar_heights[i] * bar_centers[i] / np.sum(bar_heights)
        print('mean: ', mean_of_distribution)

        if low95 is not None and show_low:
            ax.axvline(low95, color=bar_color,
                       alpha=0.8, linewidth=2.5, linestyle='-.')
        if high95 is not None and show_high:
            ax.axvline(high95, color=bar_color,
                       alpha=0.8, linewidth=2.5, linestyle='-.')

        ax.set_xticks(xtick_locs)
        ax.set_xticklabels(xtick_labels, fontsize=tick_label_font, rotation=xticklabel_rotate)
        if xlabel == r'$\frac{r_{\rm{core}}}{r_s}$':
            ax.set_xlabel(xlabel, fontsize=40 * label_scale)
        else:
            ax.set_xlabel(xlabel, fontsize=labsize * label_scale)

        if truths is not None:

            t = deepcopy(truths[p1])

            if isinstance(t, float) or isinstance(t, int):
                pmin, pmax = self._get_param_minmax(p1)
                if t <= pmin:
                    t = pmin * 1.075

                ax.axvline(t, linestyle='--', color=self.truth_color, linewidth=3)
            elif isinstance(t, list):
                ax.axvspan(t[0], t[1], alpha=0.25, color=self.truth_color)

        self._auto_scale.append(autoscale)

        return ax

    def _make_joint_i(self, p1, p2, ax, color_index, contour_colors=None, levels=[0.05, 0.22, 1],
                      filled_contours=True, contour_alpha=0.6, labsize=None, tick_label_font=None,
                               xtick_label_rotate=None, show_contours=None):

        density = self._load_projection_2D(p1, p2, color_index)

        extent, aspect = self._extent_aspect([p1, p2])
        pmin1, pmax1 = extent[0], extent[1]
        pmin2, pmax2 = extent[2], extent[3]

        xtick_locs, xtick_labels, xlabel, rotation = self._ticks_and_labels(p1)
        ytick_locs, ytick_labels, ylabel, _ = self._ticks_and_labels(p2)

        if filled_contours:
            coordsx = np.linspace(extent[0], extent[1], density.shape[0])
            coordsy = np.linspace(extent[2], extent[3], density.shape[1])

            ax.imshow(density, extent=extent, aspect=aspect,
                      origin='lower', cmap=self.cmap, alpha=0)
            self._contours(coordsx, coordsy, density, ax, extent=extent,
                           contour_colors=contour_colors[color_index], contour_alpha=contour_alpha,
                           levels=levels)
            ax.set_xlim(pmin1, pmax1)
            ax.set_ylim(pmin2, pmax2)

        else:
            coordsx = np.linspace(extent[0], extent[1], density.shape[0])
            coordsy = np.linspace(extent[2], extent[3], density.shape[1])
            ax.imshow(density, origin='lower', cmap=self.cmap, alpha=1, vmin=0,
                      vmax=np.max(density), aspect=aspect, extent=extent)
            if show_contours:
                self._contours(coordsx, coordsy, density, ax, extent=extent, filled_contours=False,
                           contour_colors=contour_colors[color_index], contour_alpha=contour_alpha,
                           levels=levels)
            ax.set_xlim(pmin1, pmax1)
            ax.set_ylim(pmin2, pmax2)

        ax.set_xticks(xtick_locs)
        ax.set_xticklabels(xtick_labels, fontsize=tick_label_font, rotation=xtick_label_rotate)

        ax.set_yticks(ytick_locs)
        ax.set_yticklabels(ytick_labels, fontsize=tick_label_font)

        if xlabel == r'$\frac{r_{\rm{core}}}{r_s}$':
            ax.set_xlabel(xlabel, fontsize=40)
        elif ylabel == r'$\frac{r_{\rm{core}}}{r_s}$':
            ax.set_ylabel(ylabel, fontsize=40)
        else:
            ax.set_xlabel(xlabel, fontsize=labsize)
            ax.set_ylabel(ylabel, fontsize=labsize)

        return ax

    def _make_triplot_i(self, axes, color_index, contour_colors=None, levels=[0.05, 0.22, 1],
                        filled_contours=True, contour_alpha=0.6, param_names=None, fig_size=8,
                        truths=None, labsize=15, tick_label_font=14, xtick_label_rotate=0,
                        load_from_file=True, transpose_idx=None,
                        bandwidth_scale=0.7, label_scale=None, cmap=None,
                        show_contours=False, marginal_alpha=0.9, show_intervals=True):

        if param_names is None:
            param_names = self.param_names

        size_scale = len(param_names) * 0.1 + 1
        self.fig.set_size_inches(fig_size * size_scale, fig_size * size_scale)

        marg_in_row, plot_index = 0, 0
        n_subplots = len(param_names)
        self._reference_grid = None
        autoscale = []

        self.triplot_densities = []
        self.joint_names = []

        for row in range(n_subplots):

            marg_done = False
            for col in range(n_subplots):

                if col < marg_in_row:

                    density = self._load_projection_2D(param_names[row], param_names[col], color_index)

                    self.triplot_densities.append(density)
                    self.joint_names.append(param_names[row]+'_'+param_names[col])

                    if transpose_idx is not None and plot_index in transpose_idx:
                        print(param_names[row], param_names[col])
                        density = density.T

                    extent, aspect = self._extent_aspect([param_names[col], param_names[row]])
                    pmin1, pmax1 = extent[0], extent[1]
                    pmin2, pmax2 = extent[2], extent[3]

                    xtick_locs, xtick_labels, xlabel, rotation = self._ticks_and_labels(param_names[col])
                    ytick_locs, ytick_labels, ylabel, _ = self._ticks_and_labels(param_names[row])

                    if row == n_subplots - 1:

                        axes[plot_index].set_xticks(xtick_locs)
                        axes[plot_index].set_xticklabels(xtick_labels, fontsize=tick_label_font,
                                                         rotation=xtick_label_rotate)

                        if col == 0:
                            axes[plot_index].set_yticks(ytick_locs)
                            axes[plot_index].set_yticklabels(ytick_labels, fontsize=tick_label_font)
                            axes[plot_index].set_ylabel(ylabel, fontsize=labsize * label_scale)
                        else:
                            axes[plot_index].set_yticks([])
                            axes[plot_index].set_yticklabels([])

                        if xlabel == r'$\frac{r_{\rm{core}}}{r_s}$':
                            axes[plot_index].set_xlabel(xlabel, fontsize=25 * label_scale)
                        else:
                            axes[plot_index].set_xlabel(xlabel, fontsize=labsize * label_scale)


                    elif col == 0:
                        axes[plot_index].set_yticks(ytick_locs)
                        axes[plot_index].set_yticklabels(ytick_labels, fontsize=tick_label_font)
                        axes[plot_index].set_xticks([])
                        if ylabel == r'$\frac{r_{\rm{core}}}{r_s}$':
                            axes[plot_index].set_ylabel(ylabel, fontsize=25 * label_scale)
                        else:
                            axes[plot_index].set_ylabel(ylabel, fontsize=labsize * label_scale)

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
                        if show_contours:
                            coordsx = np.linspace(extent[0], extent[1], density.shape[0])
                            coordsy = np.linspace(extent[2], extent[3], density.shape[1])
                            self._contours(coordsx, coordsy, density.T, axes[plot_index], filled_contours=False, extent=extent,
                                       contour_colors=contour_colors[color_index], contour_alpha=contour_alpha,
                                       levels=levels)
                        axes[plot_index].set_xlim(pmin1, pmax1)
                        axes[plot_index].set_ylim(pmin2, pmax2)

                    if truths is not None:
                        t1, t2 = truths[param_names[col]], truths[param_names[row]]
                        if isinstance(t1, list):
                            t_1 = 0.5*(t1[0]+t1[1])

                        else:
                            t_1 = t1
                        if isinstance(t2, list):
                            t_2 = 0.5*(t2[0] + t2[1])
                        else:
                            t_2 = t2

                        axes[plot_index].scatter(t_1, t_2, color=self.truth_color, s=50)
                        axes[plot_index].axvline(t_1, linestyle='--', color=self.truth_color, linewidth=3)
                        axes[plot_index].axhline(t_2, linestyle='--', color=self.truth_color, linewidth=3)

                elif marg_in_row == col and marg_done is False:

                    marg_done = True
                    marg_in_row += 1

                    # density = chain.get_projection([param_names[col]], bandwidth_scale,
                    #                               load_from_file)
                    density = self._load_projection_1D(param_names[col], color_index)

                    xtick_locs, xtick_labels, xlabel, rotation = self._ticks_and_labels(param_names[col])
                    pmin, pmax = self._get_param_minmax(param_names[col])
                    coords = np.linspace(pmin, pmax, len(density))

                    bar_centers, bar_width, bar_heights = self._bar_plot_heights(density, coords, None)

                    bar_heights *= np.sum(bar_heights) ** -1 * len(bar_centers) ** -1
                    autoscale.append(np.max(bar_heights))

                    for i, y in enumerate(bar_heights):
                        x1, x2 = bar_centers[i] - bar_width * .5, bar_centers[i] + bar_width * .5

                        if filled_contours:
                            axes[plot_index].plot([x1, x2], [y, y], color=contour_colors[color_index][1],
                                                  alpha=1)
                            axes[plot_index].fill_between([x1, x2], y, color=contour_colors[color_index][1],
                                                          alpha=marginal_alpha)
                            axes[plot_index].plot([x1, x1], [0, y], color=contour_colors[color_index][1],
                                                  alpha=1)
                            axes[plot_index].plot([x2, x2], [0, y], color=contour_colors[color_index][1],
                                                  alpha=1)
                        else:
                            if self._marginal_col is None:
                                marginal_col = cmap(self._color_eval)
                            else:
                                marginal_col = self._marginal_col
                            axes[plot_index].plot([x1, x2], [y, y], color=marginal_col,
                                                  alpha=1)
                            axes[plot_index].fill_between([x1, x2], y, color=marginal_col,
                                                          alpha=marginal_alpha)
                            axes[plot_index].plot([x1, x1], [0, y], color=marginal_col,
                                                  alpha=1)
                            axes[plot_index].plot([x2, x2], [0, y], color=marginal_col,
                                                  alpha=1)

                    axes[plot_index].set_xlim(pmin, pmax)
                    # axes[plot_index].set_ylim(0, hmax * 1.1 * self._hmax_scale)
                    axes[plot_index].set_yticks([])

                    low95 = self._confidence_int(bar_centers, bar_heights, 0.05)
                    high95 = self._confidence_int(bar_centers, bar_heights, 0.95)

                    low68 = self._confidence_int(bar_centers, bar_heights, 0.32)
                    high68 = self._confidence_int(bar_centers, bar_heights, 0.68)

                    if param_names[col] in ['log_m_break',r'$m_{\rm{hm}}$']:
                        print('half-mode mass: ')
                        print(low95, high95)
                        print(low68, high68)
                    if param_names[col] in ['sigma_sub',r'$\Sigma_{\rm{sub}}$']:
                        print('sigma-sub: ')
                        print(low95, high95)
                        print(low68, high68)
                    if param_names[col] in ['$\\alpha$', 'alpha']:
                        print('alpha: ')
                        print(low95, high95)
                        print(low68, high68)
                        print(self._confidence_int(bar_centers, bar_heights, 0.5))

                    if low95 is not None and show_intervals:
                        axes[plot_index].axvline(low95, color=contour_colors[color_index][1],
                                                 alpha=0.8, linewidth=2.5, linestyle='-.')
                    if high95 is not None and show_intervals:
                        axes[plot_index].axvline(high95, color=contour_colors[color_index][1],
                                                 alpha=0.8, linewidth=2.5, linestyle='-.')


                    if col != n_subplots - 1:
                        axes[plot_index].set_xticks([])
                    else:
                        axes[plot_index].set_xticks(xtick_locs)
                        axes[plot_index].set_xticklabels(xtick_labels, fontsize=tick_label_font, rotation=xtick_label_rotate)
                        axes[plot_index].set_xlabel(xlabel, fontsize=labsize * label_scale)

                    if truths is not None:

                        t = deepcopy(truths[param_names[col]])
                        pmin, pmax = self._get_param_minmax(param_names[col])
                        if isinstance(t, float) or isinstance(t, int):
                            if t <= pmin:
                                t_ = pmin * 1.075
                            else:
                                t_ = t
                            axes[plot_index].axvline(t_, linestyle='--', color=self.truth_color, linewidth=3)

                        else:
                            t_ = 0.5*(t[0] + t[1])
                            axes[plot_index].axvline(t_, linestyle='--', color=self.truth_color, linewidth=3)
                            axes[plot_index].axvspan(t[0], t[1], color=self.truth_color, alpha=0.25)

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

        # if index == len(centers) or index == 1:
        #    return None

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
                  contour_alpha=1, extent=None, levels=[0.05, 0.32, 1]):

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
            ax.contour(X, Y, grid, extent=extent, colors=contour_colors, zorder=1,
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
            tick_locs = np.array([0, 0.9, 1.8, 2.7, 3.6, 4.5]) * 0.01
            rotation = 45
        elif pname == r'$\Sigma_{\rm{sub}}$':
            name = r'$\Sigma_{\rm{sub}}\times 10^{2} \ \left[kpc^{-2}\right]$'
            #tick_labels = [0, 2, 4, 6, 8, 10]
            tick_locs = np.array([0.2, 1.8, 3.4, 5., 6.6, 8.2, 9.8]) * 0.01
            tick_labels = np.array([0.2, 1.8, 3.4, 5., 6.6, 8.2, 9.8])
            #tick_labels = [0.2, 2, 4, 6, 8, 10]

            rotation = 45
        elif pname == 'SIE_gamma':
            name = r'$\gamma_{\rm{macro}}$'
            tick_labels = [2, 2.05, 2.1, 2.15, 2.2]
            tick_locs = [2, 2.05, 2.1, 2.15, 2.2]
            rotation = 45
        elif pname == 'source_size_kpc' or pname == r'$\sigma_{\rm{src}}$':
            name = r'$\sigma_{\rm{src}} \ \left[\rm{pc}\right]$'
            tick_labels = [30, 40, 50, 60]
            tick_locs = np.array(tick_labels) * 0.001
        elif pname == 'log_m_break' or pname == r'$m_{\rm{hm}}$':
            name = r'$\log_{10} \left(m_{\rm{hm}}\right) \left[M_{\odot}\right]$'
            tick_labels = [5, 6, 7, 8, 9, 10]
            tick_locs = tick_labels
        elif pname == 'LOS_normalization':
            name = r'$\delta_{\rm{LOS}}$'
            tick_labels = [0.7, 0.85, 1.0, 1.15, 1.3]
            tick_locs = [0.7, 0.85, 1.0, 1.15, 1.3]
        elif pname == 'core_ratio':
            name = r'$\frac{r_{\rm{core}}}{r_s}$'
            tick_labels = [0.01, 0.2, 0.4, 0.6, 0.8]
            tick_locs = [0.01, 0.2, 0.4, 0.6, 0.8]
        elif pname == 'SIDMcross':
            name = r'$\sigma_0 \left[\rm{cm^2} \ \rm{g^{-1}}\right]$'
            tick_labels = [0.01, 2, 4, 6, 8, 10]
            tick_locs = [0.01, 2, 4, 6, 8, 10]
        elif pname == r'$\log M_{\rm{halo}}$':
            name = r'$\log_{10} M_{\rm{halo}} \left[M_{\odot}\right]$'
            tick_labels = [12.9, 13.1, 13.3, 13.5, 13.7]
            tick_locs = [12.9, 13.1, 13.3, 13.5, 13.7]
        else:
            name = pname

            tick_locs = np.round(np.linspace(self.parameter_ranges[pname][0], self.parameter_ranges[pname][1], 5), 2)
            tick_labels = tick_locs

        return tick_locs, tick_labels, name, rotation
