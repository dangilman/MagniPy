from MagniPy.Analysis.Visualization.Joint2D import Joint2D
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib import colors
from scipy.signal import resample


class Density1D(Joint2D):

    default_contour_colors = [(colors.cnames['grey'], colors.cnames['black'], 'k'),
                              (colors.cnames['skyblue'], colors.cnames['blue'], 'k'),
                              (colors.cnames['coral'], 'r', 'k'),
                              (colors.cnames['orchid'], colors.cnames['darkviolet'], 'k')]

    def make_plot_1D(self, sim_densities, param, param_ranges,
                     contour_colors=None, contour_alpha=0.6, xlabel_on=False,
                     truths=None, rebin=15, label_size=18, tick_label_font=12):

        if contour_colors is None:
            contour_colors = self.default_contour_colors

        max_height = 0

        for color_index, sim_density in enumerate(sim_densities):

            marginalized_density = self._compute_single(sim_density)

            bar_centers, bar_width, bar_heights = self._bar_plot_heights(marginalized_density,
                                                    np.linspace(param_ranges[0],param_ranges[1], 10), rebin)

            high_95 = self._quick_confidence(bar_centers, bar_heights, 0.95)
            low_95 = self._quick_confidence(bar_centers, bar_heights, 0.05)
            print(high_95)
            print(low_95)
            if param == 'log_m_break' and truths[param]>5:
                info = zip(bar_centers, bar_heights)
                print('likelihoods: '+str(list(info)))
            if max(bar_heights) > max_height:
                max_height = max(bar_heights)

            for i, h in enumerate(bar_heights):
                x1, x2, y = bar_centers[i] - bar_width * .5, bar_centers[i] + bar_width * .5, h
                self.ax.plot([x1, x2], [y, y], color=contour_colors[color_index][1])
                self.ax.fill_between([x1, x2], y, color=contour_colors[color_index][1], alpha=contour_alpha)
                self.ax.plot([x1, x1], [0, y], color=contour_colors[color_index][1])
                self.ax.plot([x2, x2], [0, y], color=contour_colors[color_index][1])

            self.ax.set_yticklabels([])
            self.ax.set_yticks([])

            if np.absolute(low_95 - truths[param]) < 0.4 and param == 'log_m_break' and truths[param]<5.5:
                pass
            else:
                self.ax.axvline(low_95, color=contour_colors[color_index][1], linestyle = '-.', linewidth = 3, alpha=0.8)
            self.ax.axvline(high_95, color=contour_colors[color_index][1], linestyle='-.', linewidth=3, alpha = 0.8)

            if xlabel_on:

                xticks = np.linspace(param_ranges[0], param_ranges[1], 5)
                label_name, ticks, tick_labels = self._convert_param_names(param, xticks)
                self.ax.set_xticks(xticks)

                if param == 'source_size_kpc':
                    self.ax.set_xticklabels(tick_labels.astype(int), fontsize = tick_label_font)
                elif param == 'a0_area':

                    self.ax.set_yticklabels(np.round(np.array(tick_labels), 1), fontsize=tick_label_font)
                else:
                    if param == 'a0_area':
                        rotation = 0
                    else:
                        rotation = 0
                    self.ax.set_xticklabels(tick_labels, fontsize=tick_label_font, rotation = rotation)
                    self.ax.xaxis.set_major_formatter(FormatStrFormatter(self._tick_formatter(pname=label_name)))

                self.ax.set_xlabel(label_name, fontsize=label_size)

            else:
                self.ax.set_xticks([])
                self.ax.set_xticklabels([])

        self.ax.set_xlim(param_ranges[0], param_ranges[1])
        self.ax.set_ylim(0, max_height*1.25)

        if truths is not None:
            if truths[param] is not None:
                if truths[param] < param_ranges[0]:
                    self.ax.axvline(param_ranges[0]*1.05, color='r', linestyle='--', linewidth=3)
                else:
                    self.ax.axvline(truths[param], color='r', linestyle='--',linewidth=3)

        self.ax.set_aspect((bar_centers[-1] - bar_centers[0])*(max_height*1.15)**-1)

        return

    def _compute_bayes_factor(self, posterior, param_ranges, cut, rebin):

        bayes_factors = []
        for color_index, sim_density in enumerate(posterior):

            marginalized_density = self._compute_single(sim_density)
            bar_centers, bar_width, bar_heights = self._bar_plot_heights(marginalized_density,
                                 np.linspace(param_ranges[0], param_ranges[1],10), rebin)
            bar_heights = np.array(bar_heights)
            bar_centers = np.array(bar_centers)
            total = np.sum(bar_heights)
            volume_low = cut - param_ranges[0]
            volume_high = (param_ranges[1] - param_ranges[0]) - volume_low

            summ = 0
            center_cut = np.argmin(np.absolute(bar_centers - cut))
            for i in range(0, center_cut):
                summ += bar_heights[i]

            prob_low = summ
            prob_high = (total - summ)

            bayes_factors.append(prob_high / prob_low)

        return bayes_factors

    def _bar_plot_heights(self, bar_heights, coords, rebin):

        if rebin is not None:
            new = []
            if len(bar_heights) % rebin == 0:
                fac = int(len(bar_heights) / rebin)
                for i in range(0, len(bar_heights), fac):
                    new.append(np.mean(bar_heights[i:(i + fac)]))

                bar_heights = np.array(new)
            else:
                print('resampling: ')
                print('length marginalized: ', len(bar_heights))
                print('new length: ', rebin)
                #raise ValueError('not yet implemented')
                bar_heights = resample(bar_heights, rebin)

        bar_width = np.absolute(coords[-1] - coords[0]) * len(bar_heights) ** -1
        bar_centers = []
        for i in range(0, len(bar_heights)):
            bar_centers.append(coords[0] + bar_width * (0.5 + i))

        integral = np.sum(bar_heights) * bar_width * len(bar_centers) ** -1

        bar_heights = bar_heights * integral ** -1

        return bar_centers, bar_width, bar_heights




