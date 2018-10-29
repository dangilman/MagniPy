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

    def make_marginalized(self, marginalized_densities, param, param_ranges,
                          contour_colors=None, contour_alpha=0.6, xlabel_on=False, truths=None):

        if contour_colors is None:
            contour_colors = self.default_contour_colors

        max_height = 0

        for color_index, sim_density in enumerate(marginalized_densities):

            bar_centers, bar_width, bar_heights = self._bar_plot_heights(marginalized_densities[color_index],
                                                    np.linspace(param_ranges[0],param_ranges[1], 10), None)

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

            if xlabel_on:
                self.ax.set_xlabel(param)
                xticks = np.linspace(param_ranges[0], param_ranges[1], 6)
                xtick_labels = [str(tick) for tick in xticks]
                self.ax.set_xticks(xticks)
                self.ax.set_xticklabels(xtick_labels, fontsize = 12)
                self.ax.xaxis.set_major_formatter(FormatStrFormatter(self._tick_formatter(pname=param)))
            else:
                self.ax.set_xticks([])
                self.ax.set_xticklabels([])

        self.ax.set_xlim(param_ranges[0], param_ranges[1])
        self.ax.set_ylim(0, max_height*1.25)

        if truths is not None:
            self.ax.axvline(truths[param], color='r', linestyle='--',linewidth=3)

        return

    def _bar_plot_heights(self, bar_heights, coords, rebin):

        if rebin is not None:
            new = []
            if len(bar_heights) % rebin == 0:
                fac = int(len(bar_heights) / rebin)
                for i in range(0, len(bar_heights), fac):
                    new.append(np.mean(bar_heights[i:(i + fac)]))

                bar_heights = new
            else:
                bar_heights = resample(bar_heights, rebin)

        bar_width = np.absolute(coords[-1] - coords[0]) * len(bar_heights) ** -1
        bar_centers = []
        for i in range(0, len(bar_heights)):
            bar_centers.append(coords[0] + bar_width * (0.5 + i))

        integral = np.sum(bar_heights) * bar_width * len(bar_centers) ** -1

        bar_heights = bar_heights * integral ** -1

        return bar_centers, bar_width, bar_heights

