import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from MagniPy.util import confidence_interval

class Joint2D:

    plt.rcParams['axes.linewidth'] = 2.5

    plt.rcParams['xtick.major.width'] = 2.5
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.minor.size'] = 2

    plt.rcParams['ytick.major.width'] = 2.5
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.minor.size'] = 2

    cmap = 'gist_heat_r'

    #default_contour_colors = (colors.cnames['orchid'], colors.cnames['darkviolet'], 'k')
    default_contour_colors = [(colors.cnames['grey'], colors.cnames['black'], 'k'),
                                (colors.cnames['skyblue'], colors.cnames['blue'], 'k'),
                              (colors.cnames['coral'], 'r', 'k'),
                              (colors.cnames['orchid'], colors.cnames['darkviolet'], 'k')]
    truth_color = 'r'
    tick_font = 12

    def __init__(self,densities=[],ax=None,fig=None):

        if not isinstance(densities,list):
            densities = [densities]

        self.simulation_densities = densities

        if fig is None:
            fig = plt.figure(1)
        if ax is None:
            ax = plt.subplot(111)

        self.fig = fig
        self.ax = ax

    def make_plot(self, xtick_labels=None, xticks=None, param_names = None,
                  param_ranges=None,ytick_labels=None, yticks=None,filled_contours=True, contour_colors=None, contour_alpha=0.6,
                  tick_label_font=12, **kwargs):

        if contour_colors is None:
            contour_colors = self.default_contour_colors

        aspect = (param_ranges[param_names[0]][1] - param_ranges[param_names[0]][0]) * \
                 (param_ranges[param_names[1]][1] - param_ranges[param_names[1]][0]) ** -1

        extent = [param_ranges[param_names[0]][0],param_ranges[param_names[0]][1],param_ranges[param_names[1]][0],
                  param_ranges[param_names[1]][1]]

        for idx,densities in enumerate(self.simulation_densities):

            final_density = np.ones_like(densities[0])

            for single_density in densities:

                final_density *= self._norm_density(single_density)

            final_density = self._norm_density(final_density)

            if filled_contours:

                x,y = np.linspace(extent[0],extent[1],final_density.shape[0]),np.linspace(extent[2],extent[3],final_density.shape[0])
                self.contours(x,y,final_density,contour_colors=contour_colors[idx],contour_alpha=contour_alpha, extent=extent,aspect=aspect)

                self.ax.imshow(final_density, extent=extent,
                               aspect=aspect, origin='lower', cmap=self.cmap, alpha=0)

            else:

                self.ax.imshow(final_density, extent=extent,
                               aspect=aspect, origin='lower', cmap=self.cmap, alpha=1,vmin=0,vmax=np.max(final_density))


        if 'truths' in kwargs:

            truth1,truth2 = kwargs['truths'][param_names[0]],kwargs['truths'][param_names[1]]

            inside1 = truth1 >= param_ranges[param_names[0]][0] and \
                              truth1 <= param_ranges[param_names[0]][1]
            inside2 = truth2 >= param_ranges[param_names[1]][0] and \
                              truth2 <= param_ranges[param_names[1]][1]

            if inside1:
                self.ax.axvline(truth1,color=self.truth_color,linestyle='--',linewidth=3)
            if inside2:
                self.ax.axhline(truth2,color=self.truth_color,linestyle='--',linewidth=3)
            if inside1 and inside2:
                self.ax.scatter(truth1,truth2,color=self.truth_color,s=25)

        self.ax.set_xlabel(param_names[0])
        self.ax.set_ylabel(param_names[1])

        if xticks is None:
            xticks = np.linspace(param_ranges[param_names[0]][0], param_ranges[param_names[0]][1], 6)
            xtick_labels = [str(tick) for tick in xticks]
        if yticks is None:
            yticks = np.linspace(param_ranges[param_names[1]][0],param_ranges[param_names[1]][1],6)
            ytick_labels = [str(tick) for tick in yticks]

        self.ax.set_yticks(yticks)
        self.ax.set_xticks(xticks)
        self.ax.set_yticklabels(ytick_labels)
        self.ax.set_xticklabels(xtick_labels)

        self.ax.xaxis.set_major_formatter(FormatStrFormatter(self._tick_formatter(pname=param_names[0])))
        self.ax.yaxis.set_major_formatter(FormatStrFormatter(self._tick_formatter(pname=param_names[1])))

        self.param_names = param_names
        self.param_ranges = param_ranges

        return self.ax

    def _norm_density(self,density):

        return density*(np.sum(density)*density.shape[0]**2)**-1

    def contours(self, x,y,grid, linewidths=4, filled_contours=True,contour_colors='',
                 contour_alpha=1,extent=None,aspect=None):

        levels = [0.05,0.22,1]
        levels = np.array(levels)*np.max(grid)
        X, Y = np.meshgrid(x, y)

        if filled_contours:

            plt.contour(X, Y, grid, levels, extent=extent,
                              colors=contour_colors, linewidths=linewidths, zorder=1)
            plt.contourf(X, Y, grid, levels, colors=contour_colors, alpha=contour_alpha, zorder=1,
                         extent=extent, aspect=aspect)


        else:
            plt.contour(X, Y, grid, extent=extent, colors=contour_colors,
                                  levels=np.array(levels) * np.max(grid), linewidths=linewidths)

    def _get_1d(self,pname,density):

        if self.param_names[0] == pname:
            axis = 1
        else:
            axis = 0

        return np.sum(density,axis=axis)

    def marginalize(self, param_name, rebin=25):

        assert param_name in self.param_names

        marginal = self._get_1d(param_name)

        if len(marginal)>rebin:

            marginalized = self.bar_plot(marginal,prange=self.param_ranges,rebin=rebin)
        else:
            marginalized = self.bar_plot(marginal,prange=self.param_ranges)

        return marginalized

    def confidence_interval(self,pname,percentile=[0.05,0.95]):

        marginal = self._get_1d(pname)

        if isinstance(percentile,list):
            interval = []
            for p in percentile:
                interval.append(confidence_interval(p,marginal))
            return interval
        else:
            return confidence_interval(percentile,marginal)

    def _tick_formatter(self,pname=''):

        if pname == 'fsub':
            return '%.3f'
        elif pname=='logmhm':
            return '%.1f'
        elif pname=='SIE_gamma':
            return '%.3f'
        elif pname=='SIE_shear':
            return '%.3f'
        else:
            return '%.2f'

