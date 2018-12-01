import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from MagniPy.util import confidence_interval
from MagniPy.Analysis.Visualization.barplot import bar_plot

class Joint2D(object):

    plt.rcParams['axes.linewidth'] = 2.5

    plt.rcParams['xtick.major.width'] = 2.5
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.minor.size'] = 2

    plt.rcParams['ytick.major.width'] = 2.5
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.minor.size'] = 2

    cmap = 'bone'
    default_contour_colors = [(colors.cnames['lightgreen'], colors.cnames['green'], 'k'),
                              (colors.cnames['orchid'], colors.cnames['darkviolet'], 'k'),
                              (colors.cnames['grey'], colors.cnames['black'], 'k'),
                              (colors.cnames['skyblue'], colors.cnames['blue'], 'k'),
                              (colors.cnames['coral'], 'r', 'k')]
    truth_color = 'r'
    tick_font = 12

    def __init__(self,simulations=[],ax=None,fig=None, cmap=None):

        self.simulation_densities = simulations

        if fig is None:
            fig = plt.figure(1)
        if ax is None:
            ax = plt.subplot(111)

        self.fig = fig
        self.ax = ax
        if cmap is not None:
            self.cmap = cmap

    def _compute_single(self, densities):

        final = np.ones_like(densities[0])

        for density in densities:

            #final_density = np.ones_like(densities)

            final *= density

        return self._norm_density(final)

    def _compute_densities(self, param_names, param_ranges):

        final_densities, coordinates = [], []

        aspect = (param_ranges[param_names[0]][1] - param_ranges[param_names[0]][0]) * \
                 (param_ranges[param_names[1]][1] - param_ranges[param_names[1]][0]) ** -1

        extent = [param_ranges[param_names[0]][0], param_ranges[param_names[0]][1], param_ranges[param_names[1]][0],
                  param_ranges[param_names[1]][1]]

        posterior_density = []

        for color_index, sim in enumerate(self.simulation_densities):

            sim_density = self._compute_single(sim)
            posterior_density.append(sim_density)

            x, y = np.linspace(extent[0], extent[1], sim_density.shape[1]), \
                   np.linspace(extent[2], extent[3], sim_density.shape[0])

            final_densities.append(sim_density)
            coordinates.append([x, y])

        return final_densities, coordinates, aspect, extent

    def make_plot(self, param_names = None, param_ranges=None,filled_contours=False, contour_colors=None, contour_alpha=0.6,
                  tick_label_font=12, levels=[0.05,0.22,1], truths = None, xlabel_on = True,
                  ylabel_on = True, label_size=18):

        if contour_colors is None:
            contour_colors = self.default_contour_colors

        final_densities, coordinates, aspect, extent = self._compute_densities(param_names, param_ranges)

        for color_index, sim_density in enumerate(final_densities):

            x, y = coordinates[color_index][0], coordinates[color_index][1]

            if filled_contours:

                self.contours(x,y,sim_density,contour_colors=contour_colors[color_index],
                              contour_alpha=contour_alpha, extent=extent,aspect=aspect,levels=levels)

                self.ax.imshow(sim_density, extent=extent,
                               aspect=aspect, origin='lower', cmap=self.cmap, alpha=0)
                self.ax.set_xticklabels([])
                self.ax.set_yticklabels([])

            else:

                self.ax.imshow(sim_density, extent=extent,
                               aspect=aspect, origin='lower', cmap=self.cmap, alpha=1,vmin=0,vmax=np.max(sim_density))
                #self.contours(x, y, sim_density, contour_colors=contour_colors[color_index],
                #              contour_alpha=contour_alpha, extent=extent, aspect=aspect, levels=levels)

        if truths is not None:

            truth1,truth2 = truths[param_names[0]],truths[param_names[1]]

            self.ax.axvline(truth1,color=self.truth_color,linestyle='--',linewidth=3)
            #if inside2:
            self.ax.axhline(truth2,color=self.truth_color,linestyle='--',linewidth=3)
            #if inside1 and inside2:
            self.ax.scatter(truth1,truth2,color=self.truth_color,s=50)

        if xlabel_on:

            nticks = 5

            xticks = np.linspace(param_ranges[param_names[0]][0], param_ranges[param_names[0]][1], nticks)

            xlabel_name, xticks, xtick_labels = self._convert_param_names(param_names[0], xticks)

            self.ax.set_xlabel(xlabel_name, fontsize=label_size)
            self.ax.set_xticks(xticks)
            if param_names[0] == 'source_size_kpc':
                self.ax.set_xticklabels(np.array(xtick_labels).astype(int), fontsize=tick_label_font)
            elif param_names[0] == 'a0_area':
                self.ax.set_xticklabels(np.round(np.array(xtick_labels),1), fontsize=tick_label_font)
            else:
                if param_names[0] == 'SIE_gamma':
                    rotation = 45
                else:
                    rotation = 0
                self.ax.set_xticklabels(np.array(xtick_labels), fontsize=tick_label_font, rotation=rotation)
                self.ax.set_ylabel(xlabel_name, fontsize=label_size)
                self.ax.xaxis.set_major_formatter(FormatStrFormatter(self._tick_formatter(pname=xlabel_name)))

            self.ax.set_xlim(xticks[0], xticks[-1])

        if ylabel_on:

            nticks = 5

            yticks = np.linspace(param_ranges[param_names[1]][0],param_ranges[param_names[1]][1],nticks)
                #ytick_labels = [str(tick) for tick in yticks]

            ylabel_name, yticks, ytick_labels = self._convert_param_names(param_names[1], yticks)
            self.ax.set_yticks(yticks)

            if param_names[1] == 'source_size_kpc':
                self.ax.set_yticklabels(np.array(ytick_labels).astype(int), fontsize=tick_label_font)
            elif param_names[1] == 'a0_area':
                self.ax.set_yticklabels(np.round(np.array(ytick_labels),1), fontsize=tick_label_font)
            else:

                self.ax.set_yticklabels(np.array(ytick_labels), fontsize=tick_label_font)
                self.ax.set_ylabel(ylabel_name, fontsize=label_size)
                self.ax.yaxis.set_major_formatter(FormatStrFormatter(self._tick_formatter(pname=ylabel_name)))
            self.ax.set_ylabel(ylabel_name, fontsize=label_size)

            self.ax.set_ylim(yticks[0], yticks[-1])

        self.param_names = param_names
        self.param_ranges = param_ranges

        self.sim_density = sim_density

        return self.ax, (coordinates, final_densities, aspect, extent)

    def _convert_param_names(self, pname, ticks):

        if pname == 'fsub':
            #pname = r'$\sigma_{\rm{sub}} \ \left[kpc^{-2}\right]$'
            pname = r'$f_{\rm{sub}}$'
            # convert between fsub and sigma sub [kpc ^ -2] with m_0 = 10^6
            #tick_labels = ticks*(0.8 / 0.01)
            tick_labels = ticks

        elif pname == 'log_m_break' or pname == 'logmhm':
            pname = r'$\log_{10} \left(m_{\rm{hm}}\right)$'
            tick_labels = ticks

        elif pname == 'LOS_normalization':
            pname = r'$\delta_{\rm{LOS}}$'
            tick_labels = ticks

        elif pname == 'source_size_kpc':
            #pname = r'$\sigma_{\rm{source}}$'
            pname = r'$\rm{source} \ \rm{size} \ \left[\rm{pc}\right]}$'
            tick_labels = np.array(ticks)*1000

        elif pname == 'a0_area':

            pname = r'$\sigma_{\rm{sub}}\times 10^{-2} \ \left[kpc^{-2}\right]$'
            tick_labels = ticks*100

        elif pname == 'SIE_gamma':
            pname = r'$\gamma_{\rm{macro}}$'
            tick_labels = ticks

        else:
            tick_labels = ticks

        return pname, ticks, tick_labels

    def _norm_density(self,density):

        return density*(np.sum(density)*density.shape[0]**2)**-1

    def contours(self, x,y,grid, linewidths=4, filled_contours=True,contour_colors='',
                 contour_alpha=1,extent=None,aspect=None, levels=[0.05,0.22,1]):

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

    def _get_1d(self,pname,density,pnames):

        if pnames[0] == pname:
            axis = 1
        else:
            axis = 0

        return np.sum(density,axis=axis), axis

    def marginalize(self, param_name, pnames, density, pranges, rebin=25):

        marginal, idx = self._get_1d(param_name, density, pnames)

        if len(marginal)>rebin:

            marginalized, bar_cen, bar_h = bar_plot(marginal,pranges[param_name],rebin=rebin)

        else:
            marginalized, bar_cen, bar_h = bar_plot(marginal,pranges[param_name])
        #print(self._quick_confidence(bar_cen, bar_h))

        return marginalized

    def _quick_confidence(self, centers, heights, percentile = 0.95):

        total = np.sum(heights)
        summ, index = 0, 0
        while summ < total * percentile:
            summ += heights[index]
            index += 1
        return centers[index-1]

    def confidence_interval(self,pname,density,pnames,percentile=[0.05,0.95]):

        marginal, idx = self._get_1d(pname,density,pnames)

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
        elif pname == r'$\sigma_{\rm{sub}}\times 10^{-2} \ \left[kpc^{-2}\right]$':
            return '%.1f'
        elif pname=='logmhm' or pname=='log_m_break' or pname == r'$\log_{10} \left(m_{\rm{hm}}\right)$':
            return '%.1f'
        elif pname == 'c_power':
            return '%.3f'
        elif pname=='SIE_gamma' or pname == r'$\gamma_{\rm{macro}}$':
            return '%.2f'
        elif pname == 'source_size_kpc' or pname ==  r'$\rm{source} \ \rm{size} \ \left[\rm{pc}\right]}$':
            return '%d'
        elif pname == r'$\rm{source} \ \rm{size} \ \left[\rm{pc}\right]}$':
            return '%.2f'
        elif pname == r'$\delta_{\rm{LOS}}$' or pname == 'LOS_normalization':
            return '%.1f'
        elif pname == r'$\sigma_{\rm{sub}}' or pname == 'sigma_sub':
            return '%.1f'
        else:
            return '%.2f'

