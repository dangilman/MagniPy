import numpy as np
import matplotlib.pyplot as plt
from MagniPy.Analysis.KDE.kde import *
from scipy.signal import resample
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter

class ProbabilityDensity:

    plt.rcParams['axes.linewidth'] = 2.5

    plt.rcParams['xtick.major.width'] = 2.5
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.minor.size'] = 2

    plt.rcParams['ytick.major.width'] = 2.5
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.minor.size'] = 2

    xlabel_default = r'$A_0 \ M_{\rm{odot}}^{-1}$'
    ylabel_default = r'$m_{\rm{hm}}$'

    cmap = 'gist_heat_r'

    default_contour_colors = (colors.cnames['orchid'],colors.cnames['darkviolet'],'k')

    def __init__(self,param_names=[],posterior_samples=None,pranges=None,kde_class ='getdist',ax=None,fig=None,
                 boundary_correction_order=1):

        self.density = None

        if fig is None:
            fig = plt.figure(1)
        else:
            fig = fig
        self.fig = fig
        if ax is None:
            ax = plt.subplot(111)
        else:
            ax = ax

        self.ax = ax

        dimension = len(param_names)
        self.param_names = param_names

        if dimension==1:
            self.posteriorsamples = []
            p1_range = pranges[param_names[0]]
            self.p1_range = p1_range
            for post in posterior_samples:
                self.posteriorsamples.append(post[param_names[0]])

        elif dimension==2:

            p1_range = pranges[param_names[0]]
            p2_range = pranges[param_names[1]]
            self.p1_range, self.p2_range = p1_range, p2_range
            for i,post in enumerate(posterior_samples):
                if i==0:
                    self.posteriorsamples = [[post[param_names[0]]],[post[param_names[1]]]]
                else:
                    self.posteriorsamples[0] += [post[param_names[0]]]
                    self.posteriorsamples[1] += [post[param_names[1]]]

        else:
            raise Exception('dimension must be 1 or 2.')

        if dimension==2:
            if kde_class=='scipy':
                self.kde = KDE_scipy(p1_range=p1_range,p2_range=p2_range)
            else:
                self.kde = KDE(p1_range=p1_range,p2_range=p2_range,boundary_correction_order=boundary_correction_order,dim=2)
        else:

            if kde_class=='scipy':
                self.kde = KDE_scipy(p1_range=p1_range)
            else:
                self.kde = KDE(p1_range=p1_range,boundary_correction_order=boundary_correction_order,dim=1)

    def _get1ddensity(self,data,bins=20,kde=True):

        if kde:
            counts = self.kde.density(data)


        else:
            counts,bin_edges = np.histogram(data,bins=bins,range=[self.p1_range[0],self.p1_range[1]],normed=True)

        return counts

    def _get2ddensity(self,data_p1,data_p2,bins=20,kde=True):

        if kde:
            data_array = np.vstack([np.array(data_p1), np.array(data_p2)])
            counts = self.kde.density(data_array.T).T

        else:

            counts, b, patches = np.histogram2d(data_p1, data_p2, bins=bins, range=[self.p1_range, self.p2_range],
                                                normed=True)

        self.density = counts.T

        return counts.T

    def _get_binned_counts_2d(self, bins, kde):

        for index in range(0,len(self.posteriorsamples[0])):

            try:
                binned_counts *= self._get2ddensity(self.posteriorsamples[0][index],self.posteriorsamples[1][index],bins=bins,kde=kde)
            except:
                binned_counts = self._get2ddensity(self.posteriorsamples[0][index],self.posteriorsamples[1][index],bins=bins,kde=kde)

        self.binned_counts = binned_counts

        return binned_counts

    def _get_binned_counts_1d(self, bins, kde):

        for index in range(0,len(self.posteriorsamples)):

            try:
                binned_counts *= self._get1ddensity(self.posteriorsamples[index],bins=bins,kde=kde)
            except:
                binned_counts = self._get1ddensity(self.posteriorsamples[index],bins=bins,kde=kde)

        self.binned_counts = binned_counts

        return binned_counts

    def MakeConditional1D(self,bins=20,kde=True,xlabel=None,ylabel=None,xtick_labels=None,xticks=None,ytick_labels=None,tick_font=12,tick_label_font=12,):

        r1 = self.p1_range[1] - self.p1_range[0]

        binned_counts = self._get_binned_counts_1d(bins,kde)

        binned_counts *= len(binned_counts) * (np.sum(binned_counts)) ** -1

        if len(binned_counts)>25:

            marginalized = self.bar_plot(binned_counts,prange=self.p1_range,rebin=25)

        else:
            marginalized = self.bar_plot(binned_counts,prange=self.p1_range)

        if xlabel is None:
            xlabel = self.param_names[0]
        self.ax.set_xlabel(xlabel)

        if xtick_labels is None:
            self.ax.xaxis.set_major_formatter(FormatStrFormatter(self._tick_formatter(pname=self.param_names[0])))

        else:
            self.ax.set_xticks(xticks,fontsize=tick_font)
            self.ax.set_xticklabels(xtick_labels,fontsize=tick_label_font)

        return self.ax

    def MakeJointPlot(self,bins=20,kde=True,xlabel=None,ylabel=None,xtick_labels=None,xticks=None,ytick_labels=None,yticks=None,tick_font = 12,
                      filled_contours=True,contour_colors=None,contour_alpha=0.6,tick_label_font=12):

        if contour_colors is None:
            contour_colors = self.default_contour_colors

        r1,r2 = self.p1_range[1]-self.p1_range[0],self.p2_range[1]-self.p2_range[0]

        binned_counts = self._get_binned_counts_2d(bins, kde)

        # normalize
        binned_counts *= len(binned_counts)**2*(np.sum(binned_counts))**-1

        if filled_contours:
            aspect = r1*r2**-1
            extent = [self.p1_range[0],self.p1_range[1],self.p2_range[0],self.p2_range[1]]
            x,y = np.linspace(self.p1_range[0],self.p1_range[1],len(binned_counts)),np.linspace(self.p2_range[0],self.p2_range[1],len(binned_counts))
            self.contours(x,y,binned_counts,contour_colors=contour_colors,contour_alpha=contour_alpha, extent=extent,aspect=aspect)
            self.ax.imshow(binned_counts, extent=[self.p1_range[0], self.p1_range[1], self.p2_range[0],
                                                  self.p2_range[1]],
                           aspect=(r1 * r2 ** -1), origin='lower', cmap=self.cmap, alpha=0)


        else:
            self.ax.imshow(binned_counts,extent=[self.p1_range[0], self.p1_range[1], self.p2_range[0],
                                                         self.p2_range[1]],
                                   aspect=(r1 * r2 ** -1), origin='lower', cmap=self.cmap, alpha=0)


        if xlabel is None:
            xlabel = self.param_names[0]
        if ylabel is None:
            ylabel = self.param_names[1]

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        if xtick_labels is None:
            self.ax.xaxis.set_major_formatter(FormatStrFormatter(self._tick_formatter(pname=self.param_names[0])))

        else:
            self.ax.set_xticks(xticks,fontsize=tick_font)
            self.ax.set_xticklabels(xtick_labels,fontsize=tick_label_font)

        if ytick_labels is None:
            self.ax.xaxis.set_major_formatter(FormatStrFormatter(self._tick_formatter(pname=self.param_names[1])))

        else:
            self.ax.set_yticks(yticks, fontsize=tick_font)
            self.ax.set_yticklabels(ytick_labels, fontsize=tick_label_font)

        return self.ax

    def _tick_formatter(self,pname=''):

        if pname == 'fsub':
            return '%.3f'
        elif pname=='mhm':
            return '%.1f'
        elif pname=='SIE_gamma':
            return '%.3f'
        elif pname=='SIE_shear':
            return '%.3f'
        else:

            return '%.3f'

    def MarginalDensity(self, param='',data_p1=None,data_p2=None,bins=20,kde=True):

        if hasattr(self,'binned_counts'):
            density = self.binned_counts
        else:
            density = self._get_binned_counts_2d(bins=bins, kde=kde)

        if param == self.param_names[0]:
            index = 0
            ran = self.p1_range
        elif param == self.param_names[1]:
            index = 1
            ran = self.p2_range

        marg = np.sum(density, axis=index)

        if len(marg)>25:

            marginalized = self.bar_plot(marg,prange=ran,rebin=25)
        else:
            marginalized = self.bar_plot(marg,prange=ran)

        return marginalized

    def bar_plot(self, bar_heights, prange, color='k', alpha='0.6', rebin=None, axis=None):

        if axis is None:
            axis = plt.subplot(111)

        if rebin is not None:
            new = []
            if len(bar_heights) % rebin == 0:
                fac = int(len(bar_heights) / rebin)
                for i in range(0, len(bar_heights), fac):
                    new.append(np.mean(bar_heights[i:(i + fac)]))

                bar_heights = new
            else:
                bar_heights = resample(bar_heights, rebin)

        bar_width = (prange[1] - prange[0]) * len(bar_heights) ** -1
        bar_centers = []
        for i in range(0, len(bar_heights)):
            bar_centers.append(prange[0] + bar_width * .5 + bar_width * i)
        for i, h in enumerate(bar_heights):
            x1, x2, y = bar_centers[i] - bar_width * .5, bar_centers[i] + bar_width * .5, h
            axis.plot([x1, x2], [y, y], color=color)
            axis.fill_between([x1, x2], y, color=colors.cnames['black'],alpha=alpha)
            axis.plot([x1, x1], [0, y], color=color)
            axis.plot([x2, x2], [0, y], color=color)

        axis.set_xlim(prange[0], prange[1])
        axis.set_ylim(0, max(bar_heights) * 1.05)
        return axis

    def contours(self, x,y,grid, levels = [.95,.68], linewidths=3.5, filled_contours=True,contour_colors='',
                 contour_alpha=1,extent=None,aspect=None):

        for i,lev in enumerate(levels):
            levels[i] = 1-lev

        X, Y = np.meshgrid(x, y)

        if filled_contours:
            levels.append(1)
            levels = np.array(levels)*np.max(grid)

            plt.contour(X, Y, grid, levels, extent=extent,
                              colors=contour_colors, linewidths=linewidths, zorder=1)
            plt.contourf(X, Y, grid, levels, colors=contour_colors, alpha=contour_alpha, zorder=1,
                         extent=extent, aspect=aspect)


        else:
            plt.contour(X, Y, grid, extent=extent, colors=contour_colors,
                                  levels=np.array(levels) * np.max(grid), linewidths=linewidths)



def bar_plot(bar_heights, prange, color='0', alpha='1', rebin=None, ax=None):

    if ax is None:
        ax = plt.subplot(111)

    if rebin is not None:
        new = []
        if len(bar_heights) % rebin == 0:
            fac = int(len(bar_heights) / rebin)
            for i in range(0, len(bar_heights), fac):
                new.append(np.mean(bar_heights[i:(i + fac)]))

            bar_heights = new
        else:
            bar_heights = resample(bar_heights, rebin)

    bar_width = (prange[1] - prange[0]) * len(bar_heights) ** -1
    bar_centers = []
    for i in range(0, len(bar_heights)):
        bar_centers.append(prange[0] + bar_width * .5 + bar_width * i)
    for i, h in enumerate(bar_heights):
        x1, x2, y = bar_centers[i] - bar_width * .5, bar_centers[i] + bar_width * .5, h
        ax.plot([x1, x2], [y, y], color=color)
        ax.fill_between([x1, x2], y, color=color, alpha=alpha)
        ax.plot([x1, x1], [0, y], color=color)
        ax.plot([x2, x2], [0, y], color=color)

    ax.set_xlim(prange[0], prange[1])
    ax.set_ylim(0, max(bar_heights) * 1.05)
    return ax


