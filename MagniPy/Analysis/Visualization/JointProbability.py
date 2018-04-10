import numpy as np
import matplotlib.pyplot as plt
from MagniPy.Analysis.KDE.kde import *
from scipy.signal import resample

class ProbabilityDensity:

    plt.rcParams['axes.linewidth'] = 2

    plt.rcParams['xtick.major.width'] = 3
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.minor.size'] = 2

    plt.rcParams['ytick.major.width'] = 3
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.minor.size'] = 2

    xlabel_default = r'$A_0 \ M_{\rm{odot}}^{-1}$'
    ylabel_default = r'$m_{\rm{hm}}$'

    def __init__(self,p1_range=None,p2_range=None,kde_class ='getdist',ax=None,fig=None,
                 boundary_correction_order=1,dimension=2):

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

        self.p1_range = p1_range
        self.p2_range = p2_range

        if dimension==2:
            if kde_class=='scipy':
                self.kde = KDE_scipy(p1_range=p1_range,p2_range=p2_range)
            else:
                self.kde = KDE(p1_range=p1_range,p2_range=p2_range,boundary_correction_order=boundary_correction_order,dim=2)
        else:

            if kde_class=='scipy':
                self.kde = KDE_scipy(p1_range=p1_range,p2_range=p2_range)
            else:
                self.kde = KDE(p1_range=p1_range,p2_range=p2_range,boundary_correction_order=boundary_correction_order,dim=1)

    def _get1ddensity(self,data,bins=20,kde=True):

        if kde:
            counts = self.kde.density(data)
            _, bin_edges = np.histogram(data, bins=len(counts), range=[self.p1_range[0], self.p1_range[1]], normed=True)

        else:
            counts,bin_edges = np.histogram(data,bins=bins,range=[self.p1_range[0],self.p1_range[1]],normed=True)

        return counts,bin_edges[0:-1]

    def _get2ddensity(self,data_p1,data_p2,bins=20,kde=True):

        if kde:
            data_array = np.vstack([np.array(data_p1), np.array(data_p2)])
            counts = self.kde.density(data_array.T).T

        else:

            counts, b, patches = np.histogram2d(data_p1, data_p2, bins=bins, range=[self.p1_range, self.p2_range],
                                                normed=True)

        self.density = counts.T

        return counts.T

    def MakeSinglePlot(self,datasets,bins=20,kde=True,xlabel=None,ylabel=None,xtick_labels=None,ytick_labels=None,tick_font=12):

        r1 = self.p1_range[1] - self.p1_range[0]

        for index in range(0,len(datasets)):

            binned_counts, xvalues = self._get1ddensity(datasets[index], bins=bins, kde=kde)

            try:
                heights *= binned_counts
            except:
                heights = binned_counts




    def MakeJointPlot(self,datasets1=[],datasets2=[],bins=20,kde=True,xlabel=None,ylabel=None,xtick_labels=None,ytick_labels=None,tick_font = 12):

        r1,r2 = self.p1_range[1]-self.p1_range[0],self.p2_range[1]-self.p2_range[0]

        for index in range(0,len(datasets1)):
            try:
                binned_counts *= self._get2ddensity(datasets1[index],datasets2[index],bins=bins,kde=kde)
            except:
                binned_counts = self._get2ddensity(datasets1[index],datasets2[index],bins=bins,kde=kde)

        # normalize
        binned_counts *= len(binned_counts)**2*(np.sum(binned_counts))**-1

        image = self.ax.imshow(binned_counts,extent=[self.p1_range[0],self.p1_range[1],self.p2_range[0],self.p2_range[1]],
                              aspect=(r1*r2**-1),origin='lower')

        if xlabel is None:
            xlabel = self.xlabel_default
        if ylabel is None:
            ylabel = self.ylabel_default

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        return image,self.ax

    def MarginalDensity(self,param_index,data_p1=None,data_p2=None):

        if self.density is None:
            self._getdensity(data_p1,data_p2)

        marg = np.sum(self.density,axis=param_index)

        return marg

    def bar_plot(self, bar_heights, prange, color='k', alpha='1', rebin=None):

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
            self.ax.plot([x1, x2], [y, y], color=color)
            self.ax.fill_between([x1, x2], y, color=color, alpha=alpha)
            self.ax.plot([x1, x1], [0, y], color=color)
            self.ax.plot([x2, x2], [0, y], color=color)

        self.ax.set_xlim(prange[0], prange[1])
        self.ax.set_ylim(0, max(bar_heights) * 1.05)
        return self.ax


