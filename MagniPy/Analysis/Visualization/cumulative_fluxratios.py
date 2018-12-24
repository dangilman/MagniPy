from MagniPy.util import *
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import cnames

default_colors = ['k',cnames['indianred'],cnames['royalblue'],
                  cnames['mediumpurple'],cnames['forestgreen']]
default_contour_colors = [(colors.cnames['grey'], colors.cnames['black'], 'k'),
                                (colors.cnames['skyblue'], colors.cnames['blue'], 'k'),
                              (colors.cnames['coral'], 'r', 'k'),
                              (colors.cnames['orchid'], colors.cnames['darkviolet'], 'k')]

plt.rcParams['axes.linewidth'] = 2.5

plt.rcParams['xtick.major.width'] = 2.5
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 2

plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.minor.size'] = 2

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

class FluxRatioCumulative:

    def __init__(self,fnames=None,reference_fluxes=None,read_fluxes=True,cut_high=None):

        self.lensdata = []
        self.cut_high = cut_high
        lensdata = []

        for fname in fnames:

            anomalies = np.loadtxt(fname)

            lensdata.append(anomalies)

        self.lensdata = lensdata

    def set_reference_data(self,refdata):

        self.reference_data = refdata

    def make_figure(self, plot_type, **kwargs):

        if plot_type == 'cumulative':

            out = self._make_figure_cumulative(**kwargs)

        elif plot_type == 'distribution':

            out = self._make_figure_hist(**kwargs)

        elif plot_type == 'correlation':

            out = self._make_figure_corr(**kwargs)

        return out

    def _make_figure_corr(self, nbins=50, xmax=0.5, color=None, xlabel=None, ylabel='', labels=None, linewidth=5,
                          linestyle=None, alpha=0.8, xlims=None, ylims=None, legend_args={}, shift_left=None,
                          plot_mode='hist'):

        if color is None:
            color = default_colors
        if linestyle is None:
            linestyle = ['-']*len(self.lensdata)

        fig = plt.figure(1)
        fig.set_size_inches(7, 10)
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)
        labels_done = False

        for i,ax in enumerate([ax1,ax2,ax3]):

            for j in range(int(len(self.lensdata))):
                if i == 0:
                    indx1 = 0
                    indx2 = 1
                elif i == 1:
                    indx1 = 0
                    indx2 = 2
                else:
                    indx1 = 1
                    indx2 = 2

                data1 = self.lensdata[j][:,indx1]
                data1 = data1[np.where(np.isfinite(data1))]

                data2 = self.lensdata[j][:,indx2]
                data2 = data2[np.where(np.isfinite(data2))]

                #if labels_done:
                #    ax.scatter(data1,data2,color = color[j], marker='o', s = linewidth, alpha=alpha)
                #else:
                #    ax.scatter(data1, data2, color=color[j], marker='o', s=linewidth, alpha=alpha, label=labels[j])
                grid, X, Y = np.histogram2d(data1, data2, bins=nbins,
                                            range=[[-xmax, xmax],[-xmax, xmax]], density=True)
                X, Y = X[0:-1], Y[0:-1]
                levels = [0.05, 1]

                levels = np.array(levels) * np.max(grid)
                #ax.contour(X, Y, grid, [], colors=color[j], linewidths=4, zorder=1)
                ax.contourf(X, Y, grid, levels, colors=color[j], alpha=alpha, zorder=1)

                if j == int(len(self.lensdata)) - 1:
                    labels_done = True

            if i == 0:
                indx1 = 0
                indx2 = 1
            elif i == 1:
                indx1 = 0
                indx2 = 2
            else:
                indx1 = 1
                indx2 = 2
            labx, laby = r'$\delta F$' + str(indx1), r'$\delta F$' + str(indx2)
            ax.set_xlabel(labx, fontsize=16)
            ax.set_ylabel(laby, fontsize=16)

            ax.set_xlim(-xmax,xmax)
            ax.set_ylim(-xmax,xmax)

        if labels is not None:
            leg = ax1.legend(**legend_args)


        return fig, [ax1,ax2,ax3]

    def _make_figure_hist(self, nbins=50, xmax=0.5, color=None, xlabel=None, ylabel='', labels=None, linewidth=5,
                          linestyle=None, alpha=0.8, xlims=None, ylims=None, legend_args={}, shift_left=None,
                          plot_mode='hist'):

        if color is None:
            color = default_colors
        if linestyle is None:
            linestyle = ['-']*len(self.lensdata)
        if xlabel is None:
            xlabel = r'$F_{\rm{data(i)}} - F_{\rm{fit(i)}}$'

        fig = plt.figure(1)
        fig.set_size_inches(7, 10)
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)
        labels_done = False

        for i,ax in enumerate([ax1,ax2,ax3]):

            for j in range(int(len(self.lensdata))):
                data = self.lensdata[j][:,i]
                data = data[np.where(np.isfinite(data))]

                if labels_done:
                    if plot_mode == 'lines':
                        h, b = np.histogram(data, bins=nbins, density=True)
                        ax.plot(b[0:-1],h, color=color[j], linewidth = linewidth, alpha=alpha, linestyle=linestyle[i])
                    else:
                        ax.hist(data, bins = nbins, color=color[j], histtype='step',linewidth=linewidth, density=True)
                        ax.hist(data, bins = nbins, color=color[j], histtype='bar', linewidth=1.5, alpha=alpha, density=True)

                else:
                    if plot_mode == 'lines':
                        h, b = np.histogram(data, bins=nbins, density=True)
                        ax.plot(b[0:-1], h, color=color[j], linewidth=linewidth, alpha=alpha, linestyle=linestyle[i],
                                label=labels[j])

                    else:
                        ax.hist(data, bins = nbins, color=color[j], histtype='step', linewidth=linewidth, density=True)
                        ax.hist(data, bins = nbins, color=color[j], histtype='bar', linewidth=1.5, alpha=alpha,label = labels[j],
                                density=True)
                    if j == int(len(self.lensdata)) - 1:
                        labels_done = True
                ax.set_yticklabels([])

            ax.set_xlim(-xmax,xmax)
            ax.set_yticks([])
            ax.set_xlabel(xlabel, fontsize=16)

        if labels is not None:
            leg = ax1.legend(**legend_args)
            for i,lh in enumerate(leg.get_patches()):
                lh.set_alpha(1)
                lh.set_facecolor(color[i])

        return fig, [ax1,ax2,ax3]

    def _make_figure_cumulative(self, nbins=100, xmax=0.5, color=None, xlabel=None, ylabel='', labels=None, linewidth=5, linestyle=None, alpha=0.8,
                                xlims=None, ylims=None, legend_args={}, shift_left=None):

        if color is None:
            color = default_colors
        if xlabel is None:
            xlabel = r'$\sqrt{\sum_{i=1}^{3} \left( \frac{F_{\rm{smooth(i)}} - F_{\rm{clumpy(i)}}}{F_{\rm{clumpy(i)}}} \right)^2}$'
        if ylabel is None:
            ylabel = 'Percent\n'+r'$ > x$'
        if linestyle is None:
            linestyle = ['-']*len(self.lensdata)

        lensdata_summed = []

        for dset in self.lensdata:

            lensdata_summed.append(np.sqrt(dset[:,0]**2 + dset[:,1]**2 + dset[:,2]**2))

        #lensdata_summed = self.lensdata
        fig = plt.figure(1)
        fig.set_size_inches(7,7)
        ax = plt.subplot(111)
        curves = []

        for i,anomalies in enumerate(lensdata_summed):

            anomalies = anomalies[np.where(np.isfinite(anomalies))]
            anomalies = anomalies[np.where(anomalies <= self.cut_high)]

            L = len(anomalies)

            y = []
            x = np.linspace(0,xmax,nbins)

            for k in range(0, len(x)):
                if shift_left is None:
                    shift = 0
                else:
                    shift = shift_left[i]
                y.append((L - sum(val < (x[k]+shift) for val in anomalies))*L**-1)

            y = np.array(y)
            curves.append(y)
            if labels is not None:
                plt.plot(x, y, c=color[i], label=labels[i], linewidth=linewidth,
                         linestyle=linestyle[i], alpha=alpha)
                leg = True

            else:

                plt.plot(x, y, c=color[i], linewidth=linewidth,
                         linestyle=linestyle[i], alpha=alpha)
                leg = False

        ax.set_xlabel(xlabel,fontsize=18)
        ax.set_ylabel(ylabel,rotation=90,fontsize=18)

        if xlims is None:
            ax.set_xlim(0,xmax)
        else:
            ax.set_xlim(xlims[0],xlims[1])

        if ylims is None:
            ylims = [0,1]

        ax.set_ylim(ylims[0],ylims[1])

        yticks = np.linspace(ylims[0],ylims[1],5)
        yticklabels = [str(int(tick*100))+'%' for tick in yticks]

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        if leg:
            plt.legend(**legend_args)

        return fig, ax, curves

if False:
    import os
    from MagniPy.paths import *
    path = os.getenv('HOME')+'/Code/jupyter_notebooks/Compute_flux_ratios/'
    fnames = [path+'lenstronomy_LOS.txt',path+'lensmodel_LOS.txt']
    fr = FluxRatioCumulative(fnames=fnames,fname_ref=path+'reference.txt')

    fig,ax = fr.make_figure(xmax=2)
    plt.show(fig)




