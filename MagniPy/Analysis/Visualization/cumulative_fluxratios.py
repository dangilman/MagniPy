from MagniPy.util import *
from MagniPy.lensdata import Data
import matplotlib.pyplot as plt
from matplotlib.colors import cnames

default_colors = ['k',cnames['indianred'],cnames['royalblue'],
                  cnames['mediumpurple'],cnames['forestgreen']]

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

    def __init__(self,fnames=None,reference_fluxes=None,read_fluxes=True,cut=None):

        self.lensdata = []
        lensdata = []

        for fname in fnames:

            anomalies = np.loadtxt(fname)

            lensdata.append(anomalies)

        if cut is not None:

            for dset in lensdata:
                dset=dset[np.where(dset<=cut)]
                self.lensdata.append(dset)
        else:
            self.lensdata = lensdata

    def set_reference_data(self,refdata):

        self.reference_data = refdata

    def make_figure(self,nbins=100,xmax=0.5,color=None,xlabel=None,ylabel='',labels=None,linewidth=5,linestyle=None,alpha=0.8,
                    xlims=None,ylims=None,legend_args={}):

        if color is None:
            color = default_colors
        if xlabel is None:
            xlabel = r'$\sqrt{\sum_{i=1}^{3} \left( \frac{F_{\rm{SIE(i)}} - F_{\rm{data(i)}}}{F_{\rm{data(i)}}} \right)^2}$'
        if ylabel is None:
            ylabel = 'Percent\n'+r'$ > x$'
        if linestyle is None:
            linestyle = ['-']*len(self.lensdata)

        fig = plt.figure(1)
        fig.set_size_inches(7,7)
        ax = plt.subplot(111)

        for i,anomalies in enumerate(self.lensdata):

            L = len(anomalies)

            y = []
            x = np.linspace(0,xmax,nbins)

            for k in range(0, len(x)):
                y.append((L - sum(val < x[k] for val in anomalies))*L**-1)

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

        return fig,ax

if False:
    import os
    from MagniPy.paths import *
    path = os.getenv('HOME')+'/Code/jupyter_notebooks/Compute_flux_ratios/'
    fnames = [path+'lenstronomy_LOS.txt',path+'lensmodel_LOS.txt']
    fr = FluxRatioCumulative(fnames=fnames,fname_ref=path+'reference.txt')

    fig,ax = fr.make_figure(xmax=2)
    plt.show(fig)




