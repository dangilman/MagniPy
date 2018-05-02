from MagniPy.util import *
from MagniPy.lensdata import Data
import matplotlib.pyplot as plt
from matplotlib.colors import cnames

default_colors = ['k',cnames['indianred'],cnames['royalblue'],
                  cnames['mediumpurple'],cnames['darkmagenta'],cnames['forestgreen']]

class FluxRatioCumulative:

    def __init__(self,fnames=None,reference_fluxes=None,read_fluxes=True):

        lensdata = []

        for fname in fnames:

            anomalies = np.loadtxt(fname)

            lensdata.append(np.sqrt(np.sum(anomalies**2,axis=1)))

        self.lensdata = lensdata

    def set_reference_data(self,refdata):

        self.reference_data = refdata

    def make_figure(self,nbins=100,xmax=0.5,color=None,xlabel=None,ylabel='',labels=None,linewidth=5,linestyle='-',alpha=0.8,
                    xlims=None,ylims=None):

        if color is None:
            color = default_colors
        if xlabel is None:
            xlabel = r'$\sqrt{\sum_{i=1}^{3} \left( \frac{F_{\rm{SIE(i)}} - F_{\rm{data(i)}}}{F_{\rm{data(i)}}} \right)^2}$'
        if ylabel is None:
            ylabel = 'Percent\n'+r'$ > x$'

        fig = plt.figure(1)
        ax = plt.subplot(111)
        fig.set_size_inches(6,6)

        for i,anomalies in enumerate(self.lensdata):

            L = len(anomalies)

            y = []
            x = np.linspace(0,xmax,nbins)

            for k in range(0, len(x)):
                y.append((L - sum(val < x[k] for val in anomalies))*L**-1)

            if labels is not None:
                plt.plot(x, y, c=color[i], label=labels[i], linewidth=linewidth,
                         linestyle=linestyle, alpha=alpha)
                leg = True
            else:
                plt.plot(x, y, c=color[i], linewidth=linewidth,
                         linestyle=linestyle, alpha=alpha)
                leg = False

        ax.set_xlabel(xlabel,fontsize=20)
        ax.set_ylabel(ylabel,rotation=90,fontsize=18)
        if xlims is None:
            ax.set_xlim(0,xmax)
        else:
            ax.set_xlim(xlims[0],xlims[1])

        if ylims is None:
            ax.set_ylim(0,1)

        else:
            ax.set_ylim(ylims[0],ylims[1])

        if leg:
            plt.legend()

        return fig,ax

if False:
    import os
    from MagniPy.paths import *
    path = os.getenv('HOME')+'/Code/jupyter_notebooks/Compute_flux_ratios/'
    fnames = [path+'lenstronomy_LOS.txt',path+'lensmodel_LOS.txt']
    fr = FluxRatioCumulative(fnames=fnames,fname_ref=path+'reference.txt')

    fig,ax = fr.make_figure(xmax=2)
    plt.show(fig)




