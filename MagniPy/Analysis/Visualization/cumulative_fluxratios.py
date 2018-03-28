from MagniPy.util import *
from MagniPy.lensdata import Data
import matplotlib.pyplot as plt
from matplotlib.colors import cnames

default_colors = ['k',cnames['indianred'],cnames['royalblue'],
                  cnames['mediumpurple'],cnames['darkmagenta'],cnames['forestgreen']]

class FluxRatioCumulative:

    def __init__(self,datasets='load',fnames=None,refdataset='load',fname_ref=None):

        lensdata = []

        if datasets=='load':

            assert isinstance(fnames,list)

            for fname in fnames:
                lensdata.append(read_data(fname))

        else:
            assert isinstance(datasets,list)
            lensdata = datasets

        if refdataset=='load':

            refdata= read_data(fname_ref)[0]

        else:
            refdata = refdataset

        self.lensdata = lensdata

        self.set_reference_data(refdata)

    def set_reference_data(self,refdata):

        self.reference_data = []

        if isinstance(refdata,list):

            self.reference_data = refdata

        else:
            self.reference_data = [refdata]*len(self.lensdata)

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

        count = 0

        for i,model_data in enumerate(self.lensdata):

            flux_ratio_residuals = []

            for j,dset in enumerate(model_data):

                flux_ratio_residuals.append(dset.flux_anomaly(other_data=self.reference_data[count], index=1, sum_in_quad=True))
            count+=1

            #values, bins = np.histogram(flux_ratio_residuals, bins=nbins, range=(-0.00001, xmax))
            L = len(model_data)

            y = []
            x = np.linspace(0,xmax,nbins)

            for k in range(0, len(x)):
                y.append((len(flux_ratio_residuals) - sum(val < x[k] for val in flux_ratio_residuals))*L**-1)

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




