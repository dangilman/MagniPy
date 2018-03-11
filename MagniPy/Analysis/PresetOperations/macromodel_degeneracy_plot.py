import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import cnames,LinearSegmentedColormap
import matplotlib
from MagniPy.util import confidence_interval

def flux_vs_position_curves(fluxfolder, positionfolder, perturbation_amplitude, position_tol, color='k', xlabel='',
                            ylabel=r'$\sqrt{\sum \delta {F_i}^2}$',percentile=68):
    matplotlib.rcParams.update({'font.size': 15})
    plt.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams['xtick.major.size'] = 7
    matplotlib.rcParams['xtick.major.width'] = 2
    matplotlib.rcParams['xtick.minor.size'] = 7
    matplotlib.rcParams['xtick.minor.width'] = 2
    sigma_pos = 0.003

    fluxanomalies = np.loadtxt(fluxfolder)

    posanomalies = np.loadtxt(positionfolder)*(sigma_pos)**-1

    flux_means = []
    flux_uppersigma = []
    flux_lowersigma = []

    for i, pert in enumerate(perturbation_amplitude):
        fluxes = fluxanomalies[:, i]
        positions = posanomalies[:, i]

        inds_to_keep = np.where(np.logical_and(positions <= position_tol,np.isfinite(fluxes)))

        fluxes = fluxes[inds_to_keep]

        flux_means.append(np.mean(fluxes))

        flux_uppersigma.append(confidence_interval(percentile*100**-1, fluxes))
        flux_lowersigma.append(confidence_interval(1-percentile*100**-1, fluxes))

    fig = plt.figure(1)
    fig.set_size_inches(6, 6)
    ax = plt.subplot(111)

    ax.fill_between(perturbation_amplitude, np.array(flux_lowersigma), np.array(flux_uppersigma), alpha=1,
                    color=color)


    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig,ax

def flux_vs_position_distributions(fluxfolder, positionfolder, perturbation_amplitude, position_tol, colors=None,
                            xlabel=r'$\sqrt{\sum_{i=1}^{3} \delta {F_i}^2}$',nbins=20,cmap='gist_heat',xmax=0.25,alpha=0.8,
                                   cmap_range=0.8,xlimit=None,use_cols = None):

    if xlimit is None:
        xlimit = xmax

    matplotlib.rcParams.update({'font.size': 15})
    plt.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams['xtick.major.size'] = 7
    matplotlib.rcParams['xtick.major.width'] = 2
    matplotlib.rcParams['xtick.minor.size'] = 7
    matplotlib.rcParams['xtick.minor.width'] = 2

    sigma_pos = 0.003
    fluxanomalies = np.loadtxt(fluxfolder)

    if isinstance(cmap,str):
        cm = plt.get_cmap(cmap)
    else:
        cm = False
    # to arcseconds
    posanomalies = np.loadtxt(positionfolder)*(sigma_pos)**-1

    fig = plt.figure(1)
    fig.set_size_inches(6, 6)

    ax = plt.subplot(1, 1, 1)

    color_index = []
    count = 0
    for i, pert in enumerate(perturbation_amplitude):

        if use_cols is not None and i not in use_cols:
            continue

        fluxes = fluxanomalies[:, i]
        positions = posanomalies[:, i]

        inds_to_keep = np.where(np.logical_and(positions <= position_tol,np.isfinite(fluxes)))

        fluxes = fluxes[inds_to_keep]

        if cm:
            plt.hist(fluxes,bins=nbins,range=(0,xmax),color=cm(0.85*pert*max(perturbation_amplitude)**-1),
                     label=r'$\Delta \gamma=$'+' '+str(np.round(pert,3)),normed=True,alpha=alpha)
            plt.hist(fluxes, bins=nbins, range=(0, xmax), color=cm(cmap_range * pert * max(perturbation_amplitude) ** -1),
                     histtype='step',normed=True, alpha=1,linewidth=3)
        else:
            plt.hist(fluxes, bins=nbins, range=(0, xmax), color=cmap[count],
                     label=r'$\Delta \gamma=$' + ' ' + str(np.round(pert, 3)), normed=True, alpha=alpha)
            plt.hist(fluxes, bins=nbins, range=(0, xmax),
                     color=cmap[count],histtype='step', normed=True, alpha=1, linewidth=4)

        count+=1

        color_index.append(cmap_range * pert * max(perturbation_amplitude) ** -1)

    leg = plt.legend()
    count = 0
    for i,lh in enumerate(leg.legendHandles):
        lh.set_alpha(1)
        if cm:
            lh.set_color(cm(color_index[i]))
        else:
            lh.set_color(cmap[count])
        count+=1
    ax.set_xlabel(xlabel)
    ax.set_ylabel('probability')

    ax.set_xlim(0,xlimit)

    ax.set_yticklabels([])

    return fig,ax
