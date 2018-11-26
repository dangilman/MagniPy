from pyHalo.pyhalo import pyHalo
from pyHalo.single_realization import *
from MagniPy.Solver.solveroutines import *
from MagniPy.Solver.analysis import Analysis
from MagniPy.LensBuild.main_deflector import Deflector
from MagniPy.MassModels.SIE import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def make_kappa_maps(realizations, zlens=0.5, zsource=1.5, Npix = 200, s = 1.3):

    analysis = Analysis(zlens, zsource)

    x, y = np.linspace(-s, s, Npix), np.linspace(-s, s, Npix)
    deltaPix = 2*s * Npix ** -1
    xx, yy = np.meshgrid(x, y)

    macroargs_start = {'R_ein': 1, 'x': 0, 'y': 0, 'ellip': 0.22, 'ellip_theta': 23, 'shear': 0.065, 'shear_theta': -30,
                       'gamma': 2.0}
    macromodel_start = Deflector(redshift=zlens, subclass=SIE(),
                                 varyflags=['1', '1', '1', '1', '1', '1', '1', '0', '0', '0'], lens_args=None,
                                 **macroargs_start)

    convergence_main_plane = []
    convergence_halos_born = []
    convergence_full_nonlinear = []
    convergence_macromodel = []
    convergence_halos_nonlinear = []

    for real in realizations:

        realization_main = realization_at_z(real, zlens)

        lens_system_main = analysis.build_system(main=macromodel_start, multiplane=False)
        lens_system_full = analysis.build_system(main=macromodel_start, realization=real, multiplane=True)
        lens_system_halos = analysis.build_system(realization=real, multiplane=False)
        lens_system_halos_main = analysis.build_system(realization=realization_main, multiplane=False)

        convergence_main = analysis.get_kappa(lens_system=lens_system_main, x=xx, y=yy, multiplane=False)
        convergence_full = analysis.get_kappa(lens_system=lens_system_full, x=xx, y=yy, multiplane=True)
        convergence_halos = analysis.get_kappa(lens_system=lens_system_halos, x=xx, y=yy, multiplane=False)
        convergencemain_halos = analysis.get_kappa(lens_system=lens_system_halos_main, x=xx, y=yy, multiplane=False)

        convergence_macromodel.append(convergence_main)
        convergence_full_nonlinear.append(convergence_full)
        convergence_halos_born.append(convergence_halos)
        convergence_main_plane.append(convergencemain_halos)
        convergence_halos_nonlinear.append(convergence_full - convergence_main)

    return convergence_macromodel, convergence_main_plane, convergence_halos_born, convergence_halos_nonlinear, (xx, yy, deltaPix)

def kappa_plot(conv_map, conv_map_subtract = None, ax = None,
               color_range = 0.1, mask = 0.25, cmap = 'bwr',annotate_text = None,
              bar_label = '', annotate_xy = (-0.075, 0.034), annotate_xy_2 = (0.3, 1.05),
               annotate_kappa_type = '', mean0 = True, savename=None, show_colorbar=False, fsize=8,
               fsize_1 = 10):
    """

    :param ax:
    :return:
    """

    plt.rcParams['axes.linewidth'] = 2.5

    plt.rcParams['xtick.major.width'] = 2.5
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['xtick.minor.size'] = 2

    plt.rcParams['ytick.major.width'] = 2.5
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['ytick.minor.size'] = 2

    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13

    if conv_map_subtract is None:
        conv_map_subtract = np.zeros_like(conv_map)

    vmin, vmax = -color_range, color_range
    L = np.shape(conv_map)[0]
    xi = yi = np.linspace(-L * 0.5, L * 0.5, L)
    xxi, yyi = np.meshgrid(xi, yi)
    r = np.sqrt(xxi ** 2 + yyi ** 2)
    rmax = L * 0.5
    conv = conv_map - conv_map_subtract
    if mean0:
        conv += -np.mean(conv)
    radial = np.where(r > rmax)
    mask = np.where(r < mask * 0.4 * L)
    conv[radial] = np.nan
    conv[mask] = 0
    theta = np.linspace(0, 2 * np.pi, 100)
    rmax_line = rmax * 0.99
    linex, liney = rmax_line * np.cos(theta), rmax_line * np.sin(theta)

    im = ax.imshow(conv, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap, extent=[-rmax, rmax, -rmax, rmax])
    ax.plot(linex, liney, color='k', linewidth=3)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.5%", pad=0.1)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(bar_label, size=19, fontsize=20)
        #cax.yaxis.set_ticks_position('left')


    if annotate_text is not None:
        ax.annotate(annotate_text, xy=annotate_xy, xycoords='axes fraction', fontsize=fsize_1)
    if annotate_kappa_type is not None:
        ax.annotate(annotate_kappa_type, xy=annotate_xy_2, xycoords='axes fraction', fontsize=fsize)

    if savename is not None:
        plt.savefig(savename)

    return ax