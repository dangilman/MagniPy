import matplotlib.pyplot as plt
from MagniPy.LensBuild.Cosmology.cosmology import Cosmo
from MagniPy.LensBuild.spatial_distribution import NFW_3D
import numpy as np

def show_convergence_effects():

    cores = [20, 60, 100]

    xbins, kappa = [], []

    for r_core in cores:

        s = NFW_3D(rmin=0.01, rmax3d=200, rmax2d=200, Rs=60, xoffset=0, yoffset=0,
                   tidal_core=True, r_core=r_core, cosmology=Cosmo(0.5, 1.5))

        x, y, r2d, r3d, z = s.draw(30000)

        annuli = np.arange(5, 202, 5)
        dr = annuli[1] - annuli[0]
        rvals = []
        for i in range(0, len(annuli)):
            rvals.append(float(annuli[i]) ** -1 * np.sum(np.absolute(r2d - annuli[i]) < dr))

        kappa.append(rvals)
    cols = ['k', 'r', 'b']
    for i, group in enumerate(kappa):
        plt.plot(annuli, group, color=cols[i])
        plt.xlim(annuli[0], 25)
    plt.show()

def convergence_kde():
    from scipy.stats.kde import gaussian_kde

    Rs = 60
    r_core = 1.15*Rs

    s = NFW_3D(rmin=0.001, rmax3d=200, rmax2d=200, Rs=Rs, xoffset=0, yoffset=0,
               tidal_core=True, r_core=r_core, cosmology=Cosmo(0.5, 1.5))

    x, y, r2d, r3d, z = s.draw(25000)

    xy = np.vstack([x,y])
    kernel = gaussian_kde(xy)

    x,y = np.linspace(-200,200,50),np.linspace(-200,200,50)

    xx,yy = np.meshgrid(x,y)
    plt.subplot(111)

    density = kernel(np.vstack([xx.ravel(),yy.ravel()])).reshape(len(x),len(x))
    density *= np.max(density)**-1
    plt.imshow(density,extent=[-200,200,-200,200],cmap='Accent',vmin=0,vmax=1)
    plt.colorbar()
    plt.xlim(-200,200)
    plt.ylim(-200,200)
    theta = np.linspace(0,2*np.pi*200)
    x_core,y_core = r_core*np.cos(theta),r_core*np.sin(theta)
    x_rs,y_rs = Rs*np.cos(theta),Rs*np.sin(theta)
    x_rein,y_rein = 7*np.cos(theta),7*np.sin(theta)
    plt.scatter(x_core,y_core,color='k',marker='^',s=4,label='tidal radius')
    plt.scatter(x_rs,y_rs,color='k',label='NFW Rs',s=5)
    plt.scatter(x_rein,y_rein,color='k',marker='d',s=3.5,label='Einstein radius')
    plt.legend(fontsize=12)

    plt.show()

convergence_kde()








