import h5py
import os
import numpy as np
from MagniPy.LensBuild.Cosmology.cosmology import Cosmo
from time import time
import pyfits
from MagniPy.util import coordinates_inbox
#from mesh import *
from collections import namedtuple

class Particle:

    def __init__(self,particle_mass=None,conversion=None,rotation=0):

        self.x = None
        self.y = None

        self.conversion = conversion
        self._particle_mass = particle_mass

    def load(self, hdf5_object):

        x = hdf5_object['Coordinates'][:, 0]
        y = hdf5_object['Coordinates'][:, 1]
        z = hdf5_object['Coordinates'][:, 2]

        self.x = np.array(x)*self.conversion
        self.y = np.array(y)*self.conversion
        self.z = np.array(z)*self.conversion

        if isinstance(self._particle_mass, float) or isinstance(self._particle_mass, int):
            self.masses = np.ones_like(self.x) * self._particle_mass
        else:
            assert len(self.x) == len(self._particle_mass)
            self.masses = self._particle_mass


class ConvergenceMap:

    def __init__(self,particle_mass,particle_x,particle_y,center_x,center_y,max_radius_kpc=1000):


        x = particle_x - center_y
        y = particle_y - center_x

        inds = coordinates_inbox(box_dx=2*max_radius_kpc,box_dy = 2*max_radius_kpc,centered_x= x,centered_y=y)

        self.x = x[inds]
        self.y = y[inds]

        self.mass = particle_mass[inds]
        self.max_radius_kpc = 0.5*max_radius_kpc


    def _find_adjacent(self,coordinate,grid,dx):

        grid = np.array(grid).T

        dr = np.sqrt((coordinate[0]-grid[:,0])**2 + (coordinate[1]-grid[:,1])**2)

        inds = np.argpartition(dr,4)[0:9]

        return inds

    def _weights(self,xvert,yvert,xp,yp,dx,dy):

        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

        ra = Rectangle(xvert-dx,yvert-dy,xvert+dx,yvert+dy)
        rb = Rectangle(xp-dx,yp-dy,xp+dx,yp+dy)

        d_x = min(ra.xmax, rb.xmax) - max(ra.xmin, rb.xmin)
        d_y = min(ra.ymax, rb.ymax) - max(ra.ymin, rb.ymin)
        d_x *= dx**-1
        d_y *= dy**-1
        if (d_x >= 0) and (d_y >= 0):

            return d_x * d_y
        else:
            return 0

    def makegrid_histogram(self,npix):

        x = y = np.linspace(-self.max_radius_kpc, self.max_radius_kpc, npix)
        dx, dy = x[1] - x[0], y[1] - y[0]

        h1, _, _ = np.histogram2d(self.x, self.y, bins=npix,weights=self.mass,
                                  range = ([-self.max_radius_kpc,self.max_radius_kpc],[-self.max_radius_kpc,self.max_radius_kpc]))

        area = dx*dy

        return h1*area**-1

    def makegrid_histogram_interp(self,npix):

        x = y = np.linspace(-self.max_radius_kpc, self.max_radius_kpc, npix)
        dx, dy = x[1] - x[0], y[1] - y[0]

        ranges = [-self.max_radius_kpc,self.max_radius_kpc],[-self.max_radius_kpc,self.max_radius_kpc]

        h1, _,_ = np.histogram2d(self.x,self.y,weights=self.mass,bins=npix,range=(ranges))
        h2,_,_ = np.histogram2d(self.x+dx,self.y+dy,weights=self.mass,bins=npix,range=(ranges))
        h3,_,_ = np.histogram2d(self.x+dx,self.y-dy,weights=self.mass,bins=npix,range=(ranges))
        h4,_,_ = np.histogram2d(self.x-dx,self.y+dy,weights=self.mass,bins=npix,range=(ranges))
        h5, _, _ = np.histogram2d(self.x - dx, self.y - dy,weights=self.mass,bins=npix,range=(ranges))

        h9,_,_= np.histogram2d(self.x+dx,self.y,bins=npix,weights=self.mass,range=(ranges))
        h6, _,_ = np.histogram2d(self.x-dx, self.y,bins=npix,weights=self.mass,range=(ranges))
        h7,_,_ = np.histogram2d(self.x, self.y + dy,bins=npix,weights=self.mass,range=(ranges))
        h8,_,_ = np.histogram2d(self.x, self.y - dy,bins=npix,weights=self.mass,range=(ranges))


        h = h1*0.5 + (h2 + h3 + h4 + h5)*0.125*np.sqrt(2)**-1 + (h6 + h7 + h8 +h9)*0.125*np.sqrt(2)

        area = dx*dy

        return h*area**-1

    def makegrid_CIC_old(self,npix,save_to_fits=True,fits_name=''):

        x = y = np.linspace(-self.max_radius_kpc,self.max_radius_kpc,npix)
        dx,dy = x[1]-x[0],y[1]-y[0]

        xx,yy = np.meshgrid(x,y)

        weights = np.zeros_like(xx.ravel())

        coords = np.vstack((xx.ravel(),yy.ravel()))

        for p in range(0,len(self.x)):

            inds = self._find_adjacent([self.x[p],self.y[p]],coords,dx*0.5)

            xvert = coords[0,inds]
            yvert = coords[1,inds]

            for idx,verticies in enumerate(zip(xvert,yvert)):

                weights[inds[idx]] += self._weights(verticies[0],verticies[1],self.x[p],self.y[p],dx,dy)*(9*dx*dy)**-1
            print np.sum(weights[inds])
            a=input('continue')

        return weights.reshape(npix,npix)

    def makegrid_kde(self,npix,save_to_fits=True,fits_name=''):

        x = y = np.linspace(-self.max_radius_kpc,self.max_radius_kpc,npix)
        xx,yy = np.meshgrid(x,y)
        from scipy.stats.kde import gaussian_kde
        kde = gaussian_kde(np.vstack((self.x,self.y)))

        return (kde(np.vstack((xx.ravel(),yy.ravel()))).reshape(npix,npix))

class ParticleLoad:

    z = 0.5
    mass_unit = 10**10
    mass_convert = 0.7 #from M/h to M
    coord_convert = (1+z)**-1 #from comoving kpc to physical kpc

    def __init__(self,name,path='',DM_particle_mass = 1.4*10**6, rotation=0):

        self.fname = path+name

        f1 = h5py.File(self.fname, 'r')

        DM_particle_mass_high_res = DM_particle_mass*self.mass_convert
        DM_particle_mass_low_res = f1['PartType2']['Masses'][:]*self.mass_unit*self.mass_convert

        gas_particle_mass = f1['PartType0']['Masses'][:]*self.mass_unit*self.mass_convert
        star_particle_mass = f1['PartType4']['Masses'][:]*self.mass_unit*self.mass_convert

        self.darkmatter_highres = Particle(DM_particle_mass_high_res,self.coord_convert,rotation)
        self.darkmatter_lowres = Particle(DM_particle_mass_low_res,self.coord_convert,rotation)
        self.stars = Particle(star_particle_mass,self.coord_convert,rotation)
        self.gas = Particle(gas_particle_mass,self.coord_convert,rotation)

    def unpack_all(self):

        self.unpack_gas()
        self.unpack_DMhighres()
        self.unpack_DMlowres()
        self.unpack_stars()

    def unpack_gas(self):

        data = []

        if isinstance(self.fname,list):
            for name in self.fname:
                data.append(h5py.File(name, 'r'))

        else:
            data.append(h5py.File(self.fname,'r'))

        for set in data:
            self.gas.load(set['PartType0'])

    def unpack_stars(self):

        data = []

        if isinstance(self.fname,list):
            for name in self.fname:
                data.append(h5py.File(name, 'r'))

        else:
            data.append(h5py.File(self.fname,'r'))

        for set in data:
            self.stars.load(set['PartType4'])

    def unpack_DMhighres(self):

        data = []

        if isinstance(self.fname,list):

            for name in self.fname:

                data.append(h5py.File(name, 'r'))

        else:
            data.append(h5py.File(self.fname,'r'))

        for set in data:
            self.darkmatter_highres.load(set['PartType1'])

    def unpack_DMlowres(self):

        data = []

        if isinstance(self.fname,list):
            for name in self.fname:
                data.append(h5py.File(name, 'r'))

        else:
            data.append(h5py.File(self.fname,'r'))

        for set in data:
            self.darkmatter_lowres.load(set['PartType2'])

hdf5path = os.getenv("HOME")+'/data/Nbody_sims/FIRE_medium_8_12/'

name = 'snapshot_340.hdf5'

nbody = ParticleLoad(name,hdf5path,rotation=0)

nbody.unpack_DMhighres()
nbody.unpack_DMlowres()
nbody.unpack_gas()
nbody.unpack_stars()

import matplotlib.pyplot as plt

zlens,zsrc = 0.5,1.1
c = Cosmo(zlens,zsrc,compute=True)

Rmax_kpc = 30*c.kpc_per_asec(zlens)
Rmax_kpc = 250

xcenter,ycenter = 35973,32232
npix = 1200
res = 2*Rmax_kpc*npix**-1
# kpc per pixel

dm_highres_x = nbody.darkmatter_highres.x
dm_highres_y = nbody.darkmatter_highres.y

dm_lowres_x = nbody.darkmatter_lowres.x
dm_lowres_y = nbody.darkmatter_lowres.y

starsx = nbody.stars.x
starsy = nbody.stars.y

gasx = nbody.gas.x
gasy = nbody.gas.y

conv_map_highres = ConvergenceMap(particle_mass=nbody.darkmatter_highres.masses,
                          particle_x=dm_highres_x,
                          particle_y=dm_highres_y,
                          center_x=xcenter,center_y=ycenter,max_radius_kpc=Rmax_kpc)

conv_map_lowres = ConvergenceMap(particle_mass=nbody.darkmatter_lowres.masses,
                          particle_x=dm_lowres_x,
                          particle_y=dm_lowres_y,
                          center_x=xcenter,center_y=ycenter,max_radius_kpc=Rmax_kpc)

conv_map_stars = ConvergenceMap(particle_mass=nbody.stars.masses,
                          particle_x=starsx,
                          particle_y=starsy,
                          center_x=xcenter,center_y=ycenter,max_radius_kpc=Rmax_kpc)

conv_map_gas = ConvergenceMap(particle_mass=nbody.gas.masses,
                          particle_x=gasx,
                          particle_y=gasy,
                          center_x=xcenter,center_y=ycenter,max_radius_kpc=Rmax_kpc)

grid_dm_highres = conv_map_highres.makegrid_histogram_interp(npix=npix)
grid_dm_lowres = conv_map_lowres.makegrid_histogram_interp(npix=npix)
grid_stars = conv_map_stars.makegrid_histogram_interp(npix=npix)
grid_gas = conv_map_gas.makegrid_histogram_interp(npix=npix)

extent = [-Rmax_kpc,Rmax_kpc,-Rmax_kpc,Rmax_kpc]
density = grid_dm_highres

density *= c.kpc_per_asec(zlens)**2*c.sigmacrit**-1
#L = np.shape(density)[0]
#x = y = np.linspace(-L,L,L)
#xx,yy = np.meshgrid(x,y)
#r = np.sqrt(xx**2+yy**2)*(L)**-1
#density[np.where(r>1)] = 0
#density = np.pad(density,20,'constant',constant_values=0)

#print res*np.shape(density)[0]*0.5*c.kpc_per_asec(zlens)**-1

#if os.path.exists('FIRE_convergence.fits'):
#    os.remove('FIRE_convergence.fits')
#hdu = pyfits.PrimaryHDU()
#hdu.data = density
#hdu.writeto('FIRE_convergence.fits')
#exit(1)
plt.imshow(np.log10(density),extent=extent,origin='lower',alpha=1,cmap='viridis',vmin=-2.5,vmax=1);plt.colorbar(label=r'$\log_{10}(\kappa)$');

plt.tight_layout()
plt.savefig('FIRE_halo_DMonly.pdf')
plt.show()

