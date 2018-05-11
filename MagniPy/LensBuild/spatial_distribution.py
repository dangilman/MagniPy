import numpy as np
import random
import math

def ray_position(z,zmain,zsrc,x,y,cosmology_calc=None,angle_deflection=None):

    if z<=zmain:
        return x,y
    else:
        if cosmology_calc is None:
            from MagniPy.LensBuild.Cosmology.cosmology import Cosmo
            cosmology_calc = Cosmo(zmain,zsrc,compute=False)
        r = (x ** 2 + y ** 2) ** .5

        rnew = r*(1-cosmology_calc.beta(z,zmain,zsrc))

        theta = np.arctan2(y,x)

        xnew = rnew*np.cos(theta)
        ynew = rnew*np.sin(theta)
        return xnew,ynew

class TwoDCoords:

    def __init__(self, cosmology=None):

        self.cosmology = cosmology

    def get_2dcoordinates(self,theta,Npoints,z=None,zmain=None):
        """

        :param z: redshift
        :param theta: maximum angle
        :param Npoints:
        : R_ein_deflection: deflection at the Einstein radius in arcsec
        :return: x,y,r2d positions in comoving units
        """
        assert z != 0
        if z <= self.cosmology.zd:

            angle = np.random.uniform(0,2*np.pi,int(Npoints))
            r = np.random.uniform(0,(theta**2),int(Npoints))

            x = r**.5*np.cos(angle)
            y = r**.5*np.sin(angle)
            r2d = (x**2+y**2)**.5
            zcoord = np.random.uniform(0,(theta**2 - r2d**2)**0.5)
            return x,y,r2d,(zcoord**2+r2d**2)**.5

        elif z > zmain:

            assert self.cosmology is not None

            beta = self.cosmology.beta(z,self.cosmology.zd,self.cosmology.zsrc)

            theta_new = theta*(1 - beta)

            angle = np.random.uniform(0, 2 * np.pi, Npoints)
            r = np.random.uniform(0, theta_new ** 2, Npoints)
            x = r ** .5 * np.cos(angle)
            y = r ** .5 * np.sin(angle)
            r2d = (x ** 2 + y ** 2) ** .5
            zcoord = np.random.uniform(0, (theta ** 2 - r2d ** 2) ** 0.5)

            return x, y, r2d, (zcoord**2+r2d**2)**.5

    def distance_beta(self,z,zmain,zsrc):
        D_12 = self.cosmology.D_A(zmain, z)
        D_os = self.cosmology.D_A(0, zsrc)
        D_1s = self.cosmology.D_A(zmain, zsrc)
        D_o2 = self.cosmology.D_A(0, z)

        return D_12 * D_os * (D_o2 * D_1s) ** -1

class Uniform_2d:

    def __init__(self,rmax2d=None,cosmology=None):

        self.cosmology = cosmology
        self.zmain = cosmology.zd
        self.zsrc = cosmology.zsrc

        self.rmax2d = rmax2d

        self.TwoD = TwoDCoords(cosmology=cosmology)

    def draw(self,N,z):

        x, y, r2d, r3d = self.TwoD.get_2dcoordinates(theta=self.rmax2d, Npoints=N, z=z, zmain = self.cosmology.zd)

        return x,y,r2d,r3d

class Uniform_cored_nfw:

    def __init__(self,rmax2d=None,rmaxz=None,cosmology=None,rc=None):

        self.TwoD = Uniform_2d(rmax2d=rmax2d,cosmology=cosmology)
        self.zmax = rmaxz
        self.rmax2d = rmax2d
        self.rc = rc

    def _draw_z(self,rmaxz,N):
        """

        :param rmaxz: physical z coordinate in kpc
        :param N: number to draw
        :return:
        """
        return np.random.uniform(-rmaxz,rmaxz,N)

    def r3d_pdf_cored(self,r):
        def f(x):
            return np.arcsinh(x)-x*(x**2+1)**-.5

        norm = (4*np.pi*self.rc**3*f(self.zmax*self.rc**-1))**-1
        return norm*(1+r**2*self.rc**-2)**-1.5

    def draw(self, N, z=None):

        def acceptance_prob(r):

            return self.r3d_pdf_cored(r) * self.r3d_pdf_cored(0) ** -1

        x, y, r2d, _ =self.TwoD.draw(N=N,z=self.TwoD.cosmology.zd)

        z = np.random.uniform(-self.zmax,self.zmax,N)

        r3d = (z**2+r2d**2)**.5

        for i in range(0, int(len(r3d))):

            u = np.random.rand()

            accept = acceptance_prob(r3d[i])

            while u >= accept:
                x_, y_, r2d_, _ = self.TwoD.draw(N=1,z=self.TwoD.cosmology.zd)
                z = np.random.uniform(-self.zmax,self.zmax)
                r3d[i] = (z**2+r2d_**2)**.5
                r2d[i] = r2d_
                u = np.random.rand()
                accept = acceptance_prob(r3d[i])

        return x,y,r2d,r3d

class Localized_uniform:

    def __init__(self,x_position=None,y_position=None,rmax2d=None,cosmology=None,main_lens_z=None):

        self.xlocations = x_position
        self.ylocations = y_position
        self.rmax2d = rmax2d

        self.TwoD = Uniform_2d(rmax2d=rmax2d, cosmology=cosmology)

        if main_lens_z is None:
            self.main_lens_z = cosmology.zd
        else:
            self.main_lens_z = main_lens_z

    def set_rmax2d(self,rmax2d):

        self.rmax2d = rmax2d

    def set_xy(self,x,y):

        self.xlocations = x
        self.ylocations = y

    def _prob_round(self,x):
        sign = np.sign(x)
        x = abs(x)
        is_up = random.random() < x-int(x)
        round_func = math.ceil if is_up else math.floor
        return sign * round_func(x)

    def draw(self,N,z):

        if isinstance(self.xlocations,float) or isinstance(self.xlocations,int):
            xloc = [self.xlocations]
        else:
            xloc = self.xlocations
        if isinstance(self.ylocations,float) or isinstance(self.ylocations,int):
            yloc = [self.ylocations]
        else:
            yloc = self.ylocations

        if z > self.main_lens_z:
            beta = self.TwoD.TwoD.distance_beta(z, self.main_lens_z, self.TwoD.cosmology.zsrc)
            factor = (1 - beta)
        else:
            factor = 1

        for imgnum in range(0, len(xloc)):

            ximg, yimg = xloc[imgnum]*factor, yloc[imgnum]*factor

            n = self._prob_round(N)

            x_locations,y_locations, R2d, _ = self.TwoD.draw(n,z=z)

            try:
                x = np.append(np.array(x_locations)+ximg)
                y = np.append(np.array(y_locations)+yimg)

            except:
                x = np.array(x_locations) + ximg
                y = np.array(y_locations) + yimg

        return x,y,np.sqrt(x**2+y**2),None

class NFW_3D:

    def __init__(self, rmin = None, rmax3d=None, rmax2d = None, Rs = None, xoffset=0, yoffset = 0,
                 tidal_core=False, r_core = None, cosmology=None):

        self.rmax3d = rmax3d
        self.rmax2d = rmax2d
        self.rs = Rs

        self.xoffset = xoffset
        self.yoffset = yoffset

        if rmin is None:
            rmin = Rs*0.001

        self.xmin = rmin*Rs**-1
        self.xmax = rmax3d*Rs**-1
        self.xoffset,self.yoffset = xoffset,yoffset
        self.tidal_core = tidal_core
        self.core_fac = 1
        self.r_core = r_core

        assert cosmology is not None
        self.cosmology = cosmology

        assert self.xmax>self.xmin

    def kpc_to_asec(self,kpc):

        return kpc*self.cosmology.kpc_per_asec(self.cosmology.zd)

    def nfw_rho_r(self,r):

        x = r*self.rs**-1

        if isinstance(x,float) or isinstance(x,int):
            x = max(self.xmin,x)
        else:
            x[np.where(x<self.xmin)] = self.xmin

        return (x*(1+x)**2)**-1

    def nfw_bound(self,r,alpha=0.9999999):

        X = r*self.rs**-1
        norm = self.nfw_rho_r(self.xmin*self.rs)

        if isinstance(X,int) or isinstance(X,float):

            if X>self.xmin:
                return norm*(X*self.xmin**-1)**-alpha
            else:
                return norm

        else:
            X[np.where(X < self.xmin)] = self.xmin
            return norm*(X*self.xmin**-1)**-alpha

    def _core_damp(self,r3d):

        if self.tidal_core:
            x = self.core_fac*self.r_core*r3d**-1
            return np.exp(-x)
        else:
            return 1

    def C_inv(self,x,alpha=0.99):

        a = 1-alpha
        A0 = (1-alpha)*(self.xmax**a - self.xmin**a)**-1

        return (a*x*A0**-1 + self.xmin**a)**(a**-1)

    def draw(self,N):

        r3d, x, y, r2d = [], [], [], []

        while len(r3d) < N:

            u = np.random.uniform(0, 1)

            theta = np.random.uniform(0,2*np.pi)

            r = np.random.uniform(0,self.rmax2d**2)
            x_,y_ = r**0.5*np.cos(theta),r**0.5*np.sin(theta)
            r = np.random.uniform(0,self.rmax3d**2)
            z_ = r**0.5*np.sin(np.random.uniform(0,2*np.pi))

            r2d_ = (x_**2+y_**2)**0.5

            r3d_ = (r2d_**2+z_**2)**0.5

            if r3d_*self.rs**-1 <= self.xmin:
                ratio = 1
            else:
                ratio = self.nfw_rho_r(r3d_) * self.nfw_bound(r3d_) ** -1
                ratio *= self._core_damp(r3d_)

            if ratio > np.random.uniform(0,1) and r2d_ <= self.rmax2d:
                r3d.append(r3d_)
                x.append(x_+self.xoffset)
                y.append(y_+self.yoffset)
                r2d.append(r2d_)

        x = np.array(x)
        y = np.array(y)
        r2d = np.array(r2d)
        r3d = np.array(r3d)

        return x,y,r2d,r3d







