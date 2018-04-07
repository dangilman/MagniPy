import numpy as np
import random
import math

class TwoDCoords:

    def __init__(self, cosmology=None):

        self.cosmology = cosmology

    def get_2dcoordinates(self,theta,Npoints,R_ein_deflection=1,z=None,zmain=None):
        """

        :param z: redshift
        :param theta: maximum angle
        :param Npoints:
        : R_ein_deflection: deflection at the Einstein radius in arcsec
        :return: x,y,r2d positions in comoving units
        """

        if z <= self.cosmology.zd:

            angle = np.random.uniform(0,2*np.pi,int(Npoints))
            r = np.random.uniform(0,(theta**2),int(Npoints))

            x = r**.5*np.cos(angle)
            y = r**.5*np.sin(angle)
            r2d = (x**2+y**2)**.5

            return x,y,r2d

        elif z > zmain:

            assert self.cosmology is not None

            theta_new = (1 - self.cosmology.D_A(zmain, z)*self.cosmology.D_s*(self.cosmology.D_A(0, z)*self.cosmology.D_ds)**-1)

            angle = np.random.uniform(0, 2 * np.pi, Npoints)
            r = np.random.uniform(0, theta_new ** 2, Npoints)
            x = r ** .5 * np.cos(angle)
            y = r ** .5 * np.sin(angle)
            r2d = (x ** 2 + y ** 2) ** .5

            return x, y, r2d

class Uniform_2d:

    def __init__(self,rmax2d=None,cosmology=None):

        self.cosmology = cosmology
        self.zmain = cosmology.zd
        self.zsrc = cosmology.zsrc

        self.rmax2d = rmax2d

        self.TwoD = TwoDCoords(cosmology=cosmology)

    def draw(self,N,z):

        x, y, r2d = self.TwoD.get_2dcoordinates(theta=self.rmax2d, Npoints=N, z=z, zmain = self.cosmology.zd)

        return x,y,r2d

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

        x, y, r2d =self.TwoD.draw(N=N,z=self.TwoD.cosmology.zd)

        z = np.random.uniform(-self.zmax,self.zmax,N)

        r3d = (z**2+r2d**2)**.5

        for i in range(0, int(len(r3d))):

            u = np.random.rand()

            accept = acceptance_prob(r3d[i])

            while u >= accept:
                x_, y_, r2d_ = self.TwoD.draw(N=1,z=0)
                z = np.random.uniform(-self.zmax,self.zmax)
                r3d[i] = (z**2+r2d_**2)**.5
                u = np.random.rand()
                accept = acceptance_prob(r3d[i])

        return r3d, x, y

class Localized_uniform:

    def __init__(self,xlocations=None,ylocations=None,rmax2d=None,cosmology=None):

        self.xlocations = xlocations
        self.ylocations = ylocations
        self.rmax2d = rmax2d

        self.TwoD = Uniform_2d(rmax2d=rmax2d, cosmology=cosmology)

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
            self.xlocations = [self.xlocations]
        if isinstance(self.ylocations,float) or isinstance(self.ylocations,int):
            self.ylocations = [self.ylocations]

        for imgnum in range(0, len(self.xlocations)):

            ximg, yimg = self.xlocations[imgnum], self.ylocations[imgnum]

            n = self._prob_round(N)

            x_locations,y_locations, _ = self.TwoD.draw(n,z=z)

            try:
                x = np.append(np.array(x_locations)+ximg)
                y = np.append(np.array(y_locations)+yimg)
            except:
                x = np.array(x_locations) + ximg
                y = np.array(y_locations) + yimg

        return x,y






