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

            theta_new = theta*(1 - self.cosmology.D_A(zmain, z)*self.cosmology.D_s*(self.cosmology.D_A(0, z)*self.cosmology.D_ds)**-1)

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

        return x,y,r3d

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
            xloc = [self.xlocations]
        if isinstance(self.ylocations,float) or isinstance(self.ylocations,int):
            yloc = [self.ylocations]

        for imgnum in range(0, len(xloc)):

            ximg, yimg = xloc[imgnum], yloc[imgnum]

            n = self._prob_round(N)

            x_locations,y_locations, R = self.TwoD.draw(n,z=z)

            try:
                x = np.append(np.array(x_locations)+ximg)
                y = np.append(np.array(y_locations)+yimg)

            except:
                x = np.array(x_locations) + ximg
                y = np.array(y_locations) + yimg

        return x,y,np.sqrt(x**2+y**2)

class NFW_2D:

    def __init__(self, rmax2d=None, rs = None):

        self.rmax2d = rmax2d
        self.rs = rs
        self.xmin = 1e-9
        self.profile = self.nfw_profile(np.linspace(self.rs*0.001,rmax2d,100))
        start,end = self.profile[0],self.profile[-1]

        self.slope = (end-start)*(rmax2d-0.001*self.rs)**-1
        self.intercept = self.profile[0] - self.slope*self.rs*0.001



    def bound(self,x):

        if isinstance(x,np.ndarray) or isinstance(x,list):
            x[np.where(x<self.xmin)] = self.xmin
        else:
            if x<self.xmin:
                return self.xmin

        return self.slope*x + self.intercept

    def prob(self,x):
        return self.nfw_profile(x)*self.bound(x)**-1

    def nfw_profile(self,r):

        if isinstance(r,list) or isinstance(r,np.ndarray):
            x = r*self.rs**-1
            xmin = self.xmin
            x[np.where(x<xmin)] = xmin

            vals = np.ones_like(x)
            inds1 = np.where(x<1)
            inds2 = np.where(x>1)

            vals[inds1] = np.arctanh((1-x[inds1]**2)**.5)*(1-x[inds1]**2)**-.5
            vals[inds2] = np.arctan((x[inds2] ** 2 - 1) ** .5) * (x[inds2] ** 2-1) ** -.5

            return vals
        else:
            xmin = self.xmin
            x = r*self.rs**-1
            if x<xmin:
                x=xmin
            if x<1:
                return np.arctanh((1-x**2)**.5)*(1-x**2)**-.5
            elif x>1:
                return np.arctan((x ** 2-1) ** .5) * (x ** 2-1) ** -.5
            else:
                return 1

    def cdf(self,x):
        y2,y1 = self.rmax2d,0.001*self.rs
        A = 0.5*self.slope*(y2**2-y1**2)**2+self.intercept*(y2-y1)
        return -0.5+(self.intercept**2*self.slope**-2 - 2*A*self.slope**-1*x - 2*self.intercept*self.slope**-1*self.rs*0.001 - (self.rs*0.001)**2)**.5

    def draw(self,N):

        xsamples,ysamples = [],[]
        while len(xsamples)<N:
            r_rand = np.random.uniform(0,self.rmax2d**2)
            theta = np.random.uniform(0,2*np.pi)
            xrand = r_rand**.5*np.cos(theta)
            yrand = r_rand**.5*np.sin(theta)
            r2d_rand = (xrand**2+yrand**2)**.5
            u = np.random.random()


            if u<=self.prob(r2d_rand):
                xsamples.append(xrand)
                ysamples.append(yrand)
        xsamples = np.array(xsamples)
        ysamples = np.array(ysamples)

        return xsamples,ysamples,np.sqrt(xsamples**2+ysamples**2)








