import numpy as np
import random
import math

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
            return x,y,r2d,zcoord

        elif z > zmain:

            assert self.cosmology is not None

            D_12 = self.cosmology.D_A(zmain, z)
            D_os = self.cosmology.D_A(0,self.cosmology.zsrc)
            D_1s = self.cosmology.D_A(zmain, self.cosmology.zsrc)
            D_o2 = self.cosmology.D_A(0, z)

            beta = D_12 * D_os * (D_o2 * D_1s) ** -1

            theta_new = theta*(1 - beta)

            angle = np.random.uniform(0, 2 * np.pi, Npoints)
            r = np.random.uniform(0, theta_new ** 2, Npoints)
            x = r ** .5 * np.cos(angle)
            y = r ** .5 * np.sin(angle)
            r2d = (x ** 2 + y ** 2) ** .5
            zcoord = np.random.uniform(0, (theta ** 2 - r2d ** 2) ** 0.5)

            return x, y, r2d, zcoord

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

class NFW_2D:

    def __init__(self, rmin = None, rmax2d=None, rs = None, xoffset=0, yoffset = 0, tidal_core=False):

        self.rmax2d = rmax2d
        self.rs = rs

        if rmin is None:
            rmin = rs*0.1

        self.xmin = rmin*rs**-1
        self.xmax = rmax2d*rs**-1
        self.xoffset,self.yoffset = xoffset,yoffset
        self.tidal_core = tidal_core
        assert self.xmax>self.xmin

    def core_damp(self,r,gamma=0.8,rs_scale=0.6):

        x_inv = (rs_scale*self.rs)*r**-1

        return np.exp(-gamma*x_inv)

    def nfw_kr(self,X):

        def f(x):

            if isinstance(x,int) or isinstance(x,float):
                if x>1:
                    return np.arctan((x**2-1)**.5)*(x**2-1)**-.5
                elif x<1:
                    return np.arctanh((1-x**2)**.5)*(1-x**2)**-.5
                else:
                    return 1
            else:
                inds1 = np.where(x<1)
                inds2 = np.where(x>1)

                vals = np.ones_like(x)
                flow = (1-x[inds1]**2)**.5
                fhigh = (x[inds2]**2-1)**.5

                vals[inds1] = np.arctanh(flow)*flow**-1
                vals[inds2] = np.arctan(fhigh)*fhigh**-1

                return vals

        return 2*(1-f(X))*(X**2-1)**-1

    def nfw_bound(self,X,alpha=0.1,xmin=None):

        norm = self.nfw_kr(self.xmin)

        if isinstance(X,int) or isinstance(X,float):

            if X>self.xmin:
                return norm*(X*self.xmin**-1)**-alpha
            else:
                return norm

        else:
            return norm*(X*(self.xmin)**-1)**-alpha

    def C_inv(self,x,xmax=10,xmin=1e-3,alpha=0.1):

        a = 1-alpha
        A0 = (1-alpha)*(xmax**a - xmin**a)**-1

        return (a*x*A0**-1 + xmin**a)**(a**-1)

    def draw(self,N):

        r2d = self.draw_r2d(N)
        x,y = self.r2d_to_xy(r2d)
        return x+self.xoffset,y+self.yoffset,r2d,None

    def draw_r2d(self,N):

        samples = []

        while len(samples)<N:

            u = np.random.uniform(0,1)
            x_sample = self.C_inv(u,xmax=self.xmax,xmin=0)

            if x_sample<=self.xmin:
                ratio = 1
            else:
                ratio = self.nfw_kr(x_sample) * self.nfw_bound(x_sample, xmin=self.xmin)

            if self.tidal_core:
                ratio *= self.core_damp(x_sample*self.rs)

            if ratio > np.random.uniform(0,1):
                samples.append(x_sample)

        samples = np.array(samples)
        return samples*self.rs

    def r2d_to_xy(self,r2d):

        theta = np.random.uniform(0,2*np.pi,len(r2d))
        xcoord = r2d*np.cos(theta)
        ycoord = r2d*np.sin(theta)

        return np.array(xcoord),np.array(ycoord)




