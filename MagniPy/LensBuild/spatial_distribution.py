import numpy as np

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
            angle = np.random.uniform(0,2*np.pi,Npoints)
            r = np.random.uniform(0,theta**2,Npoints)
            x = r**.5*np.cos(angle)
            y = r**.5*np.sin(angle)
            r2d = (x**2+y**2)**.5

            return x,y,r2d

        elif z > zmain:

            assert self.cosmology is not None

            theta *= self.cosmology.arcsec
            Rco = theta * self.cosmology.T_xy(0, zmain)

            Rco -= self.cosmology.arcsec*R_ein_deflection*self.cosmology.T_xy(0,z)

            theta_new = self.cosmology._comoving2physical(Rco,z)*self.cosmology.D_A(0,z)**-1
            theta_new *= self.cosmology.arcsec**-1

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

        r2d, x, y = self.TwoD.get_2dcoordinates(theta=self.rmax2d, Npoints=N, z=z, zmain = self.cosmology.zd)

        return r2d,x,y

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

        r2d, x, y =self.TwoD.draw(N=N,z=0)

        z = np.random.uniform(-self.zmax,self.zmax,N)

        r3d = (z**2+r2d**2)**.5

        for i in range(0, int(len(r3d))):

            u = np.random.rand()

            accept = acceptance_prob(r3d[i])

            while u >= accept:
                r2d_, x_, y_ = self.TwoD.draw(N=1,z=0)
                z = np.random.uniform(-self.zmax,self.zmax)
                r3d[i] = (z**2+r2d_**2)**.5
                u = np.random.rand()
                accept = acceptance_prob(r3d[i])

        return r3d, x, y

class TruncationFuncitons:

    def __init__(self,truncation_routine=None,**kwargs):

        self.truncation_routine = truncation_routine

        if truncation_routine == 'tidal_3d':
            self.function = self.tidal_3D
        elif truncation_routine == 'constant':
            self.function = self.constant_trunc
        elif truncation_routine == 'normal_distribution':
            self.function = self.gaussian
        elif truncation_routine == 'NFW_r200':
            self.function = self.NFW_r200

    def tidal_3D(self, **kwargs):
        return (kwargs['mass'] * kwargs['r3d'] ** 2 * (2 * kwargs['sigmacrit'] * kwargs['RE']) ** -1) ** (1. / 3)
    def constant_trunc(self, **kwargs):
        return self.value
    def gaussian(self,**kwargs):
        return np.absolute(np.random.normal(kwargs['mean'],kwargs['sigma'],size=kwargs['size']))
    def NFW_r200(self,**kwargs):
        return np.absolute(np.random.normal(kwargs['multiple']*kwargs['r200'],kwargs['sigma']))





