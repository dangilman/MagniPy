import numpy as np
from MagniPy.LensBuild.Cosmology.cosmology import Cosmo

class NFW:

    def __init__(self,z=None,zsrc=None,c_turnover=True,cosmology=None):
        """
        adopting a standard cosmology, other cosmologies not yet implemented
        :param z1: lens redshift
        :param z2: source redshift
        :param h: little h
        """
        if cosmology is None:
            self.cosmology = Cosmo(zd=z, zsrc=zsrc, compute=False)

        else:
            self.cosmology = cosmology

        self.z = z
        self.zsrc= zsrc

        self.c_turnover=c_turnover

    def def_angle(self, x, y, Rs=None, theta_Rs=None, center_x=0, center_y=0):

        x_loc = x - center_x
        y_loc = y - center_y

        r = (x_loc ** 2 + y_loc ** 2)**.5
        xnfw = r * Rs ** -1

        xmin = 0.0000001

        if isinstance(xnfw,float) or isinstance(xnfw,int):
            xnfw = max(xnfw,xmin)
        else:
            xnfw[np.where(xnfw < xmin)] = xmin

        ks = theta_Rs * (4 * Rs * (np.log(0.5) + 1)) ** -1
        magdef = 4*ks*Rs*(np.log(0.5*xnfw) + self.F(xnfw))*xnfw**-1

        return magdef * x_loc * (Rs*xnfw) ** -1, magdef * y_loc * (Rs*xnfw) ** -1

    def F(self,x):

        if isinstance(x, np.ndarray):
            nfwvals = np.ones_like(x)
            inds1 = np.where(x < 1)
            inds2 = np.where(x > 1)
            nfwvals[inds1] = (1 - x[inds1] ** 2) ** -.5 * np.arctanh((1 - x[inds1] ** 2) ** .5)
            nfwvals[inds2] = (x[inds2] ** 2 - 1) ** -.5 * np.arctan((x[inds2] ** 2 - 1) ** .5)
            return nfwvals

        elif isinstance(x, float) or isinstance(x, int):
            if x == 1:
                return 1
            if x < 1:
                return (1 - x ** 2) ** -.5 * np.arctanh((1 - x ** 2) ** .5)
            else:
                return (x ** 2 - 1) ** -.5 * np.arctan((x ** 2 - 1) ** .5)

    def convergence(self,x,y,theta_Rs=None,Rs=None,center_x=None,center_y=None):

        x,y = x - center_x,y - center_y

        r = (x**2+y**2)**.5

        xnfw = r*Rs**-1

        if isinstance(xnfw,float) or isinstance(xnfw,int):
            if xnfw < 0.0001:
                xnfw = 0.0001
        else:
            xnfw[np.where(xnfw)<0.0001] = 0.0001
        ks = theta_Rs*(4*Rs*(np.log(0.5)+1))**-1

        return 2*ks*(1-self.F(xnfw))*((xnfw)**2-1)**-1

    def params(self, x=None,y=None,mass=float, mhm=None ,trunc = None):

        assert mhm is not None
        assert mass is not None

        c = self.nfw_concentration(mass, mhm)

        rsdef,Rs = self.nfw_physical2angle(mass, c)

        #ks = rsdef*(4*rs*(np.log(0.5)+1))**-1

        subkwargs = {}
        otherkwargs = {}

        subkwargs['theta_Rs'] = rsdef
        subkwargs['Rs'] = Rs
        subkwargs['center_x'] = x
        subkwargs['center_y'] = y

        otherkwargs['mass'] = mass
        otherkwargs['c'] = c
        otherkwargs['name'] = 'NFW'

        return subkwargs, otherkwargs


    def M200(self, Rs, rho0, c):
        """
        M(R_200) calculation for NFW profile

        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param c: concentration
        :type c: float [4,40]
        :return: M(R_200) density
        """
        return 4*np.pi*rho0*Rs**3*(np.log(1+c)-c/(1+c))

    def r200_M(self, M):
        """
        computes the radius R_200 of a halo of mass M in comoving distances M/h

        :param M: halo mass in M_sun/h
        :type M: float or numpy array
        :return: radius R_200 in comoving Mpc/h
        """

        return (3*M/(4*np.pi*self.cosmology.get_rhoc()*200))**(1./3.)

    def M_r200(self, r200):
        """

        :param r200: r200 in comoving Mpc/h
        :return: M200
        """
        return self.cosmology.get_rhoc()*200 * r200**3 * 4*np.pi/3.

    def rho0_c(self, c):
        """
        computes density normalization as a function of concentration parameter
        :return: density normalization in h^2/Mpc^3 (comoving)
        """
        return 200./3*self.cosmology.get_rhoc()*c**3/(np.log(1+c)-c/(1+c))

    def nfw_concentration(self, m, logmhm, g1=60,g2=.17):

        def beta(z):
            return 0.026*z - 0.04

        redshift_factor = 0.842 # corrects to z=0.5 result in COCO Statistical Props
        c_cdm = redshift_factor*7*(m*10**-12)**-.098 # from Maccio et al 2008; similar to Bullock model

        if self.c_turnover:

            if logmhm==0:
                return c_cdm
            else:
                mhm = 10**logmhm

            c_wdm = c_cdm*(1+g1*mhm*m**-1)**-g2
            #c_wdm *= (1+z)**beta(z)

            return c_wdm
        else:

            return c_cdm

    def nfwParam_physical(self, M, c):
        """
        returns the NFW parameters in physical units
        :param M: physical mass in M_sun
        :param c: concentration
        :return:
        """
        h = self.cosmology.cosmo.h

        r200 = self.r200_M(M * h) * h * self.cosmology.a_z(self.z)  # physical radius r200
        rho0 = self.rho0_c(c) / h**2 / self.cosmology.a_z(self.z)**3 # physical density in M_sun/Mpc**3
        Rs = r200/c
        return rho0, Rs, r200

    def nfw_physical2angle(self, M, c):
        """
        converts the physical mass and concentration parameter of an NFW profile into the lensing quantities
        :param M: mass enclosed 200 \rho_crit
        :param c: NFW concentration parameter (r200/r_s)
        :return: theta_Rs (observed bending angle at the scale radius, Rs_angle (angle at scale radius) (in units of arcsec)
        """

        rho0, Rs, r200 = self.nfwParam_physical(M, c)
        Rs_angle = Rs / self.cosmology.D_A(0,self.z) / self.cosmology.arcsec #Rs in asec

        theta_Rs = rho0 * (4 * Rs ** 2 * (1 + np.log(1. / 2.)))

        return theta_Rs / self.cosmology.get_epsiloncrit(self.z,self.zsrc) / self.cosmology.D_A(0,self.z) / self.cosmology.arcsec, Rs_angle

    def M_physical(self,m200,mhm=0):
        """

        :param m200: m200
        :return: physical mass corresponding to m200
        """
        c = self.nfw_concentration(m200,mhm=mhm)
        rho0, Rs, r200 = self.nfwParam_physical(m200,c)
        return 4*np.pi*rho0*Rs**3*(np.log(1+c)-c*(1+c)**-1)
