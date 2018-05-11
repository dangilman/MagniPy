import numpy as np
from MagniPy.LensBuild.Cosmology.cosmology import Cosmo

class TNFW:

    def __init__(self,z=None,zsrc=None,c_turnover=True,cosmology=None):
        """
        adopting a standard cosmology, other cosmologies not yet implemented
        :param z1: lens redshift
        :param z2: source redshift
        :param h: little h
        """

        if cosmology is None:
            if z is None or zsrc is None:
                print 'Warning; no cosmology specified.'
            else:
                self.cosmology = Cosmo(zd=z, zsrc=zsrc, compute=False)
                self.z, self.zsrc = z, zsrc

        else:
            self.cosmology = cosmology
            self.z, self.zsrc = cosmology.zd, cosmology.zsrc

        self.c_turnover=c_turnover

    def M_finite(self, rho, Rs, tau):

        t2 = tau**2
        return 4*np.pi*Rs**3*rho*t2*(t2+1)**-2*(
            (t2-1)*np.log(tau) + np.pi*tau - (t2+1))

    def def_angle(self, x, y, Rs=None, theta_Rs=None, r_trunc=None, center_x=0, center_y=0):

        x_loc = x - center_x
        y_loc = y - center_y

        r = np.sqrt(x_loc ** 2 + y_loc ** 2)

        xnfw = r * Rs ** -1

        tau = r_trunc * Rs ** -1

        xmin = 0.00000001

        if isinstance(xnfw,float) or isinstance(xnfw,int):
            xnfw = max(xmin,xnfw)
        else:
            xnfw[np.where(xnfw<xmin)] = xmin

        magdef = theta_Rs * (1 + np.log(0.5)) ** -1 * self.t_fac(xnfw, tau) * xnfw ** -1

        return magdef * x_loc * (xnfw*Rs) ** -1, magdef * y_loc * (xnfw*Rs) ** -1

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

    def L(self,x,tau):

        return np.log(x*(tau+np.sqrt(tau**2+x**2))**-1)

    def t_fac(self, x, tau):
        return tau ** 2 * (tau ** 2 + 1) ** -2 * (
        (tau ** 2 + 1 + 2 * (x ** 2 - 1)) * self.F(x) + tau * np.pi + (tau ** 2 - 1) * np.log(tau) +
        np.sqrt(tau ** 2 + x ** 2) * (-np.pi + self.L(x, tau) * (tau ** 2 - 1) * tau ** -1))

    def _F(self, X, tau):
        """
        analytic solution of the projection integral
        (convergence)

        :param x: R/Rs
        :type x: float >0
        """
        t2 = tau ** 2
        Fx = self.F(X)

        return t2 * (2 * np.pi * (t2 + 1) ** 2) ** -1 * (
            ((t2 + 1) * (X ** 2 - 1) ** -1) * (1 - Fx)
            +
            2 * Fx
            -
            np.pi * (t2 + X ** 2) ** -.5
            +
            (t2 - 1) * (tau * (t2 + X ** 2) ** .5) ** -1 * self.L(X, tau)
        )

    def kappa(self,x, y, Rs=None, theta_Rs=None, r_trunc=None, center_x=0, center_y=0):

        x_loc = x - center_x
        y_loc = y - center_y

        r = np.sqrt(x_loc ** 2 + y_loc ** 2)

        xnfw = r * Rs ** -1

        tau = r_trunc * Rs ** -1

        xmin = 0.00000001

        if isinstance(xnfw, float) or isinstance(xnfw, int):
            xnfw = max(xmin, xnfw)
        else:
            xnfw[np.where(xnfw < xmin)] = xmin

        ks = theta_Rs*(4*Rs*(np.log(0.5)+1))**-1

        return 2*ks*self._F(xnfw,tau)

    def params(self, x=None,y=None,mass=float, mhm=None,truncation=None,c=None,**kwargs):

        assert mhm is not None
        assert mass is not None

        rsdef,Rs,rho_mpc,Rs_mpc,r200_mpc = self.nfw_physical2angle(mass, c)

        #ks = rsdef*(4*rs*(np.log(0.5)+1))**-1

        subkwargs = {}
        otherkwargs = {}

        otherkwargs['name'] = 'TNFW'
        subkwargs['theta_Rs'] = rsdef
        subkwargs['Rs'] = Rs
        subkwargs['center_x'] = x
        subkwargs['center_y'] = y

        if 'r_trunc' in kwargs:
            subkwargs['r_trunc'] = kwargs['r_trunc']
        else:
            if truncation.routine == 'fixed_radius':
                subkwargs['r_trunc'] = truncation.fixed_radius(Rs*c)
            elif truncation.routine == 'virial3d':
                subkwargs['r_trunc'] = truncation.virial3d(mass)
            else:
                raise Exception('specify truncation.')

        otherkwargs['mass'] = mass
        otherkwargs['c'] = c
        otherkwargs['name'] = 'TNFW'

        otherkwargs['mass_finite'] = self.M_finite(rho_mpc,Rs_mpc,subkwargs['r_trunc']*Rs**-1)

        return subkwargs,otherkwargs

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


    def tau(self,m,rt,mhm=False):

        ks,rs = self.nfw_params(m,mhm=mhm)

        return rt*rs**-1

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

        return theta_Rs / self.cosmology.get_epsiloncrit(self.z,self.cosmology.zsrc) / self.cosmology.D_A(0,self.z) / self.cosmology.arcsec, \
               Rs_angle, rho0, Rs, r200

    def M_physical(self,m200,mhm=0):
        """

        :param m200: m200
        :return: physical mass corresponding to m200
        """
        c = self.nfw_concentration(m200,mhm=mhm)
        rho0, Rs, r200 = self.nfwParam_physical(m200,c)
        return 4*np.pi*rho0*Rs**3*(np.log(1+c)-c*(1+c)**-1)


    def f(tau):
        return tau ** 2 * (tau ** 2 + 1) ** -2 * ((tau ** 2 - 1) * np.log(tau) + tau * np.pi - tau ** 2 - 1)

    def tau_factor(x, t):
        return t ** 2 * ((1 + x) * (1 + t ** 2) ** 2) ** -1 * (
        -x * (1 + t ** 2) + 2 * (1 + x) * t * np.arctan(x * t ** -1) + (1 + x) * (t ** 2 - 1) * np.log(
            t * (1 + x)) - 0.5 * (1 + x) * (-1 + t ** 2) * np.log(x ** 2 + t ** 2))

