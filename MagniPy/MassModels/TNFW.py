import numpy as np
from MagniPy.LensBuild.cosmology import Cosmo

class TNFW(Cosmo):

    def __init__(self,z1=0.5,z2=1.5,c_turnover=True):
        """
        adopting a standard cosmology, other cosmologies not yet implemented
        :param z1: lens redshift
        :param z2: source redshift
        :param h: little h
        """
        Cosmo.__init__(self, zd=z1, zsrc=z2)

        self.c_turnover=c_turnover

    def def_angle(self, x_grid, y_grid, x=None, y=None, rs=None, ks=None, rt=None, shear=None, shear_theta=None, **kwargs):

        assert rs > 0

        x = x_grid - x
        y = y_grid - y
        tau = rt * rs ** -1

        r = np.sqrt(x ** 2 + y ** 2 + 0.0000000000001)
        xnfw = r * rs ** -1

        softening = 0.0001
        xnfw[np.where(xnfw<softening)]=softening

        magdef = 4*ks * rs * self.t_fac(xnfw, tau) * xnfw ** -1

        return magdef * x * r ** -1, magdef * y * r ** -1

    def F(self,x):

        if isinstance(x, np.ndarray):
            nfwvals = np.ones_like(x)

            inds1 = np.where(x < 1)
            inds2 = np.where(x > 1)
            nfwvals[inds1] = (1 - x[inds1] ** 2) ** -.5 * np.arccosh(x[inds1]**-1)
            nfwvals[inds2] = (x[inds2] ** 2 - 1) ** -.5 * np.arccos(x[inds2]**-1)
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

    def convergence(self,r=None,ks=None,rs=None,xy=None):
        if xy is not None:
            assert isinstance(xy,list)
            assert len(xy)==2

            r = np.sqrt(xy[0]**2+xy[1]**2)
        if r is None:
            r = np.sqrt(self.x**2+self.y**2)
        return 2*ks*(1-self.F(r*rs**-1))*((r*rs**-1)**2-1)**-1

    def translate_to_lensmodel(self, **args):

        newargs = {}
        newargs['rs'] = args['Rs']
        newargs['ks'] = args['theta_Rs'] * (4 * args['Rs'] * (np.log(0.5) + 1)) ** -1
        newargs['x'] = args['center_x']
        newargs['y'] = args['center_y']
        return newargs

    def translate_to_lenstronomy(self, **args):

        newargs = {}
        newargs['Rs'] = args['rs']
        newargs['theta_Rs'] = 4 * args['ks'] * args['Rs'] * (np.log(0.5) + 1)
        newargs['center_x'] = args['x']
        newargs['center_x'] = args['y']
        return newargs

    def params(self, x=None,y=None,mass=float, mhm=None,trunc=None):

        assert mhm is not None
        assert mass is not None
        assert trunc is not None

        c = self.nfw_concentration(mass, mhm)

        rsdef,rs = self.nfw_physical2angle(mass, c)

        ks = rsdef*(4*rs*(np.log(0.5)+1))**-1

        subkwargs = {}
        subkwargs['name'] = 'TNFW'
        subkwargs['ks'] = ks
        subkwargs['rs'] = rs
        subkwargs['c'] = c
        subkwargs['mass'] = mass
        subkwargs['lenstronomy_name'] = 'NFW'
        subkwargs['x'] = x
        subkwargs['y'] = y
        subkwargs['rt'] = trunc

        lenstronomy_params = {}
        lenstronomy_params['Rs'] = rs
        lenstronomy_params['theta_Rs'] = rsdef
        #lenstronomy_params['t'] = trunc
        lenstronomy_params['center_x'] = x
        lenstronomy_params['center_y'] = y

        return subkwargs,lenstronomy_params

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

        return (3*M/(4*np.pi*self.rhoc*200))**(1./3.)

    def M_r200(self, r200):
        """

        :param r200: r200 in comoving Mpc/h
        :return: M200
        """
        return self.rhoc*200 * r200**3 * 4*np.pi/3.

    def rho0_c(self, c):
        """
        computes density normalization as a function of concentration parameter
        :return: density normalization in h^2/Mpc^3 (comoving)
        """
        return 200./3*self.rhoc*c**3/(np.log(1+c)-c/(1+c))

    def nfw_concentration(self, m, mhm, g1=60,g2=.17):

        z = self.zd
        def beta(z):
            return 0.026*z - 0.04

        redshift_factor = 0.842 # corrects to z=0.5 result in COCO Statistical Props
        c_cdm = redshift_factor*7*(m*10**-12)**-.098 # from Maccio et al 2008; similar to Bullock model

        if self.c_turnover:

            c_wdm = c_cdm*(1+g1*mhm*m**-1)**-g2
            #c_wdm *= (1+z)**beta(z)

            return c_wdm
        else:

            return c_cdm

    def tau(self,m,rt,mhm=False):

        ks,rs = self.nfw_params(m,mhm=mhm)

        return rt*rs**-1

    def m_in_r3d(self,m,r,rt,mhm=False):

        tau = self.tau(m,rt,mhm=mhm)
        ks,rs = self.nfw_params(m,mhm=mhm)
        return 4 * ks * rs ** 2 * self.sigmacrit * np.pi * tau_factor(r*rs**-1,tau)

    def m_infinity(self,m,rt,mhm=False):
        """

        :param m: comoving M200
        :param rt: physical truncation radius in arcseconds
        :param mhm: half mode mass; set to zero to avoid concentration damping
        :return: physical total mass of halo
        """
        tau = self.tau(m, rt, mhm=mhm)
        ks, rs = self.nfw_params(m, mhm=mhm)
        return 4*np.pi*ks*rs**2*self.sigmacrit*f(tau)

    def nfwParam_physical(self, M, c):
        """
        returns the NFW parameters in physical units
        :param M: physical mass in M_sun
        :param c: concentration
        :return:
        """
        r200 = self.r200_M(M * self.h) * self.h * self.a_z(self.zd)  # physical radius r200
        rho0 = self.rho0_c(c) / self.h**2 / self.a_z(self.zd)**3 # physical density in M_sun/Mpc**3
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
        Rs_angle = Rs / self.D_d / self.arcsec #Rs in asec

        theta_Rs = rho0 * (4 * Rs ** 2 * (1 + np.log(1. / 2.)))

        return theta_Rs / self.epsilon_crit / self.D_d / self.arcsec, Rs_angle

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

