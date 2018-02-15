#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `magnipy` package."""

import pytest
import numpy as np
from click.testing import CliRunner
import matplotlib.pyplot as plt


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'magnipy.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output

def test_profiles_nfw(plot=True):

    #x, y = np.linspace(-0.5, .5, 200), np.linspace(-0.5, 0.5, 200)

    x = np.loadtxt('xvalues.txt')
    y = np.linspace(0,0,len(x))
    #xx, yy = np.meshgrid(x, y)

    from MagniPy.MassModels.NFW import NFW
    from lenstronomy.LensModel.Profiles.nfw import NFW as NFW_L
    from MagniPy.MassModels.nfwT_temp import NFWt
    from MagniPy.MassModels.TNFW import TNFW
    import matplotlib.pyplot as plt

    rt = 1000
    fname = 'nfwdef_rt1000.txt'

    nfw = NFW()
    nfwL = NFW_L()
    TNFW = TNFW()
    nfwTL = NFWt()

    # nfw_params,nfw_params_L = nfw.params(x=0,y=0,mass=mass,mhm=mhm)
    # nfwt_params,nfwt_params_L = nfwT.params(x=0,y=0,mass=mass,mhm=mhm,trunc=rt)

    ks, rs = 0.1, 0.1
    theta_rs = 4 * ks * rs * (1 + np.log(.5))

    xdef, ydef = nfw.def_angle(x, y, center_x=0, center_y=0, theta_Rs=theta_rs, Rs=rs)
    xdef_t, ydef_t = TNFW.def_angle(x, y, center_x=0, center_y=0, theta_Rs=theta_rs, Rs=rs, t=rt)
    xdef_L, ydef_L = nfwL.derivatives(x=x, y=y, Rs=rs, theta_Rs=theta_rs)
    xdef_Lt, ydef_Lt = nfwTL.derivatives(x=x, y=y, Rs=rs, theta_Rs=theta_rs, t=rt)

    gravlens_xdef = np.loadtxt(fname)
    # xdef,ydef = nfw.def_angle(xx,yy,0,0,rs,ks)
    # xdef_L,ydef_L = nfwL.derivatives(x=xx,y=yy,**nfw_params_L)

    if plot:
        colors=['r','k','b','g']
        labels = ['my nfw','lenstronomy nfw','my nfw rt=1000rs','lensmodel']
        angles = [xdef,xdef_L,xdef_t,gravlens_xdef]
        for i,deflection in enumerate(angles[:-1]):
            plt.plot(x,deflection,color=colors[i],label=labels[i],alpha=0.5)
            plt.scatter(x, deflection, color=colors[i], label=labels[i], alpha=0.5)
        plt.plot(x,gravlens_xdef,color=colors[-1],label=labels[-1],alpha=0.5)
        plt.scatter(x, gravlens_xdef, color=colors[-1], label=labels[-1], alpha=0.5)
        plt.legend()
        plt.show()

    else:
        np.testing.assert_almost_equal(xdef, xdef_t, decimal=8)
        np.testing.assert_almost_equal(xdef, xdef_Lt, decimal=8)
        np.testing.assert_almost_equal(xdef, gravlens_xdef, decimal=8)
        np.testing.assert_almost_equal(xdef, xdef_L, decimal=8)



def test_profiles_nfwT(plot=True):

    rt = 0.4

    x = np.loadtxt('xvalues.txt')
    y = np.linspace(0, 0, len(x))

    from MagniPy.MassModels.NFW import NFW
    from lenstronomy.LensModel.Profiles.nfw import NFW as NFW_L
    from MagniPy.MassModels.nfwT_temp import NFWt
    from MagniPy.MassModels.TNFW import TNFW
    import matplotlib.pyplot as plt

    fname = 'nfwdef_rt.4.txt'

    nfw = NFW()
    nfwL = NFW_L()
    nfwT = TNFW()
    nfwTL = NFWt()

    # nfw_params,nfw_params_L = nfw.params(x=0,y=0,mass=mass,mhm=mhm)
    # nfwt_params,nfwt_params_L = nfwT.params(x=0,y=0,mass=mass,mhm=mhm,trunc=rt)

    ks, rs = 0.1, 0.1
    theta_rs = 4 * ks * rs * (1 + np.log(.5))

    xdef, ydef = nfw.def_angle(x, y, x=0, y=0, ks=ks, rs=rs)
    xdef_t, ydef_t = nfwT.def_angle(x, y, x=0, y=0, ks=ks, rs=rs, rt=rt)
    xdef_L, ydef_L = nfwL.derivatives(x=x, y=y, Rs=rs, theta_Rs=theta_rs)
    xdef_Lt, ydef_Lt = nfwTL.derivatives(x=x, y=y, Rs=rs, theta_Rs=theta_rs, t=rt)
    gravlens_xdef = np.loadtxt(fname)
    # xdef,ydef = nfw.def_angle(xx,yy,0,0,rs,ks)
    # xdef_L,ydef_L = nfwL.derivatives(x=xx,y=yy,**nfw_params_L)

    if plot:
        colors=['r','k','b']
        labels = ['my nfw rt = 4*rs','lenstronomy nfw rt=4*rs','lensmodel']
        angles = [xdef_t,xdef_Lt,gravlens_xdef]
        for i,deflection in enumerate(angles[:-1]):
            plt.plot(x,deflection,color=colors[i],label=labels[i],alpha=0.5)
            plt.scatter(x, deflection, color=colors[i], label=labels[i], alpha=0.5)
        plt.plot(x,gravlens_xdef,color=colors[-1],label=labels[-1],alpha=0.5)
        plt.scatter(x, gravlens_xdef, color=colors[-1], label=labels[-1], alpha=0.5)
        plt.legend()
        plt.show()

    else:
        np.testing.assert_almost_equal(xdef_t, xdef_Lt, decimal=8)
        np.testing.assert_almost_equal(xdef_t, gravlens_xdef, decimal=8)

def test_solving_leq_single_plane():

    multiplane = False
    identifier = 'test_magnipy'

    from MagniPy.magnipy import Magnipy
    from MagniPy.lensdata import Data
    from MagniPy.LensBuild.lens_assemble import Deflector
    from MagniPy.MassModels import SIE
    from MagniPy.Solver.solveroutines import SolveRoutines
    import matplotlib.pyplot as plt

    zmain,zsrc = 0.5,1.5
    srcx,srcy = -0.059,0.0065
    pos_sigma = [[0.003] * 4, [0.003] * 4]
    flux_sigma = [.4] * 4
    tdel_sigma = np.array([0, 2, 2, 2]) * 10
    sigmas = [pos_sigma, flux_sigma, tdel_sigma]
    solver = SolveRoutines(0.5, 1,5)
    init = Magnipy(0.5, 1.5, use_lenstronomy_halos=True)

    datax = [-0.6894179,0.4614137,0.05046926,0.3233231]
    datay = [-0.1448493,0.3844455,0.5901034,-0.4361114]
    datam = [ 0.42598153,1,0.75500025,0.36081329]
    datat = [ 0,6.314742,6.777318,9.38729]

    start = {'R_ein': .7, 'ellip': .05, 'ellip_theta': 0, 'x': 0, 'y': 0, 'shear': 0, 'shear_theta': 0}
    start = Deflector(subclass=SIE.SIE(), redshift=zmain, tovary=True,
                      varyflags=['1', '1', '1', '1', '1', '0', '0', '0', '0', '0'], **start)

    data_to_fit = [Data(x=datax,y=datay,m=datam,t=datat,source=[srcx,srcy])]

    profiles = ['NFW']

    for profile in profiles:

        halo_models2 = ["plaw_"+profile+"_[.005]_6_10_[0,1]_['uniformnfw',[3,500]]_1"]

        realizationsdata = init.generate_halos(halo_models2,Nrealizations=1)

        dx,dy = -.01,0
        if profile=='NFW':
            ks = 0.01
            newparams = {'x':data_to_fit[0].x[1]+dx,'y':data_to_fit[0].y[1]+dy,'ks':ks,'rs':0.02}

        realizationsdata[0][0].update(method='lensmodel',**newparams)
        realizationsdata[0] = [realizationsdata[0][0]]

        optimized_lenstronomy,systems = solver.fit_src_plane(macromodel=start, datatofit=data_to_fit[0],
                                                             realizations=realizationsdata, multiplane=multiplane,
                                                             method='lenstronomy', ray_trace=True, sigmas=sigmas,
                                                             identifier=identifier, srcx=srcx, srcy=srcy, grid_rmax=.05,
                                                             res=.001, source_shape='GAUSSIAN', source_size=0.0012)

        optimized_lensmodel,systems = solver.two_step_optimize(macromodel=start, datatofit=data_to_fit[0],
                                                               realizations=realizationsdata, multiplane=multiplane,
                                                               method='lensmodel', ray_trace=True, sigmas=sigmas,
                                                               identifier=identifier, srcx=srcx, srcy=srcy,
                                                               grid_rmax=.05,
                                                               res=.001, source_shape='GAUSSIAN', source_size=0.0012)

        np.testing.assert_almost_equal(optimized_lensmodel[0].m, optimized_lenstronomy[0].m, decimal=2)
        np.testing.assert_almost_equal(optimized_lensmodel[0].x, optimized_lenstronomy[0].x, decimal=4)
        np.testing.assert_almost_equal(optimized_lensmodel[0].y, optimized_lenstronomy[0].y, decimal=4)
        np.testing.assert_almost_equal(optimized_lensmodel[0].x, optimized_lenstronomy[0].x, decimal=4)
        #plt.scatter(optimized_lenstronomy[0].x,optimized_lenstronomy[0].y,color='r',marker='x',alpha=0.5)
        #lt.scatter(optimized_lensmodel[0].x,optimized_lensmodel[0].y,color='k',alpha=0.5)
        #plt.scatter(data_to_fit[0].x,data_to_fit[0].y,color='k',marker='+',s=70,alpha=0.5)
        #plt.scatter(newparams['x'],newparams['y'],color='m',s=50)
        print optimized_lensmodel[0].m
        print optimized_lenstronomy[0].m
        #plt.show()

def test_SIE():
    from MagniPy.MassModels.SIE import SIE

    values = np.loadtxt('../../SIE_def.txt',skiprows=5)
    sie = SIE()

    xdef = values[:,3]
    ydef = values[:,4]
    x = values[:,0]
    y = values[:,1]

    theta_E = 0.7
    ellip = 0.45
    ellip_theta = -10

    center_x,center_y = 0,0
    plt.plot(x,xdef,color='k')

    siexdef = sie.def_angle(x,y,center_x=center_x,center_y=center_y,theta_E=theta_E,q=1-ellip,phi_G=ellip_theta*np.pi*180**-1)[0]
    plt.plot(x,siexdef,color='r')
    plt.show()

def test_Shear():
    from MagniPy.MassModels.ExternalShear import Shear
    s = Shear()

    shear,shear_theta = 0.02,40

    values = np.loadtxt('../../shear_def.txt',skiprows=5)
    xdef = values[:, 3]
    ydef = values[:, 4]
    x = values[:, 0]
    y = values[:, 1]
    plt.plot(x,xdef)
    plt.plot(x,s.def_angle(x,y,shear,shear_theta)[0])

    plt.show()


def test_SIEShear():
    from MagniPy.MassModels.SIE import SIE
    from MagniPy.MassModels.ExternalShear import Shear
    s = Shear()

    shear, shear_theta = 0.02, 40

    values_shear = np.loadtxt('../../shear_def.txt', skiprows=5)

    x = values_shear[:, 0]
    y = values_shear[:, 1]

    values_sie = np.loadtxt('../../SIE_def.txt', skiprows=5)
    sie = SIE()

    values_sieshear = np.loadtxt('../../sieshear_def.txt',skiprows=5)

    x = values_sie[:, 0]
    y = values_sie[:, 1]

    theta_E = 0.7
    ellip = 0.45
    ellip_theta = -10

    center_x, center_y = 0, 0
    plt.plot(x, values_sieshear[:,3], color='k')

    siexdef = sie.def_angle(x, y, center_x=center_x, center_y=center_y, theta_E=theta_E, q=1 - ellip,
                            phi_G=ellip_theta * np.pi * 180 ** -1)[0]
    shearxdef = s.def_angle(x,y,shear,shear_theta)[0]
    plt.plot(x,siexdef+shearxdef,color='r')
    plt.show()

def test_nfws():
    from lenstronomy.LensModel.Profiles.nfw import NFW
    x_loc = np.linspace(0,.1,100)
    y_loc = np.linspace(0,0,100)
    args = {'theta_Rs':0.01,'Rs':0.02,'center_x':0,'center_y':0.03}
    nfw = NFW()
    xdefL, ydefL = nfw.derivatives(x=x_loc, y=y_loc, **args)
    from MagniPy.MassModels.NFW import NFW
    nfw = NFW()
    xplus,yplus = nfw.def_angle(x=x_loc,y=y_loc,**args)
    print ydefL
    print yplus
    np.testing.assert_almost_equal(ydefL, yplus, decimal=5)

test_nfws()
#test_SIEShear()

#test_profiles_nfwT(plot=False)
#test_profiles_nfw(plot=True)
#test_solving_leq_single_plane()