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

    x, y = np.linspace(-0.5, .5, 200), np.linspace(-0.5, 0.5, 200)

    xx, yy = np.meshgrid(x, y)

    from MagniPy.MassModels.NFW import NFW
    from lenstronomy.LensModel.Profiles.nfw import NFW as NFW_L
    from MagniPy.MassModels.nfwT_temp import NFWt
    from MagniPy.MassModels.TNFW import TNFW
    import matplotlib.pyplot as plt

    rt = 1000

    if rt == 1000:
        fname = 'nfwdef_rt1000.txt'
    elif rt == .4:
        fname = 'nfwdef_rt.4.txt'

    nfw = NFW()
    nfwL = NFW_L()
    nfwT = TNFW()
    nfwTL = NFWt()

    # nfw_params,nfw_params_L = nfw.params(x=0,y=0,mass=mass,mhm=mhm)
    # nfwt_params,nfwt_params_L = nfwT.params(x=0,y=0,mass=mass,mhm=mhm,trunc=rt)

    ks, rs = 0.1, 0.1
    theta_rs = 4 * ks * rs * (1 + np.log(.5))

    xdef, ydef = nfw.def_angle(xx, yy, x=0, y=0, ks=ks, rs=rs)
    xdef_t, ydef_t = nfwT.def_angle(xx, yy, x=0, y=0, ks=ks, rs=rs, rt=rt)
    xdef_L, ydef_L = nfwL.derivatives(x=xx, y=xx, Rs=rs, theta_Rs=theta_rs)
    xdef_Lt, ydef_Lt = nfwTL.derivatives(x=xx, y=yy, Rs=rs, theta_Rs=theta_rs, t=rt)

    gravlens_xdef = np.loadtxt(fname)
    # xdef,ydef = nfw.def_angle(xx,yy,0,0,rs,ks)
    # xdef_L,ydef_L = nfwL.derivatives(x=xx,y=yy,**nfw_params_L)

    if plot:
        colors=['r','k','b','g']
        labels = ['my nfw','lenstronomy nfw','my nfw rt=1000rs','lensmodel']
        angles = [xdef,xdef_L,xdef_t,gravlens_xdef]
        for i,deflection in enumerate(angles[:-1]):
            plt.plot(x[100:],deflection[100,100:],color=colors[i],label=labels[i],alpha=0.5)
            plt.scatter(x[100:], deflection[100, 100:], color=colors[i], label=labels[i], alpha=0.5)
        plt.plot(x[100:],gravlens_xdef,color=colors[-1],label=labels[-1],alpha=0.5)
        plt.scatter(x[100:], gravlens_xdef, color=colors[-1], label=labels[-1], alpha=0.5)
        plt.legend()
        plt.show()

    else:
        if rt == 1000:
            np.testing.assert_almost_equal(xdef[100, 100:], xdef_t[100, 100:], decimal=8)
            np.testing.assert_almost_equal(xdef[100, 100:], xdef_Lt[100, 100:], decimal=8)
            # np.testing.assert_almost_equal(xdef[100, 100:], xdef_L[100, 100:], decimal=8)
        if rt == .4:
            np.testing.assert_almost_equal(xdef_t[100, 100:], xdef_Lt[100, 100:], decimal=8)
            np.testing.assert_almost_equal(xdef_t[100, 110:], gravlens_xdef[10:], decimal=4)


def test_profiles_nfwT(plot=True):

    rt = 0.4

    x, y = np.linspace(-0.5, .5, 200), np.linspace(-0.5, 0.5, 200)
    xx, yy = np.meshgrid(x, y)

    from MagniPy.MassModels.NFW import NFW
    from lenstronomy.LensModel.Profiles.nfw import NFW as NFW_L
    from MagniPy.MassModels.nfwT_temp import NFWt
    from MagniPy.MassModels.TNFW import TNFW
    import matplotlib.pyplot as plt

    if rt == 1000:
        fname = 'nfwdef_rt1000.txt'
    elif rt == .4:
        fname = 'nfwdef_rt.4.txt'

    nfw = NFW()
    nfwL = NFW_L()
    nfwT = TNFW()
    nfwTL = NFWt()

    # nfw_params,nfw_params_L = nfw.params(x=0,y=0,mass=mass,mhm=mhm)
    # nfwt_params,nfwt_params_L = nfwT.params(x=0,y=0,mass=mass,mhm=mhm,trunc=rt)

    ks, rs = 0.1, 0.1
    theta_rs = 4 * ks * rs * (1 + np.log(.5))

    xdef, ydef = nfw.def_angle(xx, yy, x=0, y=0, ks=ks, rs=rs)
    xdef_t, ydef_t = nfwT.def_angle(xx, yy, x=0, y=0, ks=ks, rs=rs, rt=rt)
    xdef_L, ydef_L = nfwL.derivatives(x=xx, y=xx, Rs=rs, theta_Rs=theta_rs)
    xdef_Lt, ydef_Lt = nfwTL.derivatives(x=xx, y=yy, Rs=rs, theta_Rs=theta_rs, t=rt)
    gravlens_xdef = np.loadtxt(fname)
    # xdef,ydef = nfw.def_angle(xx,yy,0,0,rs,ks)
    # xdef_L,ydef_L = nfwL.derivatives(x=xx,y=yy,**nfw_params_L)

    if plot:
        colors=['r','k','b']
        labels = ['my nfw rt = 4*rs','lenstronomy nfw rt=4*rs','lensmodel']
        angles = [xdef_t,xdef_Lt,gravlens_xdef]
        for i,deflection in enumerate(angles[:-1]):
            plt.plot(x[100:],deflection[100,100:],color=colors[i],label=labels[i],alpha=0.5)
            plt.scatter(x[100:], deflection[100, 100:], color=colors[i], label=labels[i], alpha=0.5)
        plt.plot(x[100:],gravlens_xdef,color=colors[-1],label=labels[-1],alpha=0.5)
        plt.scatter(x[100:], gravlens_xdef, color=colors[-1], label=labels[-1], alpha=0.5)
        plt.legend()
        plt.show()

    else:
        if rt == 1000:
            np.testing.assert_almost_equal(xdef[100, 100:], xdef_t[100, 100:], decimal=8)
            np.testing.assert_almost_equal(xdef[100, 100:], xdef_L[100, 100:], decimal=8)
        if rt == .4:
            np.testing.assert_almost_equal(xdef_t[100, 100:], xdef_Lt[100, 100:], decimal=8)
            np.testing.assert_almost_equal(xdef_t[100, 110:], gravlens_xdef[10:], decimal=4)

test_profiles_nfw()