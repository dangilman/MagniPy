from pyHalo.pyhalo import pyHalo
from pyHalo.Cosmology.cosmology import Cosmology
from scipy.stats import gaussian_kde
from MagniPy.MassModels.SIE import *
import matplotlib.pyplot as plt
from MagniPy.LensBuild.defaults import *
from MagniPy.Solver.analysis import Analysis
from MagniPy.util import min_img_sep, flux_at_edge
from MagniPy.paths import *
from lenstronomy.Util.param_util import ellipticity2phi_q, shear_polar2cartesian
from lenstronomywrapper.LensSystem.macrolensmodel import MacroLensModel
from lenstronomywrapper.LensSystem.quad_lens import QuadLensSystem
from lenstronomywrapper.LensSystem.LensComponents.powerlawshear import PowerLawShear
from lenstronomywrapper.LensSystem.BackgroundSource.quasar import Quasar
from lenstronomywrapper.LensSystem.LensSystemExtensions.solver import iterative_rayshooting
from lenstronomywrapper.LensData.lensed_quasar import LensedQuasar

cosmo = Cosmology()
arcsec = 206265  # arcsec per radian

def get_density():
    quad_imgsep, quad_vdis, _, quad_zlens, quad_zsrc = np.loadtxt(prefix + '/data/quad_info.txt', unpack=True)
    data = np.vstack((quad_imgsep, quad_vdis))
    data = np.vstack((data, quad_zlens))
    data = np.vstack((data, quad_zsrc))
    kde = gaussian_kde(data)
    return kde

def get_strides():
    return np.loadtxt(prefix + '/data/quad_info.txt', unpack=True)

strides_kde_density = get_density()

def matching(want_config, config):
    if want_config == 'cross':
        want_config = 0
    elif want_config == 'fold':
        want_config = 1
    else:
        want_config = 2

    if config == want_config:
        return True
    else:
        return False

def set_Rindex(dfile_base,minidx,maxidx):

    indexes = np.arange(minidx,maxidx)

    for i in indexes:
        print(i)
        data = read_data(dfile_base + str(i)+'/lensdata.txt')[0]

        with open(dfile_base + str(i)+'/info.txt') as f:
            info = eval(f.read())
            print('maybe a '+info['config'])

        plt.scatter(data.x, data.y)
        plt.xlim(-1.7, 1.7)
        plt.ylim(-1.7, 1.7)
        ax = plt.gca()
        ax.set_aspect('equal')
        for k, m in enumerate(data.m):
            plt.annotate(str(np.round(m, 3)), xy=(data.x[k], data.y[k]))
            plt.annotate(str(k), xy=(data.x[k] - 0.2, data.y[k] + 0.2))

        plt.show()
        try:
            config = input('config: ')
        except:
            config = info['config']
        Rindex = input('R_index: ')

        write_info(str(Rindex)+'\n'+config,
                   dfile_base + str(i) + '/Rindex.txt')


def guess_source(xcaus,ycaus):

    xran = np.max(xcaus) - np.min(xcaus)
    yran = np.max(ycaus) - np.min(ycaus)

    rmax = np.sqrt(xran * yran)

    return np.random.uniform(-rmax*0.5,rmax*0.5),\
           np.random.uniform(-rmax*0.5,rmax*0.5)

def write_info(info_string, path_2_write):
    with open(path_2_write, 'w') as f:
        f.write(info_string)


def get_info_string(halo_args, lens_args):
    x = {}

    for pname in halo_args.keys():
        x.update({pname: halo_args[pname]})

    for pname in lens_args.keys():
        x.update({pname: lens_args[pname]})

    return x

def imgFinder(startmod,realization,xs,ys,multiplane,solver,analysis,source_size_kpc,print_things = False):
    if print_things: print('finding images....')
    xcrit = None

    while True:

        try:
            data_withhalos = solver.solve_lens_equation(macromodel=startmod, realizations=realization, srcx=xs, srcy=ys,
                                                        multiplane=True, method='lenstronomy', source_size_kpc=source_size_kpc,
                                                        polar_grid=False, brightimg=True, res=0.001, LOS_mass_sheet_back = 6,
                                                        LOS_mass_sheet_front = 6)

            if xcrit is None:
                xcrit, ycrit, xcaus, ycaus = analysis.critical_cruves_caustics(main=startmod,
                                                                               multiplane=multiplane, grid_scale=0.01,
                                                                               compute_window=3)
            if data_withhalos[0].nimg == 4:

                if print_things: print('done finding images.')
                return data_withhalos, xcaus, ycaus, xs, ys
            try:
                xs, ys = guess_source(xcaus, ycaus)
            except:
                xs, ys = np.random.uniform(-0.15, 0.15), np.random.uniform(-0.15, 0.15)
        except:
            xs, ys = np.random.uniform(-0.15, 0.15), np.random.uniform(-0.15, 0.15)


def sample_from_strides(nsamples):

    def cuts(imgsep, vdis, zlens, zsrc, imgsep_min=1,
             vdis_min=230, vdis_max = 310, zlens_min=0.2, zlens_max=0.8, zsrc_max=3.5):

        if imgsep > imgsep_min and vdis > vdis_min and vdis < vdis_max and zlens > zlens_min and zsrc < zsrc_max and zlens < zlens_max and zsrc > zlens + 0.2:
            return imgsep, vdis, zlens, zsrc
        else:
            return None, None, None, None

    def resample(KDE):
        samples = KDE.resample(1)
        imgsep = np.round(samples[0][0], 2)
        vdis = int(samples[1][0])
        zlens = np.round(samples[2][0], 2)
        zsrc = np.round(samples[3][0],2)

        return cuts(imgsep, vdis, zlens, zsrc)

    image_sep, v_dis, zd, zsrc = [], [], [], []

    while len(image_sep) < nsamples:
        kde = strides_kde_density
        values = resample(kde)
        if values[0] is not None:
            image_sep.append(values[0])
            v_dis.append(values[1])
            zd.append(values[2])
            zsrc.append(values[3])
    image_sep = np.array(image_sep)
    v_dis = np.array(v_dis)
    zd = np.array(zd)
    zsrc = np.array(zsrc)

    if nsamples == 1:
        image_sep, v_dis, zd, zsrc = image_sep[0], v_dis[0], zd[0], zsrc[0]

    return 0.5 * image_sep, v_dis, zd, zsrc

def draw_vdis(mean=260, sigma=15):
    return np.random.normal(mean, sigma)

def draw_ellip(mean = 0.2, sigma = 0.1, low = 0, high = 0.5):

    while True:
        ellip = np.random.normal(mean, sigma)
        if ellip > low and ellip < high:
            break
    return ellip

def draw_ellip_PA(low = -90, high = 90):

    return np.random.uniform(low, high)

def draw_shear(mean = 0.06, sigma = 0.02, low = 0.035, high = 0.1):

    while True:
        shear = np.random.normal(mean, sigma)
        if shear > low and shear < high:
            break
    return shear

def draw_shear_PA(low = -90, high = 90):

    return np.random.uniform(low, high)

def draw_shear_PA_correlated(mean, sigma):

    pa = np.random.normal(mean, sigma)

    if pa <= -360:
        pa = pa + 360
    elif pa >= 360:
        pa = pa - 360

    return pa

def run(Ntotal_cusp, Ntotal_fold, Ntotal_cross, start_idx):

    continue_loop = True
    n_computed = 0
    done_cusp, done_fold, done_cross = False, False, False
    n_cusp, n_fold, n_cross = 0, 0, 0

    if Ntotal_cusp == 0:
        done_cusp = True
    if Ntotal_fold == 0:
        done_fold = True
    if Ntotal_cross == 0:
        done_cross = True

    lens_idx = int(start_idx) + n_computed
    dpath = dpath_base + str(lens_idx)
   
    if os.path.exists(dpath + '/lensdata.txt'):
        return

    while continue_loop:

        if done_cusp and done_cross and done_fold:
            break

        while True:
            rein, vdis, zlens, zsrc = sample_from_strides(1)
            if rein <= rein_max and rein >= rein_min:
                if zlens <= z_lens_max:
                    if zsrc <= z_src_max:
                        break

        pyhalo = pyHalo(zlens, zsrc)

        print('rein, zlens, zsrc: ', str(rein)+' '+str(zlens) + ' '+str(zsrc))

        ellip = draw_ellip()
        ellip_theta = draw_ellip_PA()
        shear = draw_shear()
        
        shear_theta = draw_shear_PA_correlated(ellip_theta, sigma=40)

        halo_args = {'mdef_main': mass_def, 'mdef_los': mass_def, 'sigma_sub': sigma_sub, 'log_mlow': log_ml, 'log_mhigh': log_mh,
                     'power_law_index': -1.9, 'parent_m200': M_halo, 'r_tidal': r_tidal,
                     'R_ein_main': rein, 'SIDMcross': SIDM_cross, 'vpower': vpower}

        realization = pyhalo.render(model_type, halo_args)[0]

        e1, e2 = phi_q2_ellipticity(shear_theta*np.pi/180, 1-ellip)
        gamma1, gamma2 = shear_polar2cartesian(shear_theta*np.pi/180, shear)
        kwargs_lens = [{'theta_E': rein, 'center_x': 0, 'center_y': 0,
                       'e1': e1, 'e2': e2, 'gamma': gamma},
                       {'gamma1': gamma1, 'gamma2': gamma2}]
        macromodel = MacroLensModel([PowerLawShear(zlens, kwargs_lens)])
        source_args = {'center_x': None, 'center_y': None, 'source_fwhm_pc': source_size_fwhm_pc}
        quasar = Quasar(source_args)
        system = QuadLensSystem(macromodel, zsrc, quasar, realization, pyhalo._cosmology)

        print('zlens: ', zlens)
        print('zsrc: ', zsrc)
        print('src_size_pc: ', source_size_fwhm_pc)
        print('R_ein:', rein)
        print('shear, ellip: ', shear, ellip)
        print('nhalos: ', len(realization.halos))

        continue_findimg_loop = True

        while continue_findimg_loop:

            src_r = np.sqrt(np.random.uniform(0.015**2,0.1 ** 2))
            src_phi = np.random.uniform(-90, 90)*180/np.pi
            srcx, srcy = src_r*np.cos(src_phi), src_r*np.sin(src_phi)
            print(srcx, srcy)
            system.update_source_centroid(srcx, srcy)

            lens_model_smooth, kwargs_lens_smooth = system.get_lensmodel(False)
            lensModel, kwargs_lens = system.get_lensmodel()

            x_guess, y_guess = system.solve_lens_equation(lens_model_smooth, kwargs_lens_smooth)
            if len(x_guess) != 4:
                continue

            min_sep = min_img_sep(x_guess, y_guess)
            lens_config = identify(x_guess, y_guess, rein)

            if min_sep < 0.2:
                continue
            if lens_config == 0:
                config = 'cross'
                if done_cross:
                    continue
            elif lens_config == 1:
                config = 'fold'
                if done_fold:
                    continue
            else:
                config = 'cusp'
                if done_cusp:
                    continue

            x_image, y_image = iterative_rayshooting(srcx, srcy,
                                                     x_guess, y_guess, lensModel, kwargs_lens)
            lens_config = identify(x_image, y_image, rein)
            min_sep = min_img_sep(x_image, y_image)

            if min_sep < 0.2:
                continue

            if lens_config == 0:
                config = 'cross'
                if done_cross:
                    continue
            elif lens_config == 1:
                config = 'fold'
                if done_fold:
                    continue
            else:
                config = 'cusp'
                if done_cusp:
                    continue

            print(config)
            other_lens_args = {}
            other_lens_args['zlens'] = zlens
            other_lens_args['zsrc'] = zsrc
            other_lens_args['gamma'] = gamma
            other_lens_args['config'] = config
            other_lens_args['source_fwhm_pc'] = source_size_fwhm_pc
            other_lens_args['rmax2d_asec'] = 3*rein
            continue_findimg_loop = False

        if lens_config == 0:
            n_cross += 1
            if n_cross == Ntotal_cross:
                done_cross = True
        elif lens_config ==1:
            n_fold += 1
            if n_fold == Ntotal_fold:
                done_fold = True
        else:
            n_cusp += 1
            if n_cusp == Ntotal_cusp:
                done_cusp = True

        n_computed += 1

        create_directory(dpath)

        x_image += np.random.normal(0, 0.005, 4)
        y_image += np.random.normal(0, 0.005, 4)

        magnifications = system.quasar_magnification(x_image, y_image, lensModel, kwargs_lens)
        data_with_halos = LensedQuasar(x_image, y_image, magnifications)

        write_data(dpath + '/lensdata.txt', data_with_halos, mode='write')

        to_write = get_info_string(halo_args, other_lens_args)
        write_info(str(to_write), dpath + '/info.txt')
        #write_info(str('R_index: ' + str(None)), dpath + '/R_index.txt')

import sys
if True:
    model_type = 'composite_powerlaw'
    multiplane = True

    sigma_sub = 0.02
    M_halo = 10 ** 13
    logmhm = 0
    r_tidal = '0.5Rs'
    source_size_fwhm_pc = 5.
    log_ml, log_mh = 7, 10
    gamma = 2.05

    SIDM_cross = 9
    vpower = 0.75

    z_src_max = 2.5
    z_lens_max = 0.6
    rein_max = 1.4
    rein_min = 0.6
    nav = prefix
    mass_def = 'SIDM_TNFW'

    dpath_base = nav + '/mock_data/SIDM_cross9_vpower75/lens_'
    run(1, 0, 0, 1)

