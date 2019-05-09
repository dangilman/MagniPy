from pyHalo.pyhalo import pyHalo
from pyHalo.Cosmology.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from scipy.stats import gaussian_kde
from MagniPy.Solver.solveroutines import *
from MagniPy.MassModels.SIE import *
import matplotlib.pyplot as plt
from MagniPy.LensBuild.defaults import *
from MagniPy.Solver.analysis import Analysis
from MagniPy.util import min_img_sep
from MagniPy.paths import *
from scipy.optimize import minimize

cosmo = Cosmology()
arcsec = 206265  # arcsec per radian

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

def flux_at_edge(image):

    maxbright = np.max(image)
    edgebright = [image[0,:],image[-1,:],image[:,0],image[:,-1]]

    for edge in edgebright:
        if any(edge > maxbright * 0.01):
            return True
    else:
        return False

def imgFinder(startmod,realization,xs,ys,multiplane,solver,analysis,source_size_kpc,print_things = False):
    if print_things: print('finding images....')
    xcrit = None

    while True:

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
            xs, ys = np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)


def sample_from_strides(nsamples):

    def get_density():
        quad_imgsep, quad_vdis, _, quad_zlens, quad_zsrc = np.loadtxt(prefix + '/data/quad_info.txt', unpack=True)
        data = np.vstack((quad_imgsep, quad_vdis))
        data = np.vstack((data, quad_zlens))
        data = np.vstack((data, quad_zsrc))
        kde = gaussian_kde(data)
        return kde

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
        kde = get_density()
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
    l = LensCosmo(0.5, 4)

    if os.path.exists(dpath + '/lensdata.txt'):
        return

    while continue_loop:

        if done_cusp and done_cross and done_fold:
            break
        zsrc = 500
        zlens = 500
        rein = 400

        while True:
            rein, vdis, zlens, zsrc = sample_from_strides(1)
            if rein <= rein_max:
                if zlens <= z_lens_max:
                    if zsrc <= z_src_max:
                        break

        print('vdis, rein, zlens, zsrc: ', str(vdis) + ' '+str(rein)+' '+str(zlens) + ' '+str(zsrc))

        ellip = draw_ellip()
        ellip_theta = draw_ellip_PA()
        shear = draw_shear()
        #shear_theta = draw_shear_PA()
        shear_theta = draw_shear_PA_correlated(ellip_theta, sigma = 50)
        #gamma = np.round(np.random.normal(2.08, 0.05), 2)
        gamma = 2.08
        #while True:
        #    source_size_kpc = np.round(np.random.normal(src_size_mean, src_size_sigma), 3)
        #    if source_size_kpc < 0.01:
        #        continue
        #    if source_size_kpc > 0.06:
        #        continue
        #    else:
        #        break

        pyhalo = pyHalo(zlens, zsrc)
        c = LensCosmo(zlens, zsrc)
        solver = SolveRoutines(zlens, zsrc)
        analysis = Analysis(zlens, zsrc)
        #rein = c.vdis_to_Rein(zlens, zsrc, vdis)

        halo_args = {'mdef_main': mass_def, 'mdef_los': mass_def, 'a0_area': a0_area, 'log_mlow': log_ml, 'log_mhigh': log_mh,
                     'power_law_index': -1.9, 'log_m_break': logmhm, 'parent_m200': M_halo, 'parent_c': 4,
                     'c_scale': 60, 'c_power': -0.17, 'r_tidal': r_tidal, 'break_index': break_index,
                     'R_ein_main': rein, 'core_ratio': core_ratio}

        real = pyhalo.render(model_type, halo_args)

        lens_args = {'theta_E': c.vdis_to_Rein(zlens, zsrc, vdis), 'center_x': 0, 'center_y': 0, 'ellip': ellip, 'ellip_theta': ellip_theta,
                     'gamma': gamma, 'shear': shear, 'shear_theta': shear_theta}

        start = Deflector(subclass=SIE(), redshift=zlens, **lens_args)

        print('zlens: ', zlens)
        print('zsrc: ', zsrc)
        print('src_size_kpc: ', src_size_mean)
        print('vdis:', vdis)
        print('R_ein:', rein)
        print('shear, ellip: ', shear, ellip)
        print('nhalos: ', len(real[0].x))

        xcaus, ycaus = None, None
        continue_findimg_loop = True

        while continue_findimg_loop:
            
            if xcaus is not None:

                try:
                    xs_init, ys_init = guess_source(xcaus, ycaus)
                except:
                    xs_init, ys_init = 0.1, -0.005
            else:
                xs_init, ys_init = 0.06, -0.01

            data_withhalos, xcaus, ycaus, _, _ = imgFinder(start, real, xs_init, ys_init, True, solver, analysis, src_size_mean)
            lens_config = identify(data_withhalos[0].x, data_withhalos[0].y, c.vdis_to_Rein(zlens, zsrc, vdis))
            min_sep = min_img_sep(data_withhalos[0].x, data_withhalos[0].y)
            #print('minimum image separation: ', min_sep)

            if min_sep < rein * 0.1:
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
            other_lens_args['source_size_kpc'] = src_size_mean
            other_lens_args['rmax2d_asec'] = 3*rein
            continue_findimg_loop = False

        save = True
        for i in range(0, 4):
            magnifications, image = analysis.raytrace_images(macromodel=start, xcoord=data_withhalos[0].x[i],
                                                             ycoord=data_withhalos[0].y[i], realizations=real,
                                                             multiplane=multiplane,
                                                             srcx=data_withhalos[0].srcx, srcy=data_withhalos[0].srcy, res=0.01,
                                                             method='lenstronomy', source_shape='GAUSSIAN',
                                                             source_size_kpc=src_size_mean)


            if flux_at_edge(image):
                print('images are blended... continuing loop.')

                save = False
                break

        if save is False:
            continue

        else:

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

            data_withhalos[0].x += np.random.normal(0, 0.003, 4)
            data_withhalos[0].y += np.random.normal(0, 0.003, 4)

            write_data(dpath + '/lensdata.txt', data_withhalos, mode='write')

            system = solver.build_system(main=start, realization=real[0], multiplane=True)
            zlist, lens_list, arg_list, supplement = system.lenstronomy_lists()

            with open(dpath + '/redshifts.txt', 'w') as f:
                np.savetxt(f, X=zlist)
            with open(dpath + '/lens_list.txt', 'w') as f:
                f.write(str(lens_list))
            with open(dpath + '/lens_args.txt', 'w') as f:
                f.write(str(arg_list))

            to_write = get_info_string(halo_args, other_lens_args)
            write_info(str(to_write), dpath + '/info.txt')
            #write_info(str('R_index: ' + str(None)), dpath + '/R_index.txt')

if True:
    model_type = 'composite_powerlaw'
    multiplane = True

    a0_area = 0.015
    M_halo = 10 ** 13
    logmhm = 0
    r_tidal = '0.5Rs'
    src_size_mean = 0.02
    src_size_sigma = 0.0001
    log_ml, log_mh = 6, 10
    break_index = -1.3
    core_ratio = 0.01
    z_src_max = 2.5
    z_lens_max = 0.6
    rein_max = 1.3
    nav = prefix
    mass_def = 'TNFW'


    dpath_base = nav + '/mock_data/CDMsrc20/lens_'

    #run(0, 0, 1, 1)
    #run(0, 1, 0, 1)

    #dpath_base = nav + '/data/mock_data/replace_lens/lens_1'
    #run(0,1,0,1)
    #import sys
    #start_idx=int(sys.argv[1])

    #cusps = np.arange(1,60,3)
    #folds = cusps + 1
    #crosses = cusps + 2
    #start_idx = 7
    #if start_idx in cusps:
    #    print('cusp')
    #    run(1, 0, 0, start_idx)
    #elif start_idx in folds:
    #    print('fold')
    #    run(0, 1, 0, start_idx)
    #else:
    #    print('cross')
    #    run(0, 0, 1, start_idx)

#cusps = np.arange(1,60,3)
#folds = cusps + 1
#crosses = cusps + 2
#start_idx = 13
#if start_idx in cusps:
#    print('cusp')
#    run(1, 0, 0, start_idx)
#elif start_idx in folds:
#    print('fold')
#    run(0, 1, 0, start_idx)
#else:
#    print('cross')
#    run(0, 0, 1, start_idx)

#run(1,0,0, start_idx=43)
#if start_idx in folds:
#run(0,1,0, start_idx=9)
#if start_idx in crosses:
#    run(0,0,1, start_idx=sys.argv[1])
#elif start_idx in folds:
#    run(0, 1, 0, start_idx=sys.argv[1])
#else:
#    run(1, 0, 0, start_idx=sys.argv[1])