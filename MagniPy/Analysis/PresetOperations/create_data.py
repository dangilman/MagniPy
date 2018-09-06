from pyHalo.pyhalo import pyHalo
from pyHalo.Cosmology.lens_cosmo import LensCosmo
from MagniPy.Solver.solveroutines import *
from MagniPy.Solver.analysis import Analysis
from MagniPy.LensBuild.main_deflector import Deflector
from MagniPy.MassModels.SIE import *
import matplotlib.pyplot as plt
from lenstronomy.Util.param_util import *
import sys
import matplotlib.pyplot as plt
from MagniPy.LensBuild.defaults import *
from MagniPy.Solver.analysis import Analysis
from MagniPy.util import min_img_sep
from MagniPy.paths import *

def set_Rindex(dfile_base,minidx,maxidx):

    for i in range(minidx, maxidx):
        data = read_data(dfile_base + str(i)+'/lensdata.txt')[0]

        plt.scatter(data.x, data.y)
        plt.xlim(-1.4, 1.4)
        plt.ylim(-1.4, 1.4)
        ax = plt.gca()
        ax.set_aspect('equal')
        for i, m in enumerate(data.m):
            plt.annotate(str(np.round(m, 3)), xy=(data.x[i], data.y[i]))
            plt.annotate(str(i), xy=(data.x[i] - 0.2, data.y[i] + 0.2))
        plt.show()
        Rindex = input('R_index: ')
        write_info(str('R_index: ' + str(Rindex)),
                   dfile_base + str(i) + '/Rindex.txt')


    return int(Rindex)

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

def imgFinder(startmod,realization,xs,ys,multiplane,solver,analysis):
    print('finding images....')
    xcrit = None

    while True:

        data_withhalos = solver.solve_lens_equation(macromodel=startmod, realizations=realization, srcx=xs, srcy=ys,
                                                    multiplane=True, method='lenstronomy',
                                                    polar_grid=False, brightimg=True)

        if xcrit is None:
            xcrit, ycrit, xcaus, ycaus = analysis.critical_cruves_caustics(main=startmod,
                                                                           multiplane=multiplane, grid_scale=0.01,
                                                                           compute_window=3)
        if data_withhalos[0].nimg == 4:
            print('done finding images.')
            return data_withhalos, xcaus, ycaus

        xs, ys = guess_source(xcaus, ycaus)


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

    while continue_loop:

        if done_cusp and done_cross and done_fold:
            break

        vdis = np.random.normal(260, 20)
        ellip = np.absolute(np.random.normal(0.15, 0.05))
        ellip_theta = np.absolute(np.random.uniform(-90, 90))
        gamma = np.round(np.random.normal(2.08, 0.05), 2)
        shear = np.random.normal(0.05, 0.01)
        shear_theta = np.random.uniform(-90, 90)

        while True:
            zlens = np.round(np.random.normal(0.5, 0.2), 2)
            zsrc = np.round(zlens + 1 + np.random.normal(0, 0.5), 2)
            if zsrc - zlens > 0.4:
                if zlens > 0.3:
                    break
        while True:
            source_size_kpc = np.round(np.random.normal(src_size_mean, src_size_sigma), 3)
            if source_size_kpc < 0.01:
                continue
            if source_size_kpc > 0.07:
                continue
            else:
                break

        pyhalo = pyHalo(zlens, zsrc)
        c = LensCosmo(zlens, zsrc)
        solver = SolveRoutines(zlens, zsrc)
        analysis = Analysis(zlens, zsrc)
        rein = c.vdis_to_Rein(zlens, zsrc, vdis)

        halo_args = {'mdef_main': 'TNFW', 'mdef_los': 'NFW', 'fsub': fsub, 'log_mlow': log_ml, 'log_mhigh': log_mh,
                     'power_law_index': -1.9, 'log_m_break': logmhm, 'parent_m200': M_halo, 'parent_c': 4, 'mdef': 'TNFW',
                     'break_index': -1.3, 'c_scale': 60, 'c_power': -0.17, 'r_tidal': '0.5Rs', 'break_index': break_index,
                     'c_scale': 60, 'cone_opening_angle': 6 * rein}

        real = pyhalo.render(model_type, halo_args)

        lens_args = {'R_ein': c.vdis_to_Rein(zlens, zsrc, vdis), 'x': 0, 'y': 0, 'ellip': ellip, 'ellip_theta': ellip_theta,
                     'gamma': gamma, 'shear': shear, 'shear_theta': shear_theta}

        start = Deflector(subclass=SIE(), redshift=zlens, **lens_args)

        print('zlens: ', zlens)
        print('zsrc: ', zsrc)
        print('src_size_kpc: ', source_size_kpc)
        print('vdis:', vdis)
        print('R_ein:', rein)
        print('shear, ellip: ', shear, ellip)
        print('nhalos: ', len(real[0].x))

        xcaus, ycaus = None, None
        continue_findimg_loop = True

        while continue_findimg_loop:

            if xcaus is not None:
                xs_init, ys_init = guess_source(xcaus, ycaus)
            else:
                xs_init, ys_init = np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)

            data_withhalos, xcaus, ycaus = imgFinder(start, real, xs_init, ys_init, True, solver, analysis)
            lens_config = identify(data_withhalos[0].x, data_withhalos[0].y, c.vdis_to_Rein(zlens, zsrc, vdis))
            min_sep = min_img_sep(data_withhalos[0].x, data_withhalos[0].y)
            #print('minimum image separation: ', min_sep)

            if min_sep < rein * 0.2:
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
            other_lens_args['source_size_kpc'] = source_size_kpc
            other_lens_args['rmax2d_asec'] = 3 * c.vdis_to_Rein(zlens, zsrc, vdis)
            continue_findimg_loop = False

        mags = []
        ind = np.argsort(data_withhalos[0].m)[::-1]
        save = True
        for i in range(0, 4):
            magnifications, image = analysis.raytrace_images(macromodel=start, xcoord=data_withhalos[0].x[i],
                                                             ycoord=data_withhalos[0].y[i], realizations=real,
                                                             multiplane=multiplane,
                                                             srcx=data_withhalos[0].srcx, srcy=data_withhalos[0].srcy, res=0.01,
                                                             method='lenstronomy', source_shape='GAUSSIAN',
                                                             source_size_kpc=source_size_kpc)


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

            lens_idx = start_idx + n_computed
            n_computed += 1

            dpath = dpath_base + str(lens_idx)
            create_directory(dpath)

            data_withhalos[0].x += np.random.normal(0, 0.003, 4)
            data_withhalos[0].y += np.random.normal(0, 0.003, 4)

            write_data(dpath + '/lensdata.txt', data_withhalos, mode='write')

            system = solver.build_system(main=start, realization=real[0], multiplane=True)
            zlist, lens_list, arg_list = system.lenstronomy_lists()

            with open(dpath + '/redshifts.txt', 'w') as f:
                np.savetxt(f, X=zlist)
            with open(dpath + '/lens_list.txt', 'w') as f:
                f.write(str(lens_list))
            with open(dpath + '/lens_args.txt', 'w') as f:
                f.write(str(arg_list))

            to_write = get_info_string(halo_args, other_lens_args)
            write_info(str(to_write), dpath + '/info.txt')
            #write_info(str('R_index: ' + str(None)), dpath + '/R_index.txt')

model_type = 'composite_powerlaw'
multiplane = True

fsub = 0.01
M_halo = 10 ** 13
logmhm = 0
r_core = '0.5Rs'
src_size_mean = 0.02
src_size_sigma = 0.0001
log_ml, log_mh = 6, 10
break_index = -1.3

nav = IO_directory
dpath_base = nav + '/mock_data/LOS_CDM/lens_'

#ncusp = int(sys.argv[1])
#nfold = int(sys.argv[2])
#ncross = int(sys.argv[3])
#start_idx = int(sys.argv[4])

#run(ncusp,nfold,ncross)

