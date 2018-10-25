from MagniPy.util import *
from MagniPy.Solver.solveroutines import SolveRoutines
from MagniPy.LensBuild.defaults import *
from MagniPy.paths import *
from MagniPy.lensdata import Data
from pyHalo.pyhalo import pyHalo
from copy import copy
from time import time

def straight_line(dcurrent, dstart, dend, ystart, yend):
    slope = (yend - ystart) / (dend - dstart)
    intersect = ystart
    dx = dcurrent - dstart
    return dx * slope + intersect

def run(outidx, identifier, zlens, zsrc, log_m_break):

    pyhalo = pyHalo(zlens, zsrc)
    lens_params = {'R_ein': 1.2, 'x': 0, 'y': 0, 'ellip': 0.22, 'ellip_theta': 23, 'shear': 0.06, 'shear_theta': -40,
                   'gamma': 2}
    start = Deflector(redshift=zlens, subclass=SIE(), varyflags=['1', '1', '1', '1', '1', '1', '1', '0', '0', '0'],
                      **lens_params)

    solver = SolveRoutines(zlens, zsrc)
    multiplane = True
    init_args = {'mdef_main': 'TNFW', 'mdef_los': 'TNFW', 'log_mlow': 6, 'log_mhigh': 10, 'power_law_index': -1.9,
                 'parent_m200': 10 ** 13, 'parent_c': 3, 'mdef': 'TNFW', 'break_index': -1.3, 'c_scale': 60,
                 'c_power': -0.17, 'r_tidal': '0.4Rs', 'break_index': -1.3, 'c_scale': 60, 'c_power': -0.17,
                 'cone_opening_angle': 6}
    model_args_CDM = init_args
    model_args_CDM.update({'fsub': 0.01, 'log_m_break': log_m_break})
    halos = pyhalo.render('composite_powerlaw', model_args_CDM)

    dtofit = solver.solve_lens_equation(macromodel=start, realizations=None, multiplane=multiplane, srcx=-0.085,
                                        srcy=0.13,
                                        polar_grid=False, source_size_kpc=0.05, brightimg=True)

    opt_full, mod_full, info = solver.hierarchical_optimization(macromodel=start, datatofit=dtofit[0],
                                                                realizations=halos,
                                                                multiplane=True, n_particles=20, n_iterations=350,
                                                                verbose=True, re_optimize=True, restart=1,
                                                                particle_swarm=True,
                                                                pso_convergence_mean=20000,
                                                                pso_compute_magnification=500, source_size_kpc=0.05,
                                                                simplex_n_iter=400, grid_res=0.0025,
                                                                LOS_mass_sheet=7.7)

    relative_paths(info, zlens, identifier, outidx)

def relative_paths(outputs,zlens,identifier,outidx):

        x, y, Tzs, zshifts = outputs[0], outputs[1], outputs[2], outputs[3]

        paths_x_background, paths_x_foreground, paths_y_background, paths_y_foreground = [], [], [], []

        paths_x_coords = []

        for image_index in [0, 1, 2, 3]:

            img_x_back, img_y_back, img_x_front, img_y_front = [], [], [], []

            for sequence in range(0, len(zshifts)):
                if sequence != len(zshifts) - 1:
                    continue
                start_index = np.where(np.array(zshifts[sequence]) == zlens)[0][0]

                xaxis_coords = []
                yaxis_coords1 = []
                yaxis_coords2 = []
                for p in range(0, len(zshifts[sequence]) - start_index):

                    distance = Tzs[sequence][start_index + p]

                    xaxis_coords.append(distance)
                    dmin, dmax = Tzs[sequence][start_index], Tzs[sequence][-1]
                    y1 = x[sequence][start_index + p][image_index]
                    y2 = y[sequence][start_index + p][image_index]

                    ystart1, yend1 = x[sequence][start_index][image_index], x[sequence][-1][image_index]
                    ystart2, yend2 = y[sequence][start_index][image_index], y[sequence][-1][image_index]

                    ystraight1 = straight_line(distance, dmin, dmax, ystart1, yend1)
                    ystraight2 = straight_line(distance, dmin, dmax, ystart2, yend2)

                    # yaxis_coords1.append((y1 - ystraight1))
                    yaxis_coords1.append((y1 - ystraight1) / distance)
                    yaxis_coords2.append((y2 - ystraight2) / distance)

                img_x_back = np.array(yaxis_coords1)
                img_y_back = np.array(yaxis_coords2)
                xcoords = np.array(xaxis_coords)

            paths_x_coords.append(xcoords)
            paths_x_background.append(np.array(img_x_back))
            paths_y_background.append(np.array(img_y_back))

        for i in range(0, len(paths_x_background)):
            with open(fluxratio_data_path+'paths_folder/'+identifier + '_paths_image'+str(i+1)+'_'+str(outidx)+'.txt', 'w') as f:
                for j, xi in enumerate(paths_x_background[i]):
                    #columns are for each image
                    f.write(str(paths_x_coords[i][j])+' '+str(xi / paths_x_coords[i][j])+' '+str(paths_y_background[i][j] / paths_x_coords[i][j]) +'\n')

#for i in range(0,50):
#    run(i+1, 'LOS_msheet8_TNFW', 0.8, 2.5)



