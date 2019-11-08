from time import time

from MagniPy import paths
from MagniPy.Solver.GravlensWrap._call import *
from MagniPy.LensBuild.lens_system import LensSystem
from MagniPy.Solver.GravlensWrap.generate_input import *
from MagniPy.Solver.GravlensWrap.gravlens_to_kwargs import gravlens_to_kwargs
from MagniPy.Solver.LenstronomyWrap.lenstronomy_wrap import *
from MagniPy.Solver.RayTrace import raytrace
from MagniPy.lensdata import Data
import os
from pyHalo.Cosmology.cosmology import Cosmology

class Magnipy:
    """
    This class is used to specify a lens system and solve the lens equation or optimize a lens model
    """

    def __init__(self, zlens, zsrc, clean_up=True, temp_folder=None):

        self.zmain = zlens
        self.zsrc = zsrc
        self.cosmo = Cosmology()
        self.clean_up = clean_up

        self.paths = paths

        if temp_folder is None:
            self.temp_folder = 'temp/'
        else:
            self.temp_folder = temp_folder + '/'

        self.paths.gravlens_input_path_dump = self.paths.gravlens_input_path + self.temp_folder

    def lenstronomy_build(self):

        return LenstronomyWrap(cosmo=self.cosmo.astropy, z_source=self.zsrc)

    def print_system(self, system_index, component_index=None):

        system = self.lens_system[system_index]
        if component_index is not None:
            system.lens_components[component_index].print_args()
        else:
            for component in system.lens_components:
                component.print_args()

    def update_system(self, lens_system, newkwargs, method):

        if lens_system.lens_components[0].other_args['name'] == 'SERSIC_NFW':
            raise Exception('not yet implemented')
        else:

            if method == 'lenstronomy':

                lens_system.lens_components[0].update(method=method, is_shear=False, **newkwargs[0])

                lens_system.lens_components[0].update(method=method, is_shear=True, **newkwargs[1])

            elif method == 'lensmodel':

                for val in [True,False]:
                    lens_system.lens_components[0].update(method=method, is_shear=val, **newkwargs)

            return lens_system

    def build_system(self, main=None, realization=None, multiplane=None, LOS_mass_sheet_front=7.7,
                     LOS_mass_sheet_back = 8, satellites = None):

        assert multiplane is not None

        newsystem = LensSystem(main, satellites = satellites, realization = realization, multiplane=multiplane,
                               LOS_mass_sheet_front=LOS_mass_sheet_front,
                               LOS_mass_sheet_back = LOS_mass_sheet_back)

        return newsystem

    def _optimize_4imgs_lenstronomy(self, lens_systems, data2fit=None, tol_source=None, tol_mag=None, tol_centroid=None,
                                    centroid_0=None, n_particles=50, n_iterations=400,res=None, source_shape='GAUSSIAN', source_size_kpc=None,
                                    polar_grid=None, optimizer_routine=str, verbose=bool, re_optimize=False,
                                    particle_swarm = True, restart = 1, constrain_params=None, return_ray_path = False,
                                    pso_convergence_mean=None, pso_compute_magnification=None, tol_simplex_params=None, tol_simplex_func=None,
                                    simplex_n_iter=None, optimizer_kwargs = {}, finite_source_magnification = True, chi2_mode = 'source',
                                    tol_image = None, adaptive_grid=None, grid_rmax_scale=1):

        data, opt_sys = [], []

        lenstronomyWrap = LenstronomyWrap(cosmo=self.cosmo.astropy, z_source=self.zsrc)
        assert len(lens_systems) == 1
        for i, system in enumerate(lens_systems):

            compute_fluxes = True
            kwargs_lens, [xsrc,ysrc], [x_opt,y_opt], optimizer = lenstronomyWrap.run_optimize(system,self.zsrc,data2fit.x,
                            data2fit.y,tol_source,data2fit.m,tol_mag,tol_centroid,centroid_0,optimizer_routine,self.zmain,
                            n_particles,n_iterations,verbose,restart,re_optimize,particle_swarm,constrain_params,
                            pso_convergence_mean=pso_convergence_mean,pso_compute_magnification=pso_compute_magnification,
                             tol_simplex_params=tol_simplex_params,tol_simplex_func=tol_simplex_func,
                              simplex_n_iter=simplex_n_iter, optimizer_kwargs = optimizer_kwargs,
                               chi2_mode = chi2_mode, tol_image = tol_image)

            if len(x_opt) != len(data2fit.x) or len(y_opt) != len(data2fit.y):
                print('Warning: optimization found the wrong number of images')

                x_opt, y_opt = lenstronomyWrap.solve_leq(xsrc, ysrc, optimizer.lensModel, kwargs_lens, brightimg=True,
                                                         precision_lim=10**-12, nitermax=20)

                if len(x_opt) != len(data2fit.x) or len(y_opt) != len(data2fit.y):
                    x_opt = data2fit.x
                    y_opt = data2fit.y
                    compute_fluxes = False

            lensModel = optimizer.lensModel

            if system.multiplane:
                optimizer_kwargs = {}

                if hasattr(optimizer._optimizer, '_mags'):
                    optimizer_kwargs.update({'magnification_pointsrc': optimizer._optimizer._mags})
                else:
                    mags = lensModel.magnification(x_opt, y_opt, kwargs_lens)
                    optimizer_kwargs.update({'magnification_pointsrc': mags})

                optimizer_kwargs.update({'precomputed_rays': optimizer.lensModel._foreground._rays})

                if return_ray_path:

                    xpath, ypath = [], []
                    for xi, yi in zip(x_opt, y_opt):
                        _x, _y, redshifts, Tzlist = lensModel._full_lensmodel.\
                        lens_model.ray_shooting_partial_steps(0, 0, xi,
                                       yi, 0, self.zsrc, kwargs_lens)

                        xpath.append(_x)
                        ypath.append(_y)
                    nplanes = len(xpath[0])
                    x_path, y_path = [], []
                    for ni in range(0, nplanes):

                        arrx = np.array([xpath[0][ni], xpath[1][ni], xpath[2][ni], xpath[3][ni]])
                        arry = np.array([ypath[0][ni], ypath[1][ni], ypath[2][ni], ypath[3][ni]])
                        x_path.append(arrx)
                        y_path.append(arry)

                    optimizer_kwargs.update({'path_x': np.array(x_path)})
                    optimizer_kwargs.update({'path_y': np.array(y_path)})
                    optimizer_kwargs.update({'path_Tzlist': Tzlist})
                    optimizer_kwargs.update({'path_redshifts': redshifts})
                else:
                    optimizer_kwargs = None

            if finite_source_magnification and compute_fluxes:
                fluxes = self._ray_trace_finite(x_opt, y_opt, xsrc, ysrc, system.multiplane, lensModel, kwargs_lens, res,
                                                source_shape, source_size_kpc, polar_grid, adaptive_grid=adaptive_grid,
                                                grid_rmax_scale=grid_rmax_scale)
            elif compute_fluxes:
                try:
                    fluxes = optimizer_kwargs['magnification_pointsrc']
                except:
                    fluxes = np.absolute(lensModel.magnification(x_opt, y_opt, kwargs_lens))
            else:
                fluxes = np.array([np.nan]*4)

            optimized_sys = self.update_system(lens_system=system,newkwargs=kwargs_lens, method='lenstronomy')

            new_data = Data(x_opt,y_opt,fluxes,None,[xsrc,ysrc])
            new_data.sort_by_pos(data2fit.x,data2fit.y)
            data.append(new_data)
            opt_sys.append(optimized_sys)

        return data, opt_sys, optimizer_kwargs, {'kwargs_lens': kwargs_lens, 'lensModel': lensModel,
                                                 'source_x': xsrc, 'source_y': ysrc}

    def _ray_trace_finite(self, ximg, yimg, xsrc, ysrc, multiplane, lensModel, kwargs_lens, resolution, source_shape,
                          source_size_kpc, polar_grid, adaptive_grid, grid_rmax_scale):

        source_scale = self.cosmo.kpc_per_asec(self.zsrc)
        source_size = source_size_kpc * source_scale ** -1
        #img_sep_small = min_img_sep(ximg, yimg)
        img_seps, theta_img = min_img_sep_ranked(ximg, yimg)

        raytracing = raytrace.RayTrace(xsrc=xsrc, ysrc=ysrc, multiplane=multiplane,
                                       res=resolution, source_shape=source_shape, polar_grid=polar_grid,
                                       source_size=source_size,
                                       minimum_image_sep=[img_seps, theta_img], adaptive_grid=adaptive_grid,
                                       grid_rmax_scale=grid_rmax_scale)

        fluxes = raytracing.magnification(ximg, yimg, lensModel, kwargs_lens)

        return fluxes

    def _solve_4imgs_lenstronomy(self, lens_systems, data2fit=None, method=str, sigmas=None, ray_trace=True, res=None,
                                 source_shape='GAUSSIAN', source_size_kpc=None, print_mag=False, raytrace_with=None, polar_grid=True,
                                 solver_type=None, N_iter_max=None, brightimg=None, adaptive_grid=False, grid_rmax_scale=1):

        data,opt_sys = [],[]

        lenstronomyWrap = LenstronomyWrap(cosmo=self.cosmo.astropy, z_source=self.zsrc)

        for i,system in enumerate(lens_systems):

            lensModel, kwargs_lens = lenstronomyWrap.get_lensmodel(system)

            kwargs_lens, precision = lenstronomyWrap.fit_lensmodel(data2fit.x,data2fit.y,lensModel,solver_type,kwargs_lens)

            xsrc, ysrc = lensModel.ray_shooting(data2fit.x, data2fit.y, kwargs_lens)
            xsrc, ysrc = np.mean(xsrc), np.mean(ysrc)

            x_img,y_img = lenstronomyWrap.solve_leq(xsrc,ysrc,lensModel,solver_type,kwargs_lens,brightimg)

            if print_mag:
                print('computing mag # '+str(i+1)+' of '+str(len(lens_systems)))

            source_scale = self.cosmo.kpc_per_asec(self.zsrc)
            source_size = source_size_kpc * source_scale ** -1
            img_sep_small = min_img_sep(x_img, y_img)

            raytracing = raytrace.RayTrace(xsrc=xsrc, ysrc=ysrc, multiplane=system.multiplane,
                                           source_size=source_size,
                                           res=res, source_shape=source_shape, polar_grid=polar_grid,
                                           minimum_image_sep=img_sep_small, adaptive_grid=adaptive_grid,
                                           grid_rmax_scale=grid_rmax_scale)

            fluxes = raytracing.magnification(x_img, y_img, lensModel, kwargs_lens)

            optimized_sys = self.update_system(lens_system=system, newkwargs=kwargs_lens, method='lenstronomy')

            new_data = Data(x_img,y_img,fluxes,None,[xsrc,ysrc])
            new_data.sort_by_pos(data2fit.x,data2fit.y)
            data.append(new_data)
            opt_sys.append(optimized_sys)

        return data,opt_sys

    def _optimize_4imgs_lensmodel(self, lens_systems=None, data2fit=[], method=str, sigmas=None, identifier='', opt_routine=None,
                                  ray_trace=True, return_positions=False, res=0.0005, source_shape='GAUSSIAN',
                                  source_size_kpc=float, print_mag=False, raytrace_with=None, polar_grid=False, solver_type=None, shr_coords=1):


        if sigmas is None:
            sigmas = [self.default_pos_sigma, self.default_flux_sigma, self.default_tdelay_sigma]

        lenstronomywrap = LenstronomyWrap(cosmo=self.cosmo.astropy, z_source=self.zsrc)
        d2fit = [data2fit.x,data2fit.y,data2fit.m,data2fit.t]
        optimized_systems = []

        if os.path.exists(self.paths.gravlens_input_path):
            pass
        else:
            create_directory(self.paths.gravlens_input_path)

        create_directory(self.paths.gravlens_input_path_dump)

        assert opt_routine is not None

        solver = GravlensInput(filename=identifier, zlens=self.zmain, zsrc=self.zsrc,
                               pos_sigma=sigmas[0], flux_sigma=sigmas[1], tdelay_sigma=sigmas[2],
                               identifier=identifier, paths=self.paths, cosmology=self.cosmo,shr_coords=shr_coords)

        for system in lens_systems:

            full = FullModel(multiplane=system.multiplane)

            for i, model in enumerate(system.lens_components):
                full.populate(SingleModel(lensmodel=model, shr_coords=shr_coords))

            solver.add_lens_system(full)

        outputfile = solver.write_all(data=d2fit, zlens=self.zmain, zsrc=self.zsrc, opt_routine=opt_routine,shr_coords=shr_coords)

        call_lensmodel(inputfile=solver.outfile_path + solver.filename + '.txt',
                       path_2_lensmodel=self.paths.path_2_lensmodel)

        lensdata = []

        for i, name in enumerate(outputfile):

            xvals, yvals, mag_gravlens, tvals, macrovals, srcvals = read_dat_file(fname=name)

            lensdata.append(Data(x=xvals, y=yvals, m=mag_gravlens, t=tvals, source=srcvals))

            newmacromodel = gravlens_to_kwargs(macrovals,shr_coords=shr_coords)

            optimized_sys = self.update_system(lens_system=lens_systems[i], newkwargs=newmacromodel, method='lensmodel')

            optimized_systems.append(optimized_sys)

        if ray_trace:

            for i, name in enumerate(outputfile):

                if print_mag:
                    print('computing mag #: ', i + 1)

                source_scale = self.cosmo.kpc_per_asec(self.zsrc)
                source_size = source_size_kpc * source_scale ** -1

                raytracing = raytrace.RayTrace(xsrc=lensdata[i].srcx, ysrc=lensdata[i].srcy, multiplane=optimized_systems[i].multiplane,
                                               source_size=source_size,
                                               res=res, source_shape=source_shape, polar_grid=polar_grid)
                lensModel, kwargs_lens = lenstronomywrap.get_lensmodel(optimized_systems[i])
                fluxes = raytracing.magnification(lensdata[i].x, lensdata[i].y, lensModel, kwargs_lens)
                lensdata[i].set_mag(fluxes)

        for dataset in lensdata:
            dataset.sort_by_pos(data2fit.x, data2fit.y)

        if self.clean_up:
            delete_dir(self.paths.gravlens_input_path_dump)

        return lensdata, optimized_systems

    def _solve_4imgs(self, lens_systems=None, method=str, identifier='', srcx=None, srcy=None,
                     res=0.001, source_shape='GAUSSIAN', ray_trace=True, source_size_kpc=float, print_mag=False,
                     raytrace_with='', polar_grid=True, arrival_time=False, shr_coords=1,brightimg=None, adaptive_grid=False):

        lenstronomywrap = LenstronomyWrap(cosmo=self.cosmo.astropy,z_source=self.zsrc)

        if method == 'lensmodel':

            create_directory(self.paths.gravlens_input_path_dump)

            data = []

            solver = GravlensInput(filename=identifier, zlens=self.zmain, zsrc=self.zsrc, identifier=identifier,
                                   paths=self.paths,
                                   cosmology=self.cosmo,shr_coords=shr_coords)

            for i, system in enumerate(lens_systems):
                full = FullModel(multiplane=system.multiplane)
                for model in system.lens_components:
                    full.populate(SingleModel(lensmodel=model,shr_coords=shr_coords))
                solver.add_lens_system(full)

            outputfile = solver.write_all(data=None, zlens=self.zmain, zsrc=self.zsrc, srcx=srcx, srcy=srcy,shr_coords=shr_coords)

            call_lensmodel(inputfile=solver.outfile_path + solver.filename + '.txt',
                           path_2_lensmodel=self.paths.path_2_lensmodel)

            lens_data = read_gravlens_out(fnames=outputfile)
            t0 = time()

            for i, system in enumerate(lens_systems):

                x, y, m, t, nimg = lens_data[i]

                data.append(Data(x=x, y=y, m=m, t=t, source=[srcx, srcy]))

                if ray_trace:

                    if print_mag:
                        print('computing mag #: ', i + 1)

                    source_scale = self.cosmo.kpc_per_asec(self.zsrc)
                    source_size = source_size_kpc * source_scale ** -1
                    min_img_separation = min_img_sep(data[i].x,data[i].y)

                    raytracing = raytrace.RayTrace(xsrc=data[i].srcx, ysrc=data[i].srcy,
                                                   multiplane=system.multiplane,source_size=source_size,
                                                   res=res, source_shape=source_shape, polar_grid=polar_grid,
                                                   minimum_image_sep=min_img_separation, adaptive_grid=adaptive_grid)
                    lensModel, kwargs_lens = lenstronomywrap.get_lensmodel(system)
                    fluxes = raytracing.magnification(data[i].x, data[i].y, lensModel, kwargs_lens)

                    data[i].set_mag(fluxes)

            if self.clean_up:
                delete_dir(self.paths.gravlens_input_path_dump)

            return data

        elif method == 'lenstronomy':

            data = []

            lenstronomyWrap = LenstronomyWrap(cosmo=self.cosmo.astropy, z_source=self.zsrc)

            for i, system in enumerate(lens_systems):

                lensModel, kwargs_lens = lenstronomywrap.get_lensmodel(system)

                x_image,y_image = lenstronomyWrap.solve_leq(srcx,srcy,lensModel,kwargs_lens,brightimg)

                source_scale = self.cosmo.kpc_per_asec(self.zsrc)
                source_size = source_size_kpc * source_scale ** -1
                min_img_separation = min_img_sep_ranked(x_image, y_image)

                raytracing = raytrace.RayTrace(xsrc=srcx, ysrc=srcy,
                                               multiplane=system.multiplane, source_size=source_size,
                                               res=res, source_shape=source_shape, polar_grid=polar_grid,
                                               minimum_image_sep=min_img_separation, adaptive_grid=adaptive_grid)

                fluxes = raytracing.magnification(x_image,y_image, lensModel, kwargs_lens)

                if arrival_time:

                    if system.multiplane:
                        arrival_times = lensModel.arrival_time(x_image,y_image,kwargs_lens)

                    else:
                        arrival_times = [0,0,0,0]
                        #raise Exception('arrival times not yet implemented for single plane')
                        #fermat_potential = lensModel.fermat_potential(x_image,y_image,x_source=srcx,y_source=srcy,kwargs_lens=lensmodel_params)


                    arrival_times -= np.min(arrival_times)
                else:
                    arrival_times = [0,0,0,0]

                data.append(Data(x=x_image,y=y_image,m=fluxes,t=arrival_times,source=[srcx,srcy]))

            return data
