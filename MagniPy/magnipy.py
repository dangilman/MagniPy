from time import time
from MagniPy import paths
import MagniPy.LensBuild.lens_assemble as build
import MagniPy.LensBuild.renderhalos as halo_gen
from MagniPy.LensBuild.Cosmology.cosmology import Cosmo
from MagniPy.Solver.GravlensWrap._call import *
from MagniPy.Solver.GravlensWrap.generate_input import *
from MagniPy.Solver.GravlensWrap.gravlens_to_kwargs import gravlens_to_kwargs
from MagniPy.Solver.LenstronomyWrap.lenstronomy_wrap import *
from MagniPy.Solver.RayTrace import raytrace
from MagniPy.lensdata import Data
import os
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

class Magnipy:
    """
    This class is used to specify a lens system and solve the lens equation or optimize a lens model
    """

    def __init__(self, zlens, zsrc, clean_up=True, temp_folder=None):

        self.zmain = zlens
        self.zsrc = zsrc
        self.lens_halos = halo_gen.HaloGen(zd=zlens, zsrc=zsrc)

        self.cosmo = Cosmo(zd=zlens, zsrc=zsrc)
        self.clean_up = clean_up

        self.paths = paths

        if temp_folder is None:
            self.temp_folder = 'temp/'
        else:
            self.temp_folder = temp_folder + '/'

        self.paths.gravlens_input_path_dump = self.paths.gravlens_input_path + self.temp_folder

    def lenstronomy_build(self):

        return LenstronomyWrap(cosmo=self.cosmo.cosmo, z_source=self.zsrc)

    def print_system(self, system_index, component_index=None):

        system = self.lens_system[system_index]
        if component_index is not None:
            system.lens_components[component_index].print_args()
        else:
            for component in system.lens_components:
                component.print_args()

    def update_system(self, lens_system, newkwargs, method, solver_type):

        if lens_system.lens_components[0].other_args['name'] == 'SERSIC_NFW':
            raise Exception('not yet implemented')
        else:

            if method == 'lenstronomy':

                lens_system.lens_components[0].update(method=method, is_shear=False, **newkwargs[0])

                if solver_type == 'PROFILE_SHEAR':

                    lens_system.lens_components[0].update(method=method, is_shear=True, **newkwargs[1])

            elif method == 'lensmodel':

                for val in [True,False]:
                    lens_system.lens_components[0].update(method=method, is_shear=val, **newkwargs)

            return lens_system


    def build_system(self, main=None, additional_halos=None, multiplane=None):

        """
        This routine sets up the lens model. It assembles a "system", which is a list of Deflector instances
        :param main: instance of Deflector for the main deflector
        :param halo_model: specifies the halo content of the deflector
        :param realization: specifies a specific realization; suborinate to halo_model
        :param multiplane: whether halo_model includes multiple lens planes
        :return:
        """

        assert multiplane is not None

        newsystem = build.LensSystem(multiplane=multiplane)

        if main is not None:
            newsystem.main_lens(main)

        if additional_halos is not None:
            newsystem.halos(additional_halos)

        return newsystem

    def build_system_fromlists(self,redshift_list=[],lens_list=[],arg_list=[],multiplane=None,zsrc=None):

        assert len(redshift_list) == len(lens_list) == len(arg_list)

        from MassModels.ExternalShear import Shear
        from MassModels.SIE import SIE
        from MassModels.NFW import NFW
        from MassModels.TNFW import TNFW
        halos = []
        main = None

        for i in range(0,len(redshift_list)):

            name = lens_list[i]
            z = redshift_list[i]
            args = arg_list[i]

            if name == 'SPEMD':

                if lens_list[i+1] == 'SHEAR':

                    shear_mag,shear_pa = cart_to_polar(arg_list[i+1]['e1'],arg_list[i+1]['e2'])

                    main = build.Deflector(subclass=SIE(),redshift=z,lens_params=args,shear=shear_mag,shear_theta=shear_pa,name=name)
                else:
                    main = build.Deflector(subclass=SIE(), redshift=z, lens_params=args, shear=shear_mag,
                                           shear_theta=shear_pa,name=name)

            elif name=='SHEAR':
                continue

            elif name=='NFW':
                halo = build.Deflector(subclass=NFW(z,zsrc),redshift=z,lens_params=args,name=name)
                halos.append(halo)

            elif name == 'TNFW':
                halo = build.Deflector(subclass=TNFW(z, zsrc), redshift=z, lens_params=args,name=name)
                halos.append(halo)

        return self.build_system(main=main,additional_halos=halos,multiplane=multiplane)

    def _optimize_4imgs_lenstronomy(self, lens_systems, data2fit=None,tol_source=None,tol_mag=None,tol_centroid=None,
                                    centroid_0=None,n_particles=50,n_iterations=400,
                                    grid_rmax=None, res=None,source_shape='GAUSSIAN', source_size=None,raytrace_with='lenstronomy',
                                    polar_grid=True,solver_type='PROFILE_SHEAR',optimizer_routine=str,verbose=bool,re_optimize=False,
                                    particle_swarm = True, restart = 1,constrain_params=None,shifting_background=False):

        data, opt_sys = [], []

        lenstronomyWrap = LenstronomyWrap(cosmo=self.cosmo.cosmo, z_source=self.zsrc)

        for i, system in enumerate(lens_systems):

            kwargs_lens, [xsrc,ysrc], [x_opt,y_opt],lensModel,optimizer = lenstronomyWrap.run_optimize(system,self.zsrc,data2fit.x,
                            data2fit.y,tol_source,data2fit.m,tol_mag,tol_centroid,centroid_0,optimizer_routine,self.zmain,
                            n_particles,n_iterations,verbose,restart,re_optimize,particle_swarm,constrain_params,shifting_background)

            #ray_shooting_function = optimizer.ray_shooting_function
            fluxes = self.do_raytrace(x_opt, y_opt, lensmodel=lensModel, xsrc=xsrc, ysrc=ysrc,
                                      multiplane=system.multiplane, grid_rmax=grid_rmax,
                                      res=res, source_shape=source_shape, source_size=source_size,
                                      raytrace_with=raytrace_with, lens_model_params=kwargs_lens, polar_grid=polar_grid)

            optimized_sys = self.update_system(lens_system=system,newkwargs=kwargs_lens, method='lenstronomy',solver_type=solver_type)


            new_data = Data(x_opt,y_opt,fluxes,None,[xsrc,ysrc])
            new_data.sort_by_pos(data2fit.x,data2fit.y)
            data.append(new_data)
            opt_sys.append(optimized_sys)

        return data,opt_sys

    def _solve_4imgs_lenstronomy(self, lens_systems, data2fit=None, method=str, sigmas=None, ray_trace=True, grid_rmax=None, res=None,
                                 source_shape='GAUSSIAN', source_size=None, print_mag=False, raytrace_with=None, polar_grid=True,
                                 solver_type=None, N_iter_max=None):

        data,opt_sys = [],[]

        lenstronomyWrap = LenstronomyWrap(cosmo=self.cosmo.cosmo, z_source=self.zsrc)

        for i,system in enumerate(lens_systems):

            redshift_list, lens_list, lensmodel_params = system.lenstronomy_lists()

            lensModel = LensModelExtensions(lens_model_list=lens_list, multi_plane=system.multiplane,
                                       redshift_list=redshift_list, z_source=self.zsrc)

            kwargs_lens, precision = lenstronomyWrap.fit_lensmodel(data2fit.x,data2fit.y,lensModel,solver_type,lensmodel_params)

            xsrc, ysrc = lensModel.ray_shooting(data2fit.x, data2fit.y, kwargs_lens)
            xsrc, ysrc = np.mean(xsrc), np.mean(ysrc)

            x_img,y_img = lenstronomyWrap.solve_leq(xsrc,ysrc,lensModel,solver_type,kwargs_lens)

            if print_mag:
                print('computing mag # '+str(i+1)+' of '+str(len(lens_systems)))

            fluxes = self.do_raytrace(x_img,y_img,lensmodel=lensModel,xsrc=xsrc,ysrc=ysrc,multiplane=system.multiplane,grid_rmax=grid_rmax,
                                          res=res,source_shape=source_shape,source_size=source_size,
                                          raytrace_with=raytrace_with,lens_model_params=kwargs_lens,polar_grid=polar_grid)

            optimized_sys = self.update_system(lens_system=system, newkwargs=kwargs_lens, method='lenstronomy',
                                               solver_type=solver_type)

            new_data = Data(x_img,y_img,fluxes,None,[xsrc,ysrc])
            new_data.sort_by_pos(data2fit.x,data2fit.y)
            data.append(new_data)
            opt_sys.append(optimized_sys)

        return data,opt_sys

    def _optimize_4imgs_lensmodel(self, lens_systems=None, data2fit=[], method=str, sigmas=None, identifier='', opt_routine=None,
                                  ray_trace=True, return_positions=False, grid_rmax=int, res=0.0005, source_shape='GAUSSIAN',
                                  source_size=float, print_mag=False, raytrace_with=None, polar_grid=False, solver_type=None, shr_coords=1):


        if sigmas is None:
            sigmas = [self.default_pos_sigma, self.default_flux_sigma, self.default_tdelay_sigma]

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
                full.populate(SingleModel(lensmodel=model, units=system.units,shr_coords=shr_coords))

            solver.add_lens_system(full)

        outputfile = solver.write_all(data=d2fit, zlens=self.zmain, zsrc=self.zsrc, opt_routine=opt_routine,shr_coords=shr_coords)

        call_lensmodel(inputfile=solver.outfile_path + solver.filename + '.txt',
                       path_2_lensmodel=self.paths.path_2_lensmodel)

        lensdata = []

        for i, name in enumerate(outputfile):

            xvals, yvals, mag_gravlens, tvals, macrovals, srcvals = read_dat_file(fname=name)

            lensdata.append(Data(x=xvals, y=yvals, m=mag_gravlens, t=tvals, source=srcvals))

            newmacromodel = gravlens_to_kwargs(macrovals,shr_coords=shr_coords)

            optimized_sys = self.update_system(lens_system=lens_systems[i], newkwargs=newmacromodel, method='lensmodel',
                                               solver_type=solver_type)

            optimized_systems.append(optimized_sys)

        if ray_trace:

            for i, name in enumerate(outputfile):

                if print_mag:
                    print('computing mag #: ', i + 1)

                if raytrace_with == 'lenstronomy':
                    redshift_list, lens_list, lensmodel_params = optimized_systems[i].lenstronomy_lists()
                    lensModel = LensModelExtensions(lens_model_list=lens_list, multi_plane=system.multiplane,
                                                    redshift_list=redshift_list, z_source=self.zsrc)
                else:
                    lensModel = None
                    lensmodel_params = None

                fluxes = self.do_raytrace(lensdata[i].x, lensdata[i].y,lens_system=optimized_systems[i],lensmodel=lensModel,xsrc=lensdata[i].srcx,
                                          ysrc=lensdata[i].srcy,multiplane=system.multiplane, grid_rmax=grid_rmax,
                                          res=res, source_shape=source_shape, source_size=source_size,
                                          raytrace_with=raytrace_with,polar_grid=polar_grid,lens_model_params=lensmodel_params)

                lensdata[i].set_mag(fluxes)

        for dataset in lensdata:
            dataset.sort_by_pos(data2fit.x, data2fit.y)

        if self.clean_up:
            delete_dir(self.paths.gravlens_input_path_dump)

        return lensdata, optimized_systems

    def _solve_4imgs(self, lens_systems=None, method=str, identifier='', srcx=None, srcy=None, grid_rmax=.1,
                     res=0.001, source_shape='GAUSSIAN', ray_trace=True, source_size=float, print_mag=False,
                     raytrace_with='', polar_grid=True, arrival_time=False, shr_coords=1):

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

                    redshift_list, lens_list, lensmodel_params = system.lenstronomy_lists()
                    lensModel = LensModelExtensions(lens_model_list=lens_list, multi_plane=system.multiplane,
                                                    redshift_list=redshift_list, z_source=self.zsrc)

                    fluxes = self.do_raytrace(lens_system=system, xpos=data[i].x, ypos=data[i].y, xsrc=srcx,
                                                        ysrc=srcy, multiplane=lens_systems[i].multiplane,
                                                        grid_rmax=grid_rmax,res=res, source_shape=source_shape,lensmodel=lensModel,
                                                    source_size=source_size,raytrace_with=raytrace_with,polar_grid=polar_grid,
                                              lens_model_params=lensmodel_params)

                    data[i].set_mag(fluxes)

            if self.clean_up:
                delete_dir(self.paths.gravlens_input_path_dump)

            return data

        elif method == 'lenstronomy':

            data = []

            lenstronomyWrap = LenstronomyWrap(cosmo=self.cosmo.cosmo, z_source=self.zsrc)

            for i, system in enumerate(lens_systems):

                redshift_list, lens_list, lensmodel_params = system.lenstronomy_lists()

                lensModel = LensModelExtensions(lens_model_list=lens_list, multi_plane=system.multiplane,
                                                redshift_list=redshift_list, z_source=self.zsrc)

                x_image,y_image = lenstronomyWrap.solve_leq(srcx,srcy,lensModel,lensmodel_params)

                fluxes = None
                if ray_trace:

                    fluxes = self.do_raytrace(x_image,y_image,xsrc=srcx,ysrc=srcy,multiplane=system.multiplane,lens_system=system,lensmodel=lensModel,grid_rmax=grid_rmax,
                                              res=res,source_shape=source_shape,source_size=source_size,raytrace_with=raytrace_with,
                                              lens_model_params=lensmodel_params,polar_grid=polar_grid)

                if arrival_time:

                    if system.multiplane:
                        arrival_times = lensModel.arrival_time(x_image,y_image,lensmodel_params)

                    else:
                        arrival_times = [0,0,0,0]
                        #raise Exception('arrival times not yet implemented for single plane')
                        #fermat_potential = lensModel.fermat_potential(x_image,y_image,x_source=srcx,y_source=srcy,kwargs_lens=lensmodel_params)


                    arrival_times -= np.min(arrival_times)
                else:
                    arrival_times = [0,0,0,0]

                data.append(Data(x=x_image,y=y_image,m=fluxes,t=arrival_times,source=[srcx,srcy]))

            return data

    def do_raytrace(self, xpos, ypos, lens_system=None,lensmodel=None,xsrc=float, ysrc=float, multiplane=None,
                    grid_rmax=None,res=None, source_shape=None, source_size=None,
                              raytrace_with=None,polar_grid=None,lens_model_params=None):

        if raytrace_with == 'lenstronomy':

            assert lensmodel is not None

            #fluxes = lenstronomy_wrap.compute_mags(xpos,ypos,lensmodel,lens_model_params,source_size,
            #                                       2*grid_rmax,2*grid_rmax*res**-1,source_shape)
            ray_shooter = raytrace.RayTrace(xsrc=xsrc, ysrc=ysrc, multiplane=multiplane,
                                                grid_rmax=grid_rmax, res=res, source_shape=source_shape,
                                                source_size=source_size, cosmology=self.cosmo, zsrc=self.zsrc,
                                                raytrace_with=raytrace_with, polar_grid=polar_grid)

            fluxes = ray_shooter.compute_mag(xpos, ypos,lensmodel=lensmodel,lens_model_params=lens_model_params)

        else:

            assert lens_system is not None

            ray_shooter = raytrace.RayTrace(xsrc=xsrc, ysrc=ysrc, multiplane=multiplane,
                                                grid_rmax=grid_rmax, res=res, source_shape=source_shape,
                                                source_size=source_size, cosmology=self.cosmo, zsrc=self.zsrc,
                                                raytrace_with=raytrace_with, polar_grid=polar_grid)

            fluxes = ray_shooter.compute_mag(xpos, ypos, lens_system=lens_system)

        return fluxes
