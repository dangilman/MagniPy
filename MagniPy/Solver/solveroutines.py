from MagniPy.magnipy import Magnipy
import numpy as np
from MagniPy.LensBuild.defaults import raytrace_with_default,default_sigmas,default_gridrmax,default_res,\
    default_source_shape,default_source_size,default_solve_method,default_file_identifier,get_default_SIE_random
import copy
from MagniPy.Solver.LenstronomyWrap.lenstronomy_wrap import LenstronomyWrap

class SolveRoutines(Magnipy):
    """
    This class uses the routines set up in MagniPy to solve the lens equation in various ways with lenstronomy or lensmodel
    """

    def initialzed_over_param_range(self,macromodels=[], datatofit=None, multiplane=None,sigmas=None,
                              identifier=None, grid_rmax=None, res=None,
                              source_shape='GAUSSIAN', source_size=None,param_name=None):

        macromodel_list = []

        param_values = []

        assert param_name in ['gamma']

        for macromodel in macromodels:

            _, macro = self.macromodel_initialize(macromodel,datatofit,multiplane,method='lensmodel',sigmas=sigmas,identifier=identifier,
                                       grid_rmax=grid_rmax,res=res,source_shape=source_shape,source_size=source_size)

            macromodel_list.append(macro)
            param_values.append(macromodel.lenstronomy_args['gamma'])

        return param_values,macromodel_list

    def optimize_4imgs_lenstronomy(self,lens_systems=None,macromodel=None,datatofit=None,realizations=None, multiplane = None, sigmas=None,
                      grid_rmax=None, res=None,source_shape='GAUSSIAN', source_size=None,raytrace_with=None,
                      polar_grid=True,initialize = False,init_macromodel=None,identifier=None,solver_type='PROFILE_SHEAR',n_particles=50,
                      n_iterations=50,tol_source=5e-4,tol_centroid=0.05,tol_mag=0.2,centroid_0=[0,0],method='PS',refit=False,
                                   optimizer_routine='optimize_SIE_shear',interpolate=False):

        if raytrace_with is None:
            raytrace_with = raytrace_with_default

        if grid_rmax is None:
            grid_rmax = default_gridrmax(srcsize=source_size)

        if source_shape is None:
            source_shape = default_source_shape

        if source_size is None:
            source_size = default_source_size

        if res is None:
            res = default_res(source_size)

        if identifier is None:
            identifier = default_file_identifier

        if lens_systems is None:

            if initialize:

                if init_macromodel is None:

                    if isinstance(macromodel,list):
                        _, init_macromodel = self.macromodel_initialize(macromodel[0], datatofit, False, method='lensmodel', sigmas=sigmas,
                                      identifier=identifier, grid_rmax=grid_rmax, res=res,
                                      source_shape=source_shape, source_size=source_size, print_mag=False,
                                      solver_type='PROFILE_SHEAR',shr_coords=1)


                    else:
                        _, init_macromodel = self.macromodel_initialize(macromodel, datatofit, False, method='lensmodel', sigmas=sigmas,
                                      identifier=identifier, grid_rmax=grid_rmax, res=res,
                                      source_shape=source_shape, source_size=source_size, print_mag=False,
                                      solver_type='PROFILE_SHEAR',shr_coords=1)

                macromodel = init_macromodel

            lens_systems = []

            if isinstance(macromodel,list):

                assert len(macromodel) == len(realizations), 'if macromodel is a list, must have same number of elements as realizations'

                for macro,real in zip(macromodel,realizations):
                    lens_systems.append(self.build_system(main=macro,additional_halos=real,multiplane=multiplane))
            else:
                if realizations is not None:
                    for real in realizations:
                        lens_systems.append(self.build_system(main=copy.deepcopy(macromodel),additional_halos=real,
                                                              multiplane=multiplane))
                else:
                    lens_systems.append(self.build_system(main=copy.deepcopy(macromodel),multiplane=multiplane))


        optimized_data, model = self._optimize_4imgs_lenstronomy(lens_systems=lens_systems, data2fit=datatofit, tol_source=tol_source,
                                  tol_mag=tol_mag, tol_centroid=tol_centroid,centroid_0=centroid_0, n_particles=n_particles,
                                  n_iterations=n_iterations,grid_rmax=grid_rmax, res=res,
                                 source_size=source_size,raytrace_with=raytrace_with,initialized=initialize,
                                 source_shape=source_shape,polar_grid=polar_grid, solver_type=solver_type,method=method,refit=refit,
                                 optimizer_routine=optimizer_routine,interpolate=interpolate)

        return optimized_data,model



    def solve_lens_equation(self, full_system=None, macromodel=None, realizations=None, multiplane=None, method=None,
                            ray_trace=True, identifier=None, srcx=None, srcy=None, grid_rmax=None, res=None,
                            source_shape='GAUSSIAN', source_size=None, sort_by_pos=None, filter_subhalos=False,
                            filter_by_pos=False, filter_kwargs={},raytrace_with=None,polar_grid=True,shr_coords=1):

        if raytrace_with is None:
            raytrace_with = raytrace_with_default

        if grid_rmax is None:
            grid_rmax = default_gridrmax(srcsize=source_size)

        if source_shape is None:
            source_shape = default_source_shape

        if source_size is None:
            source_size = default_source_size

        if res is None:
            res = default_res(source_size)

        if method is None:
            method = default_solve_method

        if identifier is None:
            identifier = default_file_identifier

        lens_systems = []

        if full_system is None:
            assert macromodel is not None
            if realizations is not None:
                for real in realizations:
                    lens_systems.append(self.build_system(main=macromodel, additional_halos=real, multiplane=multiplane))
            else:
                lens_systems.append(self.build_system(main=copy.deepcopy(macromodel),additional_halos=None,multiplane=multiplane))

        else:

            lens_systems.append(copy.deepcopy(full_system))

        assert method is not None
        assert method in ['lensmodel', 'lenstronomy']

        data = self._solve_4imgs(lens_systems=lens_systems, method=method, identifier=identifier, srcx=srcx, srcy=srcy,
                                 grid_rmax=grid_rmax, res=res, source_shape=source_shape, ray_trace=ray_trace,
                                 raytrace_with=raytrace_with, source_size=source_size, polar_grid=polar_grid, shr_coords=shr_coords)

        if sort_by_pos is not None:
            data[0].sort_by_pos(sort_by_pos.x,sort_by_pos.y)
        return data


    def two_step_optimize(self, macromodel=None, datatofit=None, realizations=None, multiplane=False, method=None, ray_trace=None, sigmas=None,
                          identifier=None, srcx=None, srcy=None, grid_rmax=None, res=None,
                          source_shape='GAUSSIAN', source_size=None, print_mag=False, raytrace_with=None,
                          filter_by_position=False, polar_grid=True, filter_kwargs={},solver_type='PROFILE_SHEAR',
                          N_iter_max=100,shr_coords=1):

        # optimizes the macromodel first, then uses it to optimize with additional halos in the lens model

        assert datatofit is not None

        if sigmas is None:
            sigmas = default_sigmas

        if raytrace_with is None:
            raytrace_with = raytrace_with_default

        if grid_rmax is None:
            grid_rmax = default_gridrmax(srcsize=source_size)

        if source_shape is None:
            source_shape = default_source_shape

        if source_size is None:
            source_size = default_source_size

        if res is None:
            res = default_res(source_size)

        if method is None:
            method = default_solve_method

        if identifier is None:
            identifier = default_file_identifier

        if method == 'lensmodel':
            _,macromodel_init = self.macromodel_initialize(macromodel=copy.deepcopy(macromodel), datatofit=datatofit,
                                                           multiplane=multiplane, method='lensmodel', sigmas=sigmas,
                                                           identifier=identifier,grid_rmax=grid_rmax, res=res,
                                                           source_shape=source_shape, source_size=source_size, print_mag=print_mag,
                                                           shr_coords=shr_coords)
            optimized_data, newsystem = self.fit(macromodel=macromodel_init, datatofit=datatofit,
                                                 realizations=realizations, multiplane=multiplane,
                                                 ray_trace=ray_trace, sigmas=sigmas, identifier=identifier, srcx=srcx,
                                                 srcy=srcy, grid_rmax=grid_rmax, res=res,method='lensmodel',
                                                 source_shape=source_shape, source_size=source_size,
                                                 print_mag=print_mag, raytrace_with=raytrace_with,
                                                 filter_by_position=filter_by_position, polar_grid=polar_grid,
                                                 filter_kwargs=filter_kwargs,shr_coords=shr_coords)
        else:

            optimized_data, newsystem = self.fit(macromodel=macromodel, datatofit=datatofit,
                                                 realizations=realizations, multiplane=multiplane, method=method,
                                                 ray_trace=ray_trace, sigmas=sigmas, identifier=identifier, srcx=srcx,
                                                 srcy=srcy, grid_rmax=grid_rmax, res=res,
                                                 source_shape=source_shape, source_size=source_size,
                                                 print_mag=print_mag, raytrace_with=raytrace_with,
                                                 filter_by_position=filter_by_position, polar_grid=polar_grid,
                                                 filter_kwargs=filter_kwargs, solver_type=solver_type,N_iter_max=N_iter_max)




        return optimized_data,newsystem

    def macromodel_initialize(self, macromodel, datatofit, multiplane, method=None, sigmas=None,
                              identifier=None, grid_rmax=None, res=None,
                              source_shape='GAUSSIAN', source_size=None, print_mag=False,
                              solver_type=None,shr_coords=1):

        # fits just a single macromodel profile to the data

        assert method is not None
        assert method in ['lensmodel','lenstronomy']

        if sigmas is None:
            sigmas = default_sigmas

        if method is None:
            method = default_solve_method

        lens_systems = []

        lens_systems.append(self.build_system(main=copy.deepcopy(macromodel), additional_halos=None, multiplane=multiplane))

        optimized_data, model = self._optimize_4imgs_lensmodel(lens_systems=lens_systems, data2fit=datatofit, method=method,
                                                               sigmas=sigmas, identifier=identifier, grid_rmax=grid_rmax,
                                                               res=res, source_shape=source_shape, ray_trace=False,
                                                               source_size=source_size, print_mag=print_mag, opt_routine='randomize',
                                                               solver_type=solver_type, shr_coords=shr_coords)

        newmacromodel = model[0].lens_components[0]

        return optimized_data,newmacromodel

    def fit(self, macromodel=None, datatofit=None, realizations=None, multiplane = None, method=None, ray_trace=True, sigmas=None,
                      identifier=None, srcx=None, srcy=None, grid_rmax=None, res=None,
                      source_shape='GAUSSIAN', source_size=None, print_mag=False, raytrace_with=None, filter_by_position=False,
                      polar_grid=True,filter_kwargs={},which_chi = 'src',solver_type='PROFILE_SHEAR',N_iter_max=100,shr_coords=1):

        # uses source plane chi^2


        if which_chi == 'src':
            basic_or_full = 'basic'
        else:
            basic_or_full = 'full'
        if raytrace_with is None:
            raytrace_with = raytrace_with_default

        if grid_rmax is None:
            grid_rmax = default_gridrmax(srcsize=source_size)

        if source_shape is None:
            source_shape = default_source_shape

        if source_size is None:
            source_size = default_source_size

        if res is None:
            res = default_res(source_size)

        if method is None:
            method = default_solve_method

        if sigmas is None:
            sigmas = default_sigmas

        if method is None:
            method = default_solve_method

        if identifier is None:
            identifier = default_file_identifier

        lens_systems= []

        ################################################################################

        # If macromodel is a list same length as realizations, build the systems and fit each one

        if isinstance(macromodel,list):

            assert len(macromodel) == len(realizations), 'if macromodel is a list, must have same number of elements as realizations'

            for macro,real in zip(macromodel,realizations):
                lens_systems.append(self.build_system(main=macro,additional_halos=real,multiplane=multiplane))
        else:
            if realizations is not None:
                for real in realizations:
                    lens_systems.append(self.build_system(main=copy.deepcopy(macromodel),additional_halos=real,
                                                          multiplane=multiplane))
            else:
                lens_systems.append(self.build_system(main=copy.deepcopy(macromodel),multiplane=multiplane))

        if method == 'lenstronomy':

            optimized_data, model = self.optimize_4imgs_lenstronomy(lens_systems=lens_systems,datatofit=datatofit,
                               multiplane = multiplane, grid_rmax=grid_rmax, res=res,source_shape=source_shape,
                               source_size=source_size,raytrace_with=raytrace_with,polar_grid=polar_grid,initialize = False,
                               solver_type=solver_type,n_particles=40,n_iterations=200,tol_source=1e-3,tol_centroid=0.05,
                               tol_mag=0.2,centroid_0=[0,0])

        else:
            optimized_data, model = self._optimize_4imgs_lensmodel(lens_systems=lens_systems, data2fit=datatofit,
                                                                   method=method,
                                                                   sigmas=sigmas, identifier=identifier,
                                                                   grid_rmax=grid_rmax,
                                                                   res=res, source_shape=source_shape,
                                                                   ray_trace=ray_trace,
                                                                   source_size=source_size, print_mag=print_mag,
                                                                   opt_routine=basic_or_full,
                                                                   raytrace_with=raytrace_with, polar_grid=polar_grid,
                                                                   solver_type=solver_type, shr_coords=shr_coords)

        return optimized_data,model

    def _trace(self, lens_system, data=None, multiplane=None, grid_rmax=None, res=None, source_shape=None, source_size=None,
               raytrace_with=None, srcx=None, srcy=None, build=False, realizations=None):

        if raytrace_with is None:
            raytrace_with = raytrace_with_default

        if grid_rmax is None:
            grid_rmax = default_gridrmax(srcsize=source_size)

        if source_shape is None:
            source_shape = default_source_shape

        if source_size is None:
            source_size = default_source_size

        if res is None:
            res = default_res(source_size)

        if build:
            lens_system = self.build_system(main=lens_system,realizations=realizations,multiplane=multiplane)

        if raytrace_with == 'lensmodel':

            xsrc,ysrc = data.srcx,data.srcy

            if srcx is not None:
                xsrc = srcx
            if srcy is not None:
                ysrc = srcy

            fluxes = self.do_raytrace_lensmodel(lens_system=lens_system,xpos=data.x,ypos=data.y,xsrc=xsrc,ysrc=ysrc,
                                    multiplane=multiplane,grid_rmax=grid_rmax,res=res,source_shape=source_shape,source_size=source_size,
                                       cosmology=self.cosmo,zsrc=self.cosmo.zsrc,raytrace_with=raytrace_with)

        elif raytrace_with == 'lenstronomy':

            lenstronomy_wrap = LenstronomyWrap(multiplane=multiplane,cosmo=self.cosmo,z_source=self.cosmo.zsrc)
            lenstronomy_wrap.assemble(lens_system)

            fluxes = self.do_raytrace_lenstronomy(lenstronomy_wrap,data.x,data.y,multiplane=multiplane,gridsize=grid_rmax,res=res,
                                         source_shape=source_shape,source_size=source_size,cosmology=self.cosmo.cosmo,zsrc=self.cosmo.zsrc)

        return fluxes*max(fluxes)**-1







